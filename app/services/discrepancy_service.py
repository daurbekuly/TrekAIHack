"""Inventory discrepancy detection: estimate unrecorded ("shadow") sales.

SME shops routinely sell goods without ringing them through the POS. If the
forecasting model learns only from POS data it underestimates demand and the
shop runs out of stock. This module reconciles physical stock movement against
recorded sales and folds the gap back into the daily sales series.

Reconciliation, per product, between two physical counts (t0, t1]:

    actual_sold   = start_count + inbound_invoiced - end_count
    shadow_sales  = actual_sold - pos_sold

The window is bounded by real count timestamps, not by the nominal period
boundaries: inventory is only *known* at the moment it was counted, so the
inbound and POS sums must cover exactly the same interval.

Layout:
    load_*                    -- thin SQLAlchemy loaders (I/O only)
    select_count_windows,
    reconcile,
    distribute_shadow_sales,
    compute_shadow_sales      -- pure pandas, unit-testable without a database
    analyze_shadow_sales,
    calculate_shadow_sales    -- public entry points (load + compute)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import InboundInvoice, InventoryCount, InvoiceItem, SalesHistory

AllocationMethod = Literal["proportional", "sample"]

WEEKEND_DAYS = (5, 6)  # Saturday, Sunday (pandas: Monday == 0)
DEFAULT_WEEKEND_MULTIPLIER = 1.3
DEFAULT_PROFILE_LOOKBACK_DAYS = 90
# How strongly the day-of-week profile is pulled toward its fallback, expressed
# in "pseudo sale-days". With 14, a product needs ~14 days with sales before its
# own pattern outweighs the store-wide (or weekend-prior) pattern.
PROFILE_PRIOR_STRENGTH = 14.0
# Oldest count considered as a starting point for the reconciliation window.
COUNT_LOOKBACK_DAYS = 365


class ReconciliationStatus(str, Enum):
    SHADOW = "shadow"  # more stock left the shelf than the POS recorded
    BALANCED = "balanced"  # physical movement matches POS exactly
    SURPLUS = "surplus"  # POS recorded more than physically left: missing invoice or miscount
    INSUFFICIENT_COUNTS = "insufficient_counts"  # need two counts to reconcile


@dataclass(frozen=True)
class ShadowSalesResult:
    daily: pd.DataFrame
    """One row per (product_id, ds) for every day in the period, zero-filled.

    Columns: product_id, ds (tz-naive local midnight, as Prophet expects),
    y_pos, y_shadow, y (= y_pos + y_shadow), reconciled (day lies inside a
    window where shadow sales could be measured).
    """

    reconciliation: pd.DataFrame
    """One row per product: window bounds, the balance terms, shadow_qty, status."""


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_inventory_counts(
    db: Session, store_id: uuid.UUID, start: datetime, end: datetime
) -> pd.DataFrame:
    stmt = (
        select(InventoryCount.product_id, InventoryCount.counted_quantity, InventoryCount.counted_at)
        .where(
            InventoryCount.store_id == store_id,
            InventoryCount.counted_at >= start,
            InventoryCount.counted_at <= end,
        )
        .order_by(InventoryCount.product_id, InventoryCount.counted_at)
    )
    df = pd.DataFrame(db.execute(stmt).all(), columns=["product_id", "counted_quantity", "counted_at"])
    df["counted_at"] = pd.to_datetime(df["counted_at"], utc=True)
    return df


def load_inbound_items(
    db: Session, store_id: uuid.UUID, start: datetime, end: datetime
) -> pd.DataFrame:
    stmt = (
        select(InvoiceItem.product_id, InvoiceItem.quantity, InboundInvoice.invoice_date)
        .join(InboundInvoice, InvoiceItem.invoice_id == InboundInvoice.id)
        .where(
            InboundInvoice.store_id == store_id,
            InboundInvoice.invoice_date > start,
            InboundInvoice.invoice_date <= end,
        )
    )
    df = pd.DataFrame(db.execute(stmt).all(), columns=["product_id", "quantity", "invoice_date"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    return df


def load_pos_sales(
    db: Session, store_id: uuid.UUID, start: datetime, end: datetime
) -> pd.DataFrame:
    # Raw rows rather than SQL daily buckets: reconciliation windows start and
    # end at exact count timestamps, often mid-day.
    stmt = select(
        SalesHistory.product_id, SalesHistory.quantity_sold, SalesHistory.timestamp
    ).where(
        SalesHistory.store_id == store_id,
        SalesHistory.timestamp > start,
        SalesHistory.timestamp <= end,
    )
    df = pd.DataFrame(db.execute(stmt).all(), columns=["product_id", "quantity_sold", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #

def select_count_windows(
    counts: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> pd.DataFrame:
    """Pick the (start count, end count) pair that best covers the period, per product.

    Start: the latest count at or before period_start; if none, the earliest
    count inside the period (the window is then shorter than the period).
    End: the latest count after the start count and at or before period_end.
    Products without such a pair are omitted.
    """
    columns = ["product_id", "window_start", "start_qty", "window_end", "end_qty"]
    rows = []
    for product_id, group in counts.sort_values("counted_at").groupby("product_id", sort=False):
        before = group[group["counted_at"] <= period_start]
        inside = group[(group["counted_at"] > period_start) & (group["counted_at"] <= period_end)]
        start = before.iloc[-1] if not before.empty else (inside.iloc[0] if not inside.empty else None)
        if start is None:
            continue
        after_start = inside[inside["counted_at"] > start["counted_at"]]
        if after_start.empty:
            continue
        end = after_start.iloc[-1]
        rows.append(
            (product_id, start["counted_at"], int(start["counted_quantity"]),
             end["counted_at"], int(end["counted_quantity"]))
        )
    return pd.DataFrame(rows, columns=columns)


def _sum_within_windows(
    events: pd.DataFrame, windows: pd.DataFrame, ts_col: str, qty_col: str
) -> pd.Series:
    """Sum `qty_col` per product for events with window_start < ts <= window_end."""
    merged = events.merge(windows[["product_id", "window_start", "window_end"]], on="product_id")
    in_window = merged[(merged[ts_col] > merged["window_start"]) & (merged[ts_col] <= merged["window_end"])]
    return in_window.groupby("product_id")[qty_col].sum()


def reconcile(
    counts: pd.DataFrame,
    inbound: pd.DataFrame,
    pos: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Apply the balance formula per product and classify the result."""
    product_ids = pd.Index(
        pd.concat([counts["product_id"], inbound["product_id"], pos["product_id"]]).unique(),
        name="product_id",
    )
    windows = select_count_windows(counts, period_start, period_end)

    summary = windows.set_index("product_id").reindex(product_ids)
    summary["inbound_qty"] = _sum_within_windows(inbound, windows, "invoice_date", "quantity")
    summary["pos_sold"] = _sum_within_windows(pos, windows, "timestamp", "quantity_sold")

    has_window = summary["window_start"].notna()
    # Inside a window "no rows" means zero; outside a window the terms are unknown (NA).
    for col in ("inbound_qty", "pos_sold"):
        summary[col] = pd.to_numeric(summary[col]).fillna(0).where(has_window)
    int_cols = ["start_qty", "end_qty", "inbound_qty", "pos_sold"]
    summary[int_cols] = summary[int_cols].astype("Int64")

    summary["actual_sold"] = summary["start_qty"] + summary["inbound_qty"] - summary["end_qty"]
    summary["shadow_raw"] = summary["actual_sold"] - summary["pos_sold"]
    summary["shadow_qty"] = summary["shadow_raw"].clip(lower=0).fillna(0).astype("int64")

    summary["status"] = np.select(
        [
            ~has_window,
            (summary["shadow_raw"] > 0).fillna(False).astype(bool),
            (summary["shadow_raw"] < 0).fillna(False).astype(bool),
        ],
        [
            ReconciliationStatus.INSUFFICIENT_COUNTS.value,
            ReconciliationStatus.SHADOW.value,
            ReconciliationStatus.SURPLUS.value,
        ],
        default=ReconciliationStatus.BALANCED.value,
    )
    return summary.reset_index()[
        ["product_id", "window_start", "window_end", "start_qty", "inbound_qty", "end_qty",
         "actual_sold", "pos_sold", "shadow_raw", "shadow_qty", "status"]
    ]


# --------------------------------------------------------------------------- #
# Distribution of shadow sales over days
# --------------------------------------------------------------------------- #

def weekend_prior(weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER) -> np.ndarray:
    """Day-of-week weights used when there is no history: flat, weekends boosted."""
    weights = np.ones(7)
    weights[list(WEEKEND_DAYS)] = weekend_multiplier
    return weights / weights.sum()


def day_of_week_profile(
    daily: pd.DataFrame, fallback: np.ndarray, prior_strength: float = PROFILE_PRIOR_STRENGTH
) -> np.ndarray:
    """Share of sales per weekday, shrunk toward `fallback` when data is thin.

    `daily` must be a zero-filled daily grid (columns ds, y) so that the mean
    per weekday reflects slow days too. Shrinkage weight is
    n / (n + prior_strength), where n is the number of days with sales.
    """
    sale_days = int((daily["y"] > 0).sum()) if not daily.empty else 0
    if sale_days == 0:
        return fallback
    mean_by_dow = daily.groupby(daily["ds"].dt.dayofweek)["y"].mean().reindex(range(7), fill_value=0.0)
    observed = mean_by_dow.to_numpy(dtype=float)
    observed /= observed.sum()
    alpha = sale_days / (sale_days + prior_strength)
    return alpha * observed + (1 - alpha) * fallback


def allocate_units(
    total: int,
    weights: np.ndarray,
    method: AllocationMethod = "proportional",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Split an integer total across slots proportionally to `weights`.

    "proportional": largest-remainder rounding. Deterministic, so the training
        data is reproducible and the sum is exact.
    "sample": one multinomial draw. Useful for simulating plausible daily
        noise (e.g. to build several training sets for uncertainty estimates).
    """
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=np.int64)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    if method == "sample":
        return (rng or np.random.default_rng()).multinomial(total, p)
    raw = total * p
    units = np.floor(raw).astype(np.int64)
    leftover = total - int(units.sum())
    units[np.argsort(-(raw - units), kind="stable")[:leftover]] += 1
    return units


def _local_day(ts: pd.Series | pd.Timestamp, tz: str):
    """UTC timestamp(s) -> tz-naive local midnight."""
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert(tz).tz_localize(None).normalize()
    return ts.dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()


def build_daily_pos_grid(
    pos: pd.DataFrame, product_ids: pd.Index, first_day: pd.Timestamp, last_day: pd.Timestamp, tz: str
) -> pd.DataFrame:
    """Aggregate POS rows to local days; every product gets every day (zeros included)."""
    days = pd.date_range(first_day, last_day, freq="D", name="ds")
    grid = pd.MultiIndex.from_product([product_ids, days], names=["product_id", "ds"])
    daily = (
        pos.assign(ds=_local_day(pos["timestamp"], tz))
        .groupby(["product_id", "ds"])["quantity_sold"]
        .sum()
        .reindex(grid, fill_value=0)
        .rename("y_pos")
        .reset_index()
    )
    daily["y_pos"] = daily["y_pos"].astype("int64")
    return daily


def distribute_shadow_sales(
    daily_pos: pd.DataFrame,
    reconciliation: pd.DataFrame,
    tz: str,
    *,
    weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER,
    method: AllocationMethod = "proportional",
    seed: int | None = None,
) -> pd.DataFrame:
    """Spread each product's shadow_qty across the days of its reconciliation window.

    Weights come from a hierarchical day-of-week profile:
    product history -> store history -> weekend prior. A product's shadow
    sales follow its own weekly rhythm when it has one and fall back to the
    store's rhythm otherwise.

    Returns `daily_pos` with y_shadow and reconciled columns added.
    """
    rng = np.random.default_rng(seed)
    store_daily = daily_pos.groupby("ds", as_index=False)["y_pos"].sum().rename(columns={"y_pos": "y"})
    store_profile = day_of_week_profile(store_daily, weekend_prior(weekend_multiplier))

    out = daily_pos.copy()
    out["y_shadow"] = np.int64(0)
    out["reconciled"] = False
    by_product = {pid: idx for pid, idx in out.groupby("product_id").groups.items()}

    measurable = reconciliation[
        reconciliation["status"] != ReconciliationStatus.INSUFFICIENT_COUNTS.value
    ]
    for rec in measurable.itertuples(index=False):
        idx = by_product.get(rec.product_id)
        if idx is None:
            continue
        rows = out.loc[idx]
        # Edge days are included whole: a count at 20:00 still leaves sales that evening.
        in_window = rows["ds"].between(_local_day(rec.window_start, tz), _local_day(rec.window_end, tz))
        window_idx = rows.index[in_window]
        out.loc[window_idx, "reconciled"] = True
        if rec.shadow_qty <= 0 or window_idx.empty:
            continue

        history = rows[["ds", "y_pos"]].rename(columns={"y_pos": "y"})
        profile = day_of_week_profile(history, store_profile)
        weights = profile[out.loc[window_idx, "ds"].dt.dayofweek.to_numpy()]
        out.loc[window_idx, "y_shadow"] = allocate_units(int(rec.shadow_qty), weights, method, rng)

    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compute_shadow_sales(
    counts: pd.DataFrame,
    inbound: pd.DataFrame,
    pos: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    *,
    tz: str,
    weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER,
    method: AllocationMethod = "proportional",
    seed: int | None = None,
) -> ShadowSalesResult:
    """Pure computation. `pos` should cover the profile lookback plus every window."""
    reconciliation = reconcile(counts, inbound, pos, period_start, period_end)

    # The grid spans all loaded history (for weekday profiles) and every
    # reconciliation window, which may begin before the period.
    starts = [period_start, reconciliation["window_start"].min()]
    if not pos.empty:
        starts.append(pos["timestamp"].min())
    first_day = _local_day(min(ts for ts in starts if pd.notna(ts)), tz)
    last_day = _local_day(period_end, tz)

    product_ids = pd.Index(reconciliation["product_id"], name="product_id")
    daily = build_daily_pos_grid(pos, product_ids, first_day, last_day, tz)
    daily = distribute_shadow_sales(
        daily, reconciliation, tz, weekend_multiplier=weekend_multiplier, method=method, seed=seed
    )

    # Shadow units that fall into a window's pre-period days are dropped here
    # on purpose: the output covers only the requested period.
    daily = daily[daily["ds"] >= _local_day(period_start, tz)].copy()
    daily["y"] = daily["y_pos"] + daily["y_shadow"]
    daily = daily[["product_id", "ds", "y_pos", "y_shadow", "y", "reconciled"]]
    return ShadowSalesResult(
        daily=daily.sort_values(["product_id", "ds"]).reset_index(drop=True),
        reconciliation=reconciliation,
    )


def analyze_shadow_sales(
    db: Session,
    store_id: uuid.UUID,
    period_days: int,
    *,
    as_of: datetime | None = None,
    profile_lookback_days: int = DEFAULT_PROFILE_LOOKBACK_DAYS,
    weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER,
    method: AllocationMethod = "proportional",
    seed: int | None = None,
) -> ShadowSalesResult:
    """Load a store's data and return both the daily series and the per-product reconciliation."""
    if period_days < 1:
        raise ValueError("period_days must be >= 1")

    period_end = pd.Timestamp(as_of or datetime.now(timezone.utc))
    period_end = period_end.tz_localize("UTC") if period_end.tzinfo is None else period_end.tz_convert("UTC")
    period_start = period_end - timedelta(days=period_days)

    counts = load_inventory_counts(
        db, store_id, period_start - timedelta(days=COUNT_LOOKBACK_DAYS), period_end
    )
    windows = select_count_windows(counts, period_start, period_end)

    load_from = period_start - timedelta(days=profile_lookback_days)
    if not windows.empty:
        load_from = min(load_from, windows["window_start"].min())

    inbound = load_inbound_items(db, store_id, load_from, period_end)
    pos = load_pos_sales(db, store_id, load_from, period_end)

    return compute_shadow_sales(
        counts, inbound, pos, period_start, period_end,
        tz=get_settings().STORE_TIMEZONE,
        weekend_multiplier=weekend_multiplier,
        method=method,
        seed=seed,
    )


def calculate_shadow_sales(
    db: Session, store_id: uuid.UUID, period_days: int, **kwargs
) -> pd.DataFrame:
    """Return the shadow-adjusted daily sales series for a store, ready for Prophet.

    Columns: product_id, ds, y_pos, y_shadow, y, reconciled. To train one
    Prophet model per product:

        for pid, g in df.groupby("product_id"):
            Prophet().fit(g[["ds", "y"]])

    See `analyze_shadow_sales` for keyword options and the reconciliation report.
    """
    return analyze_shadow_sales(db, store_id, period_days, **kwargs).daily
