import uuid

import numpy as np
import pandas as pd
import pytest

from app.services.discrepancy_service import (
    ReconciliationStatus,
    allocate_units,
    compute_shadow_sales,
    reconcile,
)

TZ = "Asia/Almaty"
P1 = uuid.UUID(int=1)
P2 = uuid.UUID(int=2)


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def counts_df(rows):
    df = pd.DataFrame(rows, columns=["product_id", "counted_quantity", "counted_at"])
    df["counted_at"] = pd.to_datetime(df["counted_at"], utc=True)
    return df


def inbound_df(rows=()):
    df = pd.DataFrame(list(rows), columns=["product_id", "quantity", "invoice_date"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    return df


def pos_df(rows=()):
    df = pd.DataFrame(list(rows), columns=["product_id", "quantity_sold", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


PERIOD_START = ts("2026-09-01 00:00")
PERIOD_END = ts("2026-09-29 00:00")


# --------------------------------------------------------------------------- #
# allocate_units
# --------------------------------------------------------------------------- #

def test_allocate_proportional_is_exact_and_deterministic():
    weights = np.array([1.0, 1.0, 1.3, 1.3, 1.0])
    first = allocate_units(17, weights)
    assert first.sum() == 17
    assert np.array_equal(first, allocate_units(17, weights))
    assert first[2] >= first[0]


def test_allocate_sample_preserves_total():
    units = allocate_units(50, np.ones(7), method="sample", rng=np.random.default_rng(0))
    assert units.sum() == 50


def test_allocate_zero_total():
    assert allocate_units(0, np.ones(3)).tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- #
# reconcile
# --------------------------------------------------------------------------- #

def test_reconcile_applies_balance_formula():
    counts = counts_df([(P1, 100, "2026-09-01 00:00"), (P1, 10, "2026-09-28 18:00")])
    inbound = inbound_df([(P1, 50, "2026-09-10 08:00")])
    pos = pos_df([(P1, 60, "2026-09-05 10:00"), (P1, 60, "2026-09-20 10:00")])

    row = reconcile(counts, inbound, pos, PERIOD_START, PERIOD_END).iloc[0]

    assert row["actual_sold"] == 100 + 50 - 10
    assert row["pos_sold"] == 120
    assert row["shadow_qty"] == 20
    assert row["status"] == ReconciliationStatus.SHADOW.value


def test_reconcile_ignores_events_outside_count_window():
    counts = counts_df([(P1, 100, "2026-09-03 00:00"), (P1, 90, "2026-09-25 00:00")])
    pos = pos_df([
        (P1, 999, "2026-09-02 10:00"),  # before the start count
        (P1, 5, "2026-09-10 10:00"),
        (P1, 999, "2026-09-26 10:00"),  # after the end count
    ])
    inbound = inbound_df([(P1, 999, "2026-09-01 10:00")])

    row = reconcile(counts, inbound, pos, PERIOD_START, PERIOD_END).iloc[0]

    assert row["inbound_qty"] == 0
    assert row["pos_sold"] == 5
    assert row["shadow_qty"] == 5


def test_reconcile_flags_surplus_and_insufficient_counts():
    counts = counts_df([
        (P1, 10, "2026-09-01 00:00"), (P1, 10, "2026-09-28 00:00"),  # POS sold 3 but stock unchanged
        (P2, 50, "2026-09-10 00:00"),  # only one count
    ])
    pos = pos_df([(P1, 3, "2026-09-05 10:00"), (P2, 4, "2026-09-12 10:00")])

    result = reconcile(counts, inbound_df(), pos, PERIOD_START, PERIOD_END).set_index("product_id")

    assert result.loc[P1, "status"] == ReconciliationStatus.SURPLUS.value
    assert result.loc[P1, "shadow_qty"] == 0
    assert result.loc[P2, "status"] == ReconciliationStatus.INSUFFICIENT_COUNTS.value
    assert result.loc[P2, "shadow_qty"] == 0
    assert pd.isna(result.loc[P2, "actual_sold"])


# --------------------------------------------------------------------------- #
# compute_shadow_sales (end to end, no DB)
# --------------------------------------------------------------------------- #

def test_shadow_sales_without_history_follow_weekend_prior():
    counts = counts_df([(P1, 100, "2026-09-01 00:00"), (P1, 72, "2026-09-28 18:00")])

    result = compute_shadow_sales(counts, inbound_df(), pos_df(), PERIOD_START, PERIOD_END, tz=TZ)
    daily = result.daily

    assert daily["y_shadow"].sum() == 28
    assert (daily["y"] == daily["y_pos"] + daily["y_shadow"]).all()
    weekend = daily["ds"].dt.dayofweek >= 5
    assert daily.loc[weekend, "y_shadow"].mean() > daily.loc[~weekend, "y_shadow"].mean()
    # Prophet needs tz-naive ds and a complete daily grid.
    assert daily["ds"].dt.tz is None
    assert daily["ds"].diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_shadow_sales_follow_product_weekday_pattern():
    mondays = pd.date_range("2026-06-01 06:00", "2026-09-28 06:00", freq="W-MON", tz="UTC")
    pos = pos_df([(P1, 10, t) for t in mondays])
    pos_in_window = int(pos.loc[pos["timestamp"] > PERIOD_START, "quantity_sold"].sum())
    counts = counts_df([
        (P1, 500, "2026-09-01 00:00"),
        (P1, 500 - pos_in_window - 20, "2026-09-28 18:00"),  # 20 shadow units
    ])

    result = compute_shadow_sales(counts, inbound_df(), pos, PERIOD_START, PERIOD_END, tz=TZ)
    daily = result.daily

    assert result.reconciliation.iloc[0]["shadow_qty"] == 20
    assert daily["y_shadow"].sum() == 20
    is_monday = daily["ds"].dt.dayofweek == 0
    assert daily.loc[is_monday, "y_shadow"].sum() > daily.loc[~is_monday, "y_shadow"].sum()


def test_window_starting_before_period_keeps_only_in_period_share():
    counts = counts_df([(P1, 100, "2026-08-22 00:00"), (P1, 62, "2026-09-28 18:00")])

    result = compute_shadow_sales(counts, inbound_df(), pos_df(), PERIOD_START, PERIOD_END, tz=TZ)

    assert result.reconciliation.iloc[0]["shadow_qty"] == 38
    assert 0 < result.daily["y_shadow"].sum() < 38
    assert result.daily["ds"].min() == pd.Timestamp("2026-09-01")


def test_empty_store_returns_empty_frames():
    result = compute_shadow_sales(counts_df([]), inbound_df(), pos_df(), PERIOD_START, PERIOD_END, tz=TZ)
    assert result.daily.empty
    assert result.reconciliation.empty


def test_sample_method_is_reproducible_with_seed():
    counts = counts_df([(P1, 100, "2026-09-01 00:00"), (P1, 50, "2026-09-28 18:00")])
    args = (counts, inbound_df(), pos_df(), PERIOD_START, PERIOD_END)

    a = compute_shadow_sales(*args, tz=TZ, method="sample", seed=42).daily
    b = compute_shadow_sales(*args, tz=TZ, method="sample", seed=42).daily

    assert a["y_shadow"].sum() == 50
    pd.testing.assert_frame_equal(a, b)
