"""AI forecasting engine: demand forecasts, stock policy and LLM purchase advice.

Pipeline:

    shadow-adjusted daily sales (Phase 2)
        -> forecast_demand            Prophet, or a Poisson baseline for sparse items
        -> assess_stock               deterministic stock policy -> order quantities
        -> InsightsWriter             LLM turns the numbers into a Telegram message

Design rule: every number shown to the shop owner is computed here in code.
The LLM only phrases and prioritises; it never calculates quantities. If the
LLM is unavailable or misbehaves, a deterministic template message is sent
instead, so the owner always gets the recommendation.

Layout:
    forecast_demand, assess_stock,
    render_template_message           -- pure, unit-testable without DB or network
    OpenAIInsightsWriter              -- the only code that talks to OpenAI
    forecast_product_demand,
    generate_purchase_recommendations -- public entry points (DB + orchestration)
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from openai import OpenAI, OpenAIError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Product, Store
from app.services.discrepancy_service import calculate_shadow_sales

logger = logging.getLogger(__name__)

ForecastMethod = Literal["prophet", "baseline"]

DEFAULT_DAYS_AHEAD = 7
DEFAULT_HISTORY_DAYS = 180
# Interval used both for reporting and as the service level for ordering:
# ordering up to the upper bound covers demand in ~90% of plausible futures.
DEFAULT_INTERVAL_WIDTH = 0.8
# Prophet needs enough signal to estimate a weekly pattern; below this the
# Poisson baseline is more honest than a fitted curve.
MIN_PROPHET_HISTORY_DAYS = 28
MIN_PROPHET_SALE_DAYS = 10
BASELINE_WINDOW_DAYS = 28
UNCERTAINTY_SAMPLES = 500
# Stock covering more than this many days of expected demand is overstock.
DEFAULT_OVERSTOCK_COVER_DAYS = 30
# Caps on what is sent to the LLM, to bound tokens and message length.
MAX_ITEMS_PER_SECTION = 15
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_ALLOWED_TAGS = {"b", "strong", "i", "em", "u", "s", "code", "pre"}


class ForecastingError(Exception):
    """Base error for this module."""


class NotFoundError(ForecastingError):
    """The store or product does not exist (or the product is not in that store)."""


class StockStatus(str, Enum):
    STOCKOUT_RISK = "stockout_risk"  # expected demand exceeds stock within the horizon
    REORDER = "reorder"  # enough for expected demand, not for the upper bound
    OVERSTOCK = "overstock"  # stock covers far more days than needed
    OK = "ok"


@dataclass(frozen=True)
class DemandForecast:
    product_id: uuid.UUID
    days_ahead: int
    method: ForecastMethod
    interval_width: float
    expected_total: float
    lower_total: float
    upper_total: float
    daily: pd.DataFrame = field(repr=False)
    """Columns: ds, yhat, yhat_lower, yhat_upper (per-day, clipped at 0)."""

    @property
    def required_stock(self) -> int:
        """Units to have on hand to cover the horizon at the configured service level."""
        return math.ceil(self.upper_total)


@dataclass(frozen=True)
class ProductRecommendation:
    product_id: uuid.UUID
    name: str
    current_stock: int
    expected_demand: float
    demand_upper: float
    required_stock: int
    order_qty: int
    days_of_cover: float | None  # None when no demand is expected
    days_until_stockout: int | None  # None when stock lasts the whole horizon
    excess_units: int
    status: StockStatus
    forecast_method: ForecastMethod


@dataclass(frozen=True)
class PurchaseRecommendationReport:
    store_id: uuid.UUID
    days_ahead: int
    message: str
    """Telegram-ready text (parse_mode="HTML")."""
    items: list[ProductRecommendation]
    generated_by: Literal["llm", "template"]


# --------------------------------------------------------------------------- #
# Forecasting (pure)
# --------------------------------------------------------------------------- #

def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Sort, coerce types and drop leading zero days before the first sale.

    Leading zeros usually mean "not stocked yet", not "zero demand", and would
    drag the level and trend down.
    """
    h = history[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    h["y"] = h["y"].astype(float)
    first_sale = h.index[h["y"] > 0]
    return h.iloc[first_sale[0]:].reset_index(drop=True) if len(first_sale) else h.iloc[0:0]


def _future_days(last_day: pd.Timestamp, days_ahead: int) -> pd.DatetimeIndex:
    return pd.date_range(last_day + pd.Timedelta(days=1), periods=days_ahead, freq="D")


def _summarize_samples(
    product_id: uuid.UUID,
    days: pd.DatetimeIndex,
    samples: np.ndarray,
    method: ForecastMethod,
    interval_width: float,
) -> DemandForecast:
    """Build a forecast from a (days x samples) matrix of simulated demand.

    The interval of the *total* is taken from per-sample sums. Summing daily
    upper bounds instead would overstate it, since days rarely all peak together.
    """
    samples = np.clip(samples, 0, None)
    q_lo, q_hi = (1 - interval_width) / 2, 1 - (1 - interval_width) / 2
    totals = samples.sum(axis=0)
    daily = pd.DataFrame({
        "ds": days,
        "yhat": samples.mean(axis=1),
        "yhat_lower": np.quantile(samples, q_lo, axis=1),
        "yhat_upper": np.quantile(samples, q_hi, axis=1),
    })
    return DemandForecast(
        product_id=product_id,
        days_ahead=len(days),
        method=method,
        interval_width=interval_width,
        expected_total=float(totals.mean()),
        lower_total=float(np.quantile(totals, q_lo)),
        upper_total=float(np.quantile(totals, q_hi)),
        daily=daily,
    )


def _baseline_forecast(
    product_id: uuid.UUID,
    history: pd.DataFrame,
    last_day: pd.Timestamp,
    days_ahead: int,
    interval_width: float,
    rng: np.random.Generator,
) -> DemandForecast:
    """Poisson demand at the recent average rate. Used for sparse or new items."""
    recent = history.tail(BASELINE_WINDOW_DAYS)
    rate = float(recent["y"].mean()) if not recent.empty else 0.0
    samples = rng.poisson(rate, size=(days_ahead, UNCERTAINTY_SAMPLES)).astype(float)
    return _summarize_samples(
        product_id, _future_days(last_day, days_ahead), samples, "baseline", interval_width
    )


def _prophet_forecast(
    product_id: uuid.UUID,
    history: pd.DataFrame,
    days_ahead: int,
    interval_width: float,
    seed: int,
) -> DemandForecast:
    # Imported lazily: Prophet pulls in cmdstanpy and takes seconds to import,
    # which the API process should not pay at startup.
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=len(history) >= 365,
        daily_seasonality=False,
        interval_width=interval_width,
        uncertainty_samples=UNCERTAINTY_SAMPLES,
    )
    model.add_country_holidays(country_name="KZ")
    model.fit(history)

    future = model.make_future_dataframe(periods=days_ahead, freq="D", include_history=False)
    np.random.seed(seed)  # Prophet samples through the global numpy RNG
    samples = model.predictive_samples(future)["yhat"]
    return _summarize_samples(
        product_id, pd.DatetimeIndex(future["ds"]), samples, "prophet", interval_width
    )


def forecast_demand(
    product_id: uuid.UUID,
    history: pd.DataFrame,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    *,
    interval_width: float = DEFAULT_INTERVAL_WIDTH,
    seed: int = 0,
) -> DemandForecast:
    """Forecast daily demand for one product from a complete daily (ds, y) series.

    `history` must end at the last *complete* day; the forecast starts the day after.
    Falls back to the Poisson baseline when history is too short or too sparse,
    or if Prophet fails to fit.
    """
    if days_ahead < 1:
        raise ValueError("days_ahead must be >= 1")
    if history.empty:
        raise ValueError("history must contain at least one day")

    last_day = pd.Timestamp(history["ds"].max())
    h = _prepare_history(history)
    rng = np.random.default_rng(seed)

    enough_data = len(h) >= MIN_PROPHET_HISTORY_DAYS and int((h["y"] > 0).sum()) >= MIN_PROPHET_SALE_DAYS
    if not enough_data:
        return _baseline_forecast(product_id, h, last_day, days_ahead, interval_width, rng)
    try:
        return _prophet_forecast(product_id, h, days_ahead, interval_width, seed)
    except Exception:  # Stan/Prophet failures are varied and not worth enumerating
        logger.exception("Prophet failed for product %s; using baseline", product_id)
        return _baseline_forecast(product_id, h, last_day, days_ahead, interval_width, rng)


# --------------------------------------------------------------------------- #
# Stock policy (pure)
# --------------------------------------------------------------------------- #

def assess_stock(
    forecast: DemandForecast,
    name: str,
    current_stock: int,
    *,
    overstock_cover_days: int = DEFAULT_OVERSTOCK_COVER_DAYS,
) -> ProductRecommendation:
    """Turn a forecast and the book stock into a status and an order quantity.

    Order quantity tops stock up to `required_stock` (the upper bound of
    demand over the horizon). Negative book stock is a data-quality artefact
    (see Phase 2) and counts as zero on the shelf.
    """
    available = max(int(current_stock), 0)
    order_qty = max(forecast.required_stock - available, 0)

    cumulative = forecast.daily["yhat"].cumsum().to_numpy()
    runs_out = np.flatnonzero(cumulative > available)
    days_until_stockout = int(runs_out[0]) + 1 if runs_out.size else None

    avg_daily = forecast.expected_total / forecast.days_ahead
    days_of_cover = available / avg_daily if avg_daily > 0 else None
    target_max = math.ceil(avg_daily * overstock_cover_days)
    excess_units = max(available - max(target_max, forecast.required_stock), 0)

    if available < forecast.expected_total:
        status = StockStatus.STOCKOUT_RISK
    elif order_qty > 0:
        status = StockStatus.REORDER
    elif available > 0 and (days_of_cover is None or days_of_cover > overstock_cover_days):
        status = StockStatus.OVERSTOCK
    else:
        status = StockStatus.OK

    return ProductRecommendation(
        product_id=forecast.product_id,
        name=name,
        current_stock=int(current_stock),
        expected_demand=round(forecast.expected_total, 1),
        demand_upper=round(forecast.upper_total, 1),
        required_stock=forecast.required_stock,
        order_qty=order_qty,
        days_of_cover=round(days_of_cover, 1) if days_of_cover is not None else None,
        days_until_stockout=days_until_stockout,
        excess_units=excess_units if status is StockStatus.OVERSTOCK else 0,
        status=status,
        forecast_method=forecast.method,
    )


def _by_status(items: list[ProductRecommendation], status: StockStatus) -> list[ProductRecommendation]:
    return [i for i in items if i.status is status]


def build_llm_payload(
    store_name: str, days_ahead: int, items: list[ProductRecommendation]
) -> dict:
    """Compact, pre-sorted facts for the LLM. Most urgent first, capped per section."""
    def row(i: ProductRecommendation) -> dict:
        d = asdict(i)
        d.pop("product_id")
        d.pop("forecast_method")
        d["status"] = i.status.value
        return d

    stockout = sorted(
        _by_status(items, StockStatus.STOCKOUT_RISK), key=lambda i: (i.days_until_stockout or 0, -i.order_qty)
    )
    reorder = sorted(_by_status(items, StockStatus.REORDER), key=lambda i: -i.order_qty)
    overstock = sorted(_by_status(items, StockStatus.OVERSTOCK), key=lambda i: -i.excess_units)
    return {
        "store_name": store_name,
        "horizon_days": days_ahead,
        "stockout_risk": [row(i) for i in stockout[:MAX_ITEMS_PER_SECTION]],
        "reorder": [row(i) for i in reorder[:MAX_ITEMS_PER_SECTION]],
        "overstock": [row(i) for i in overstock[:MAX_ITEMS_PER_SECTION]],
        "omitted_items": max(len(stockout) - MAX_ITEMS_PER_SECTION, 0)
        + max(len(reorder) - MAX_ITEMS_PER_SECTION, 0)
        + max(len(overstock) - MAX_ITEMS_PER_SECTION, 0),
    }


def render_template_message(payload: dict) -> str:
    """Deterministic Telegram HTML message. Fallback when the LLM is unavailable."""
    esc = html.escape
    lines = [f"<b>📦 {esc(payload['store_name'])}: закуп на {payload['horizon_days']} дн.</b>", ""]

    if payload["stockout_risk"]:
        lines.append("<b>🔴 Скоро закончатся</b>")
        for i in payload["stockout_risk"]:
            when = f", хватит на ~{i['days_until_stockout']} дн." if i["days_until_stockout"] else ""
            lines.append(f"• {esc(i['name'])}: остаток {i['current_stock']}{when} → заказать <b>{i['order_qty']}</b>")
        lines.append("")
    if payload["reorder"]:
        lines.append("<b>🟡 Пополнить запас</b>")
        for i in payload["reorder"]:
            lines.append(f"• {esc(i['name'])}: остаток {i['current_stock']} → заказать <b>{i['order_qty']}</b>")
        lines.append("")
    if payload["overstock"]:
        lines.append("<b>🔵 Излишки (не заказывать)</b>")
        for i in payload["overstock"]:
            lines.append(f"• {esc(i['name'])}: остаток {i['current_stock']}, лишних ~{i['excess_units']}")
        lines.append("")
    if not (payload["stockout_risk"] or payload["reorder"] or payload["overstock"]):
        lines.append("✅ Запасов хватает, заказывать ничего не нужно.")
    if payload["omitted_items"]:
        lines.append(f"…и ещё {payload['omitted_items']} позиций в полном отчёте.")
    return "\n".join(lines).strip()[:TELEGRAM_MESSAGE_LIMIT]


# --------------------------------------------------------------------------- #
# LLM insights
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a purchasing assistant for a small retail shop in Kazakhstan. You write \
the weekly purchase-order message the owner receives in Telegram.

You receive JSON with pre-computed facts. Rules:
- Use ONLY the products and numbers in the JSON. Never invent products, never \
recompute or round quantities differently. `order_qty` is the exact amount to order.
- Structure: a one-line greeting/summary, then sections in this order, skipping \
empty ones: 🔴 stockout risk (most urgent first, mention days_until_stockout), \
🟡 reorder, 🔵 overstock (advise not to reorder; suggest a promotion if excess is large).
- One bullet per product: name, current stock, order quantity in bold.
- End with one short practical tip. Friendly, concise, no filler.
- If omitted_items > 0, say how many more items are in the full report.
- Format for Telegram parse_mode=HTML: only <b>, <i>, <u>, <s>, <code> tags. \
No Markdown, no headings with #, no tables. Escape &, <, > in product names.
- Keep the whole message under 3500 characters.
- Write in {language}.
"""


class InsightsWriter(Protocol):
    def write(self, payload: dict) -> str: ...


class InsightsGenerationError(ForecastingError):
    """The LLM call failed or returned something unusable."""


def validate_telegram_html(text: str) -> str:
    """Strip code fences and reject output Telegram would refuse to render."""
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text.strip())
    if not text:
        raise InsightsGenerationError("empty LLM response")
    if len(text) > TELEGRAM_MESSAGE_LIMIT:
        raise InsightsGenerationError(f"LLM response too long ({len(text)} chars)")
    tags = {t.lower() for t in re.findall(r"</?\s*([a-zA-Z0-9]+)", text)}
    if not tags <= TELEGRAM_ALLOWED_TAGS:
        raise InsightsGenerationError(f"unsupported tags: {sorted(tags - TELEGRAM_ALLOWED_TAGS)}")
    return text


class OpenAIInsightsWriter:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        language: str = "Russian",
        max_completion_tokens: int = 1500,
    ) -> None:
        self._client = client
        self._model = model
        self._language = language
        self._max_completion_tokens = max_completion_tokens

    @classmethod
    def from_settings(cls, *, language: str = "Russian") -> OpenAIInsightsWriter | None:
        """Build from app settings, or return None when no API key is configured."""
        settings = get_settings()
        key = settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else ""
        if not key:
            return None
        client = OpenAI(api_key=key, timeout=settings.OPENAI_TIMEOUT_SECONDS, max_retries=2)
        return cls(client, settings.OPENAI_MODEL, language=language)

    def write(self, payload: dict) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(language=self._language)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                ],
                max_completion_tokens=self._max_completion_tokens,
            )
        except OpenAIError as exc:
            raise InsightsGenerationError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        return validate_telegram_html(content or "")


# --------------------------------------------------------------------------- #
# Data access & orchestration
# --------------------------------------------------------------------------- #

def _get_store(db: Session, store_id: uuid.UUID) -> Store:
    store = db.get(Store, store_id)
    if store is None:
        raise NotFoundError(f"store {store_id} not found")
    return store


def _load_products(db: Session, store_id: uuid.UUID) -> list[Product]:
    return list(db.scalars(select(Product).where(Product.store_id == store_id).order_by(Product.name)))


def _load_history(db: Session, store_id: uuid.UUID, history_days: int) -> pd.DataFrame:
    """Shadow-adjusted daily sales, ending at the last complete local day."""
    daily = calculate_shadow_sales(db, store_id, history_days)
    tz = get_settings().STORE_TIMEZONE
    today = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(tz).tz_localize(None).normalize()
    return daily[daily["ds"] < today]  # today is still in progress


def _history_for(product_id: uuid.UUID, daily: pd.DataFrame, history_days: int) -> pd.DataFrame:
    """One product's (ds, y) series; an all-zero series if it never appeared in the data."""
    h = daily.loc[daily["product_id"] == product_id, ["ds", "y"]]
    if not h.empty:
        return h
    tz = get_settings().STORE_TIMEZONE
    yesterday = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(tz).tz_localize(None).normalize() - pd.Timedelta(days=1)
    days = pd.date_range(end=yesterday, periods=history_days, freq="D")
    return pd.DataFrame({"ds": days, "y": 0.0})


def forecast_product_demand(
    db: Session,
    store_id: uuid.UUID,
    product_id: uuid.UUID,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    *,
    history_days: int = DEFAULT_HISTORY_DAYS,
    interval_width: float = DEFAULT_INTERVAL_WIDTH,
) -> DemandForecast:
    """Forecast one product's demand for the next `days_ahead` days.

    `required_stock` on the result is the stock needed to cover the horizon at
    the `interval_width` service level; `lower_total`/`upper_total` give the
    interval of total demand.
    """
    product = db.get(Product, product_id)
    if product is None or product.store_id != store_id:
        raise NotFoundError(f"product {product_id} not found in store {store_id}")
    daily = _load_history(db, store_id, history_days)
    return forecast_demand(
        product_id, _history_for(product_id, daily, history_days), days_ahead, interval_width=interval_width
    )


def generate_purchase_recommendations(
    db: Session,
    store_id: uuid.UUID,
    *,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    history_days: int = DEFAULT_HISTORY_DAYS,
    overstock_cover_days: int = DEFAULT_OVERSTOCK_COVER_DAYS,
    writer: InsightsWriter | None = None,
    language: str = "Russian",
) -> PurchaseRecommendationReport:
    """Forecast every product in a store and produce a Telegram purchase-order message.

    `writer` defaults to OpenAI from settings; pass a fake in tests. Any LLM
    failure falls back to the deterministic template, so this never fails
    because of the LLM.
    """
    store = _get_store(db, store_id)
    products = _load_products(db, store_id)
    daily = _load_history(db, store_id, history_days)

    items: list[ProductRecommendation] = []
    for product in products:
        try:
            forecast = forecast_demand(
                product.id, _history_for(product.id, daily, history_days), days_ahead
            )
        except Exception:
            # One bad series must not sink the whole store's report.
            logger.exception("Forecast failed for product %s in store %s", product.id, store_id)
            continue
        if forecast.expected_total == 0 and product.current_stock <= 0:
            continue  # inactive item: no stock, no demand
        items.append(
            assess_stock(forecast, product.name, product.current_stock, overstock_cover_days=overstock_cover_days)
        )

    payload = build_llm_payload(store.name, days_ahead, items)
    actionable = payload["stockout_risk"] or payload["reorder"] or payload["overstock"]

    message, generated_by = render_template_message(payload), "template"
    writer = writer if writer is not None else OpenAIInsightsWriter.from_settings(language=language)
    if actionable and writer is not None:
        try:
            message, generated_by = writer.write(payload), "llm"
        except InsightsGenerationError:
            logger.warning("LLM insights failed for store %s; sending template", store_id, exc_info=True)

    return PurchaseRecommendationReport(
        store_id=store_id, days_ahead=days_ahead, message=message, items=items, generated_by=generated_by
    )
