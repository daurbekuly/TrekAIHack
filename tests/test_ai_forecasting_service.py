import uuid

import numpy as np
import pandas as pd
import pytest

from app.services.ai_forecasting_service import (
    DemandForecast,
    InsightsGenerationError,
    StockStatus,
    assess_stock,
    build_llm_payload,
    forecast_demand,
    render_template_message,
    validate_telegram_html,
)

P1 = uuid.UUID(int=1)


def history(values, end="2026-10-14") -> pd.DataFrame:
    return pd.DataFrame({"ds": pd.date_range(end=end, periods=len(values), freq="D"), "y": values})


def flat_forecast(per_day: float, days: int = 7, upper_factor: float = 1.3) -> DemandForecast:
    daily = pd.DataFrame({
        "ds": pd.date_range("2026-10-15", periods=days, freq="D"),
        "yhat": [per_day] * days,
        "yhat_lower": [per_day * 0.7] * days,
        "yhat_upper": [per_day * upper_factor] * days,
    })
    total = per_day * days
    return DemandForecast(P1, days, "baseline", 0.8, total, total * 0.7, total * upper_factor, daily)


# --------------------------------------------------------------------------- #
# forecast_demand
# --------------------------------------------------------------------------- #

def test_sparse_history_uses_baseline():
    fc = forecast_demand(P1, history([0] * 50 + [2, 0, 3, 0, 1]), days_ahead=7)
    assert fc.method == "baseline"
    assert fc.days_ahead == 7
    assert fc.daily["ds"].iloc[0] == pd.Timestamp("2026-10-15")
    assert fc.lower_total <= fc.expected_total <= fc.upper_total


def test_zero_history_forecasts_zero():
    fc = forecast_demand(P1, history([0] * 30))
    assert fc.expected_total == 0
    assert fc.required_stock == 0


def test_leading_zeros_do_not_dilute_baseline_rate():
    # New product: 60 days not stocked, then 5 days selling 4/day.
    fc = forecast_demand(P1, history([0] * 60 + [4] * 5), days_ahead=7)
    assert fc.expected_total == pytest.approx(28, rel=0.15)


def test_prophet_learns_weekly_pattern():
    pytest.importorskip("prophet")
    days = pd.date_range(end="2026-10-14", periods=120, freq="D")
    rng = np.random.default_rng(1)
    y = np.where(days.dayofweek >= 5, 20, 8) + rng.normal(0, 1, len(days))
    fc = forecast_demand(P1, pd.DataFrame({"ds": days, "y": y}), days_ahead=7)

    assert fc.method == "prophet"
    weekend = fc.daily["ds"].dt.dayofweek >= 5
    assert fc.daily.loc[weekend, "yhat"].mean() > fc.daily.loc[~weekend, "yhat"].mean() + 5
    assert fc.expected_total == pytest.approx(5 * 8 + 2 * 20, rel=0.15)
    assert fc.lower_total < fc.expected_total < fc.upper_total


# --------------------------------------------------------------------------- #
# assess_stock
# --------------------------------------------------------------------------- #

def test_stockout_risk_orders_up_to_upper_bound():
    rec = assess_stock(flat_forecast(10), "Milk", current_stock=25)  # 70 expected, 91 upper
    assert rec.status is StockStatus.STOCKOUT_RISK
    assert rec.days_until_stockout == 3
    assert rec.order_qty == 91 - 25


def test_reorder_when_stock_covers_mean_but_not_upper():
    rec = assess_stock(flat_forecast(10), "Bread", current_stock=80)
    assert rec.status is StockStatus.REORDER
    assert rec.days_until_stockout is None
    assert rec.order_qty == 11


def test_overstock_reports_excess():
    rec = assess_stock(flat_forecast(1), "Salt", current_stock=200)  # 30-day target = 30
    assert rec.status is StockStatus.OVERSTOCK
    assert rec.order_qty == 0
    assert rec.excess_units == 170


def test_dead_stock_is_overstock():
    rec = assess_stock(flat_forecast(0), "Old item", current_stock=12)
    assert rec.status is StockStatus.OVERSTOCK
    assert rec.days_of_cover is None


def test_negative_book_stock_counts_as_empty_shelf():
    rec = assess_stock(flat_forecast(2), "Eggs", current_stock=-5)
    assert rec.current_stock == -5
    assert rec.order_qty == flat_forecast(2).required_stock
    assert rec.days_until_stockout == 1


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #

def _items():
    return [
        assess_stock(flat_forecast(10), "Milk <2.5%> & Co", 25),
        assess_stock(flat_forecast(10), "Bread", 80),
        assess_stock(flat_forecast(1), "Salt", 200),
        assess_stock(flat_forecast(3), "Sugar", 40),
    ]


def test_payload_groups_and_omits_ok_items():
    payload = build_llm_payload("Shop", 7, _items())
    assert [i["name"] for i in payload["stockout_risk"]] == ["Milk <2.5%> & Co"]
    assert [i["name"] for i in payload["reorder"]] == ["Bread"]
    assert [i["name"] for i in payload["overstock"]] == ["Salt"]
    assert "product_id" not in payload["reorder"][0]


def test_template_message_is_valid_telegram_html():
    message = render_template_message(build_llm_payload("Shop", 7, _items()))
    assert "Milk &lt;2.5%&gt; &amp; Co" in message
    assert "<b>66</b>" in message  # Milk order_qty
    assert validate_telegram_html(message) == message


def test_template_message_when_nothing_to_do():
    message = render_template_message(build_llm_payload("Shop", 7, []))
    assert "✅" in message


@pytest.mark.parametrize(
    "bad",
    ["", "x" * 5000, "<h1>Order</h1>", "<table><tr><td>1</td></tr></table>"],
)
def test_validate_rejects_unrenderable_output(bad):
    with pytest.raises(InsightsGenerationError):
        validate_telegram_html(bad)


def test_validate_strips_code_fences():
    assert validate_telegram_html("```html\n<b>Hi</b>\n```") == "<b>Hi</b>"


# --------------------------------------------------------------------------- #
# generate_purchase_recommendations (DB loaders patched out)
# --------------------------------------------------------------------------- #

class FakeWriter:
    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.payloads: list[dict] = []

    def write(self, payload: dict) -> str:
        self.payloads.append(payload)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def fake_store(monkeypatch):
    from types import SimpleNamespace

    from app.services import ai_forecasting_service as svc

    store_id = uuid.UUID(int=99)
    p_busy, p_idle = uuid.UUID(int=10), uuid.UUID(int=11)
    products = [
        SimpleNamespace(id=p_busy, name="Milk", current_stock=3),
        SimpleNamespace(id=p_idle, name="Discontinued", current_stock=0),
    ]
    days = pd.date_range(end="2026-10-14", periods=20, freq="D")
    daily = pd.concat([
        pd.DataFrame({"product_id": p_busy, "ds": days, "y": 5.0}),
        pd.DataFrame({"product_id": p_idle, "ds": days, "y": 0.0}),
    ])
    monkeypatch.setattr(svc, "_get_store", lambda db, sid: SimpleNamespace(id=sid, name="Taraz Market"))
    monkeypatch.setattr(svc, "_load_products", lambda db, sid: products)
    monkeypatch.setattr(svc, "_load_history", lambda db, sid, hd: daily)
    return store_id


def test_recommendations_use_llm_message(fake_store):
    from app.services.ai_forecasting_service import generate_purchase_recommendations

    writer = FakeWriter("<b>Order 35 Milk</b>")
    report = generate_purchase_recommendations(None, fake_store, writer=writer)

    assert report.generated_by == "llm"
    assert report.message == "<b>Order 35 Milk</b>"
    assert [i.name for i in report.items] == ["Milk"]  # inactive item skipped
    assert writer.payloads[0]["stockout_risk"][0]["name"] == "Milk"


def test_recommendations_fall_back_to_template_on_llm_failure(fake_store):
    from app.services.ai_forecasting_service import generate_purchase_recommendations

    report = generate_purchase_recommendations(
        None, fake_store, writer=FakeWriter(InsightsGenerationError("timeout"))
    )
    assert report.generated_by == "template"
    assert "Milk" in report.message


def test_llm_not_called_when_nothing_actionable(fake_store, monkeypatch):
    from app.services import ai_forecasting_service as svc

    monkeypatch.setattr(svc, "_load_products", lambda db, sid: [])
    writer = FakeWriter("unused")
    report = svc.generate_purchase_recommendations(None, fake_store, writer=writer)

    assert writer.payloads == []
    assert report.generated_by == "template"
