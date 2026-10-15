import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.bot import CONFIRM_BUTTON_TEXT, OrderAction, confirm_keyboard
from app.services.ai_forecasting_service import ProductRecommendation, StockStatus
from app.services.purchase_order_service import (
    MAX_WHATSAPP_URL_LENGTH,
    create_draft_order,
    format_order_text,
    whatsapp_share_url,
)


def rec(name: str, order_qty: int) -> ProductRecommendation:
    return ProductRecommendation(
        product_id=uuid.uuid4(), name=name, current_stock=0, expected_demand=1, demand_upper=1,
        required_stock=order_qty, order_qty=order_qty, days_of_cover=None, days_until_stockout=None,
        excess_units=0, status=StockStatus.REORDER, forecast_method="baseline",
    )


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass


# --------------------------------------------------------------------------- #
# Purchase orders
# --------------------------------------------------------------------------- #

def test_draft_order_keeps_only_lines_to_order():
    db = FakeSession()
    order = create_draft_order(db, uuid.uuid4(), [rec("Сүт", 12), rec("Тұз", 0), rec("Нан", 5)])
    assert [line["name"] for line in order.items] == ["Нан", "Сүт"]
    assert order.items[1]["quantity"] == 12
    assert db.added == [order]


def test_no_draft_order_when_nothing_to_order():
    db = FakeSession()
    assert create_draft_order(db, uuid.uuid4(), [rec("Тұз", 0)]) is None
    assert db.added == []


def test_order_text_and_whatsapp_link_round_trip():
    order = SimpleNamespace(
        created_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
        items=[{"name": "Сүт 2.5% & co", "quantity": 12}, {"name": "Нан", "quantity": 5}],
    )
    text = format_order_text("Береке", order)
    assert "«Береке» дүкенінен тапсырыс (15.10.2026)" in text
    assert "1. Сүт 2.5% & co — 12 дана" in text

    url = whatsapp_share_url(text)
    assert url.startswith("https://wa.me/?text=")
    assert parse_qs(urlparse(url).query)["text"][0] == text


def test_whatsapp_link_omitted_when_too_long():
    assert whatsapp_share_url("x" * MAX_WHATSAPP_URL_LENGTH) is None


# --------------------------------------------------------------------------- #
# Bot keyboard
# --------------------------------------------------------------------------- #

def test_confirm_button_fits_telegram_callback_limit():
    order_id = uuid.uuid4()
    button = confirm_keyboard(order_id).inline_keyboard[0][0]

    assert button.text == CONFIRM_BUTTON_TEXT
    assert len(button.callback_data.encode()) <= 64
    parsed = OrderAction.unpack(button.callback_data)
    assert parsed.action == "confirm"
    assert uuid.UUID(hex=parsed.order_id) == order_id


# --------------------------------------------------------------------------- #
# FastAPI wiring (no lifespan: no DB or Telegram needed)
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "ADMIN_API_KEY", SecretStr("admin-secret"))
    monkeypatch.setattr(main.settings, "TELEGRAM_WEBHOOK_URL", "https://example.com/telegram/webhook")
    monkeypatch.setattr(main.settings, "TELEGRAM_WEBHOOK_SECRET", SecretStr("hook-secret"))
    main.app.state.bot = object()
    main.app.state.dispatcher = object()
    yield TestClient(main.app)  # not used as a context manager, so lifespan does not run
    main.app.state.bot = main.app.state.dispatcher = None


def test_require_admin_rejects_bad_key(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "ADMIN_API_KEY", SecretStr("admin-secret"))
    with pytest.raises(HTTPException) as exc:
        main.require_admin("wrong")
    assert exc.value.status_code == 401
    main.require_admin("admin-secret")


def test_admin_api_disabled_without_key(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "ADMIN_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        main.require_admin("anything")
    assert exc.value.status_code == 503


def test_link_endpoint_requires_admin_key(client):
    response = client.post(f"/stores/{uuid.uuid4()}/telegram-link")
    assert response.status_code == 401


@pytest.mark.parametrize("header", [None, "wrong"])
def test_webhook_rejects_missing_or_wrong_secret(client, header):
    headers = {"X-Telegram-Bot-Api-Secret-Token": header} if header else {}
    response = client.post("/telegram/webhook", json={"update_id": 1}, headers=headers)
    assert response.status_code == 403
