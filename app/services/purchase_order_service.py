"""Purchase orders: snapshot recommended quantities, confirm, format for the distributor."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote

from sqlalchemy.orm import Session

from app.models import PurchaseOrder, Store
from app.services.ai_forecasting_service import ProductRecommendation

# wa.me links much longer than this get truncated by some clients; the order
# text is always sent as a plain message too, so the link is a convenience.
MAX_WHATSAPP_URL_LENGTH = 2000


@dataclass(frozen=True)
class ConfirmResult:
    order: PurchaseOrder
    store_name: str
    already_confirmed: bool


def create_draft_order(
    db: Session, store_id: uuid.UUID, items: list[ProductRecommendation]
) -> PurchaseOrder | None:
    """Persist the lines with order_qty > 0. Returns None when there is nothing to order."""
    lines = [
        {"product_id": str(i.product_id), "name": i.name, "quantity": i.order_qty}
        for i in sorted(items, key=lambda i: i.name)
        if i.order_qty > 0
    ]
    if not lines:
        return None
    order = PurchaseOrder(store_id=store_id, items=lines)
    db.add(order)
    db.commit()
    return order


def confirm_order(db: Session, order_id: uuid.UUID, chat_id: int) -> ConfirmResult | None:
    """Mark an order confirmed. Idempotent; returns None if the chat doesn't own the order."""
    order = db.get(PurchaseOrder, order_id, with_for_update=True)
    if order is None:
        return None
    store = db.get(Store, order.store_id)
    if store is None or store.telegram_chat_id != chat_id:
        return None

    already = order.status == "confirmed"
    if not already:
        order.status = "confirmed"
        order.confirmed_at = datetime.now(timezone.utc)
        db.commit()
    return ConfirmResult(order=order, store_name=store.name, already_confirmed=already)


def format_order_text(store_name: str, order: PurchaseOrder) -> str:
    """Plain-text order for the distributor (WhatsApp does not render HTML)."""
    created = order.created_at or datetime.now(timezone.utc)
    lines = [
        "Сәлеметсіз бе!",
        f"«{store_name}» дүкенінен тапсырыс ({created:%d.%m.%Y}):",
        "",
    ]
    lines += [f"{n}. {item['name']} — {item['quantity']} дана" for n, item in enumerate(order.items, 1)]
    lines += ["", "Рахмет!"]
    return "\n".join(lines)


def whatsapp_share_url(text: str) -> str | None:
    """wa.me link that opens WhatsApp with `text` prefilled; the owner picks the distributor."""
    url = "https://wa.me/?text=" + quote(text, safe="")
    return url if len(url) <= MAX_WHATSAPP_URL_LENGTH else None
