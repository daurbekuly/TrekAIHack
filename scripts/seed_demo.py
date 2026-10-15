"""Fill the database with a demo store: products, 90 days of sales, invoices, stock counts.

Run from the project root:
    python -m scripts.seed_demo

Safe to run twice: if the demo store already exists, it only prints its id.
"""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.models import (
    InboundInvoice,
    InventoryCount,
    InvoiceItem,
    Product,
    SalesHistory,
    Store,
)

DEMO_STORE_NAME = "Демо дүкен"
HISTORY_DAYS = 90
SHADOW_SHARE = 0.08  # 8% of real sales never reach the POS

# name, price, avg units/day, current book stock (chosen to show every status)
PRODUCTS = [
    ("Сүт 2.5% 1л", "520", 14, 12),  # runs out in about a day
    ("Нан ақ", "180", 25, 60),  # runs out within the week
    ("Жұмыртқа 10 дана", "890", 6, 38),  # enough on average, reorder for safety
    ("Қант 1кг", "650", 3, 30),  # fine
    ("Тұз 1кг", "150", 1, 140),  # overstock
    ("Шай Пиала 250г", "1450", 2, 15),  # fine
]


def main() -> None:
    rng = np.random.default_rng(7)
    tz = ZoneInfo(get_settings().STORE_TIMEZONE)
    today = datetime.now(tz).date()
    first_day = today - timedelta(days=HISTORY_DAYS)

    with SessionLocal() as db:
        existing = db.scalar(select(Store).where(Store.name == DEMO_STORE_NAME))
        if existing is not None:
            print(f"Demo store already exists. STORE ID: {existing.id}")
            return

        store = Store(name=DEMO_STORE_NAME, address="Төле би 1", owner_contact="+7 700 000 00 00")
        db.add(store)
        db.flush()

        start_count_at = datetime.combine(first_day, time(8, 0), tz).astimezone(timezone.utc)
        end_count_at = datetime.combine(today - timedelta(days=1), time(22, 0), tz).astimezone(timezone.utc)
        weekly_invoices: dict[int, InboundInvoice] = {}

        for name, price, per_day, book_stock in PRODUCTS:
            product = Product(
                store_id=store.id, name=name, price=Decimal(price), current_stock=book_stock
            )
            db.add(product)
            db.flush()

            start_qty = per_day * 14
            pos_total = shadow_total = inbound_total = 0
            for offset in range(HISTORY_DAYS):
                day = first_day + timedelta(days=offset)
                weekend = 1.4 if day.weekday() >= 5 else 1.0
                sold = int(rng.poisson(per_day * weekend))
                shadow = int(rng.binomial(sold, SHADOW_SHARE))
                recorded = sold - shadow
                sale_at = datetime.combine(day, time(13, 0), tz).astimezone(timezone.utc)
                if recorded > 0 and start_count_at < sale_at <= end_count_at:
                    db.add(SalesHistory(
                        store_id=store.id, product_id=product.id,
                        quantity_sold=recorded, timestamp=sale_at,
                    ))
                    pos_total += recorded
                    shadow_total += shadow

                # Weekly delivery every Monday covering ~a week of demand.
                if day.weekday() == 0 and offset < HISTORY_DAYS - 1:
                    week = offset // 7
                    invoice = weekly_invoices.get(week)
                    if invoice is None:
                        invoice = InboundInvoice(
                            store_id=store.id, supplier_name="ТОО Тараз Дистрибуция",
                            invoice_date=datetime.combine(day, time(10, 0), tz).astimezone(timezone.utc),
                            total_amount=Decimal("0"),
                        )
                        db.add(invoice)
                        db.flush()
                        weekly_invoices[week] = invoice
                    qty = math.ceil(per_day * 7 * 1.2)
                    purchase_price = (Decimal(price) * Decimal("0.75")).quantize(Decimal("0.01"))
                    db.add(InvoiceItem(
                        invoice_id=invoice.id, product_id=product.id,
                        quantity=qty, purchase_price=purchase_price,
                    ))
                    invoice.total_amount += purchase_price * qty
                    inbound_total += qty

            end_qty = max(start_qty + inbound_total - pos_total - shadow_total, 0)
            db.add_all([
                InventoryCount(store_id=store.id, product_id=product.id,
                               counted_quantity=start_qty, counted_at=start_count_at),
                InventoryCount(store_id=store.id, product_id=product.id,
                               counted_quantity=end_qty, counted_at=end_count_at),
            ])
            print(f"  {name:<20} POS {pos_total:>5}  shadow {shadow_total:>4}  stock {book_stock}")

        db.commit()
        print(f"\nDemo store created. STORE ID: {store.id}")


if __name__ == "__main__":
    main()
