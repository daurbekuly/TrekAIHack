import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

# Money is stored as NUMERIC, never FLOAT: binary floats accumulate rounding
# errors (0.1 + 0.2 != 0.3), which breaks invoice totals and margin reports.
Money = Numeric(12, 2)


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Store(Base):
    """A tenant: one SME shop. Every other entity is scoped by store_id."""

    __tablename__ = "stores"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(100), default="Taraz", server_default="Taraz")
    address: Mapped[str] = mapped_column(String(500))
    owner_contact: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Owner's Telegram chat. Not unique: one owner may run several stores.
    telegram_chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    # One-time deep-link code (t.me/<bot>?start=<code>) proving the chat belongs
    # to this store's owner. Cleared once used.
    telegram_link_token: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    telegram_link_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    products: Mapped[list["Product"]] = relationship(
        back_populates="store", cascade="all, delete-orphan", passive_deletes=True
    )
    inbound_invoices: Mapped[list["InboundInvoice"]] = relationship(
        back_populates="store", cascade="all, delete-orphan", passive_deletes=True
    )
    sales: Mapped[list["SalesHistory"]] = relationship(
        back_populates="store", cascade="all, delete-orphan", passive_deletes=True
    )
    inventory_counts: Mapped[list["InventoryCount"]] = relationship(
        back_populates="store", cascade="all, delete-orphan", passive_deletes=True
    )
    purchase_orders: Mapped[list["PurchaseOrder"]] = relationship(
        back_populates="store", cascade="all, delete-orphan", passive_deletes=True
    )


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        # The same barcode may exist in many stores, but only once per store.
        # NULL barcodes are allowed and never collide in PostgreSQL.
        UniqueConstraint("store_id", "barcode", name="uq_products_store_id_barcode"),
        CheckConstraint("price >= 0", name="price_non_negative"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    store_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stores.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    barcode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price: Mapped[Decimal] = mapped_column(Money)
    # Not constrained to >= 0 on purpose: real POS data often goes negative
    # (sales recorded before the supplier invoice is entered). We want to
    # surface that as a data-quality signal, not reject the write.
    current_stock: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    store: Mapped["Store"] = relationship(back_populates="products")
    invoice_items: Mapped[list["InvoiceItem"]] = relationship(back_populates="product")
    sales: Mapped[list["SalesHistory"]] = relationship(back_populates="product")
    inventory_counts: Mapped[list["InventoryCount"]] = relationship(back_populates="product")


class InboundInvoice(Base):
    """Supplier delivery note (кіріс накладная) — the inflow side of the stock balance."""

    __tablename__ = "inbound_invoices"
    __table_args__ = (
        Index("ix_inbound_invoices_store_id_invoice_date", "store_id", "invoice_date"),
        CheckConstraint("total_amount >= 0", name="total_amount_non_negative"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    store_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("stores.id", ondelete="CASCADE"))
    supplier_name: Mapped[str] = mapped_column(String(255))
    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    total_amount: Mapped[Decimal] = mapped_column(Money)

    store: Mapped["Store"] = relationship(back_populates="inbound_invoices")
    items: Mapped[list["InvoiceItem"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", passive_deletes=True
    )


class InvoiceItem(Base):
    __tablename__ = "invoice_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("purchase_price >= 0", name="purchase_price_non_negative"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inbound_invoices.id", ondelete="CASCADE"), index=True
    )
    # RESTRICT: a product with purchase history must not silently vanish.
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT"), index=True
    )
    quantity: Mapped[int] = mapped_column(Integer)
    purchase_price: Mapped[Decimal] = mapped_column(Money)

    invoice: Mapped["InboundInvoice"] = relationship(back_populates="items")
    product: Mapped["Product"] = relationship(back_populates="invoice_items")


class SalesHistory(Base):
    """Append-only sales log — the primary time series for demand forecasting.

    Uses a BIGINT identity PK instead of UUID: this is by far the largest table,
    rows are never referenced externally, and sequential keys keep the B-tree
    index compact and insert-friendly.
    """

    __tablename__ = "sales_history"
    __table_args__ = (
        # Main forecasting access path: "sales of product X in store Y over time".
        Index(
            "ix_sales_history_store_id_product_id_timestamp",
            "store_id",
            "product_id",
            "timestamp",
        ),
        # Store-wide aggregations: "all sales in store Y for last 30 days".
        Index("ix_sales_history_store_id_timestamp", "store_id", "timestamp"),
        CheckConstraint("quantity_sold > 0", name="quantity_sold_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    store_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("stores.id", ondelete="CASCADE"))
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT"), index=True
    )
    quantity_sold: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    store: Mapped["Store"] = relationship(back_populates="sales")
    product: Mapped["Product"] = relationship(back_populates="sales")


class InventoryCount(Base):
    """Physical stock count (ревизия / түгендеу) of one product at a point in time.

    Two consecutive counts bracket a reconciliation window: everything that left
    the shelf between them and is not explained by POS sales is a shadow sale.
    `Product.current_stock` is a running book value and cannot serve as a count.
    """

    __tablename__ = "inventory_counts"
    __table_args__ = (
        Index(
            "ix_inventory_counts_store_id_product_id_counted_at",
            "store_id",
            "product_id",
            "counted_at",
        ),
        CheckConstraint("counted_quantity >= 0", name="counted_quantity_non_negative"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    store_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("stores.id", ondelete="CASCADE"))
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT")
    )
    counted_quantity: Mapped[int] = mapped_column(Integer)
    counted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    store: Mapped["Store"] = relationship(back_populates="inventory_counts")
    product: Mapped["Product"] = relationship(back_populates="inventory_counts")


class PurchaseOrder(Base):
    """A recommended order sent to the owner, and whether they confirmed it.

    Items are a JSON snapshot (product_id, name, quantity) taken when the
    recommendation was generated, so the confirmed order matches exactly what
    the owner saw even if product names or stock change later.
    """

    __tablename__ = "purchase_orders"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'confirmed')", name="status_valid"),
        Index("ix_purchase_orders_store_id_created_at", "store_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    store_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("stores.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="draft", server_default="draft")
    items: Mapped[list[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    store: Mapped["Store"] = relationship(back_populates="purchase_orders")
