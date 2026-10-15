"""Linking a store to its owner's Telegram chat.

Flow: an admin issues a one-time code for a store (POST /stores/{id}/telegram-link),
the owner opens t.me/<bot>?start=<code>, and the bot's /start handler calls
`link_chat`. The code proves the chat belongs to that store's owner; a bare
/start cannot claim a store.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Store


def issue_link_token(db: Session, store_id: uuid.UUID, ttl: timedelta) -> str | None:
    """Create (or replace) the store's link code. Returns None if the store doesn't exist."""
    store = db.get(Store, store_id)
    if store is None:
        return None
    # 16 random bytes -> 22 URL-safe chars, within Telegram's 64-char start-param limit.
    store.telegram_link_token = secrets.token_urlsafe(16)
    store.telegram_link_expires_at = datetime.now(timezone.utc) + ttl
    db.commit()
    return store.telegram_link_token


def link_chat(db: Session, token: str, chat_id: int) -> Store | None:
    """Attach `chat_id` to the store owning `token`. Returns None if the code is invalid or expired."""
    store = db.scalar(select(Store).where(Store.telegram_link_token == token).with_for_update())
    if store is None:
        return None
    if store.telegram_link_expires_at is None or store.telegram_link_expires_at < datetime.now(timezone.utc):
        return None
    store.telegram_chat_id = chat_id
    store.telegram_link_token = None  # one-time use
    store.telegram_link_expires_at = None
    db.commit()
    return store


def stores_for_chat(db: Session, chat_id: int) -> list[Store]:
    return list(db.scalars(select(Store).where(Store.telegram_chat_id == chat_id).order_by(Store.name)))


def linked_stores(db: Session) -> list[Store]:
    return list(db.scalars(select(Store).where(Store.telegram_chat_id.is_not(None)).order_by(Store.id)))


def unlink_chat(db: Session, chat_id: int) -> None:
    """Detach a chat from all its stores, e.g. after the owner blocked the bot."""
    for store in stores_for_chat(db, chat_id):
        store.telegram_chat_id = None
    db.commit()
