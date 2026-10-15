"""Telegram bot: store linking, on-demand and scheduled purchase reports, order confirmation.

Commands:
    /start <code>  link this chat to a store (code issued via POST /stores/{id}/telegram-link)
    /start         greeting + which stores this chat is linked to
    /forecast      generate purchase recommendations now

All database and forecasting work is synchronous (SQLAlchemy, Prophet, OpenAI),
so it runs in worker threads via asyncio.to_thread and never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import html
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.config import get_settings
from app.database import SessionLocal
from app.services import telegram_link_service as links
from app.services.ai_forecasting_service import generate_purchase_recommendations
from app.services.purchase_order_service import (
    confirm_order,
    create_draft_order,
    format_order_text,
    whatsapp_share_url,
)

logger = logging.getLogger(__name__)
router = Router(name="inventory")

CONFIRM_BUTTON_TEXT = "✅ Тапсырысты растау / WhatsApp-қа жіберу"

T = TypeVar("T")

# Chats with a /forecast already running, to ignore repeated taps.
_busy_chats: set[int] = set()
# Strong refs so fire-and-forget tasks are not garbage-collected mid-run.
_background_tasks: set[asyncio.Task] = set()


class OrderAction(CallbackData, prefix="po"):
    action: str
    order_id: str  # UUID hex; keeps callback_data well under Telegram's 64-byte limit


# --------------------------------------------------------------------------- #
# Sync helpers (run in worker threads, one DB session each)
# --------------------------------------------------------------------------- #

def _link_chat(token: str, chat_id: int) -> str | None:
    with SessionLocal() as db:
        store = links.link_chat(db, token, chat_id)
        return store.name if store else None


def _stores_for_chat(chat_id: int) -> list[tuple[uuid.UUID, str]]:
    with SessionLocal() as db:
        return [(s.id, s.name) for s in links.stores_for_chat(db, chat_id)]


def _linked_stores() -> list[tuple[uuid.UUID, int]]:
    with SessionLocal() as db:
        return [(s.id, s.telegram_chat_id) for s in links.linked_stores(db)]


def _unlink_chat(chat_id: int) -> None:
    with SessionLocal() as db:
        links.unlink_chat(db, chat_id)


def _prepare_report(store_id: uuid.UUID) -> tuple[str, uuid.UUID | None]:
    """Run forecasting for one store and snapshot the order. Returns (message, order_id)."""
    with SessionLocal() as db:
        report = generate_purchase_recommendations(db, store_id, language=get_settings().REPORT_LANGUAGE)
        order = create_draft_order(db, store_id, report.items)
        return report.message, (order.id if order else None)


def _confirm(order_id: uuid.UUID, chat_id: int) -> tuple[bool, str] | None:
    """Returns (already_confirmed, order_text), or None if the order isn't this chat's."""
    with SessionLocal() as db:
        result = confirm_order(db, order_id, chat_id)
        if result is None:
            return None
        return result.already_confirmed, format_order_text(result.store_name, result.order)


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #

async def _with_flood_retry(call: Callable[[], Awaitable[T]]) -> T:
    """Retry once after Telegram's flood-control wait."""
    try:
        return await call()
    except TelegramRetryAfter as exc:
        await asyncio.sleep(exc.retry_after)
        return await call()


def confirm_keyboard(order_id: uuid.UUID) -> InlineKeyboardMarkup:
    data = OrderAction(action="confirm", order_id=order_id.hex).pack()
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=CONFIRM_BUTTON_TEXT, callback_data=data)]])


async def deliver_report(bot: Bot, store_id: uuid.UUID, chat_id: int) -> None:
    """Generate one store's recommendations and send them, with the confirm button if there is anything to order."""
    message, order_id = await asyncio.to_thread(_prepare_report, store_id)
    markup = confirm_keyboard(order_id) if order_id else None
    await _with_flood_retry(lambda: bot.send_message(chat_id, message, reply_markup=markup))


async def send_morning_reports(bot: Bot) -> None:
    """Scheduled job: one report per linked store. A failing store never stops the rest."""
    stores = await asyncio.to_thread(_linked_stores)
    logger.info("Morning reports: %d linked stores", len(stores))
    for store_id, chat_id in stores:
        try:
            await deliver_report(bot, store_id, chat_id)
        except TelegramForbiddenError:
            logger.warning("Chat %s blocked the bot; unlinking its stores", chat_id)
            await asyncio.to_thread(_unlink_chat, chat_id)
        except Exception:
            logger.exception("Morning report failed for store %s", store_id)


def _spawn(coro: Awaitable[None]) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

@router.message(CommandStart(deep_link=True))
async def start_with_code(message: Message, command: CommandObject) -> None:
    store_name = await asyncio.to_thread(_link_chat, command.args or "", message.chat.id)
    if store_name is None:
        await message.answer(
            "❌ Сілтеме жарамсыз немесе мерзімі өтіп кеткен. Жаңа сілтеме сұраңыз.\n"
            "❌ Ссылка недействительна или устарела. Запросите новую."
        )
        return
    await message.answer(
        f"✅ <b>{html.escape(store_name)}</b> дүкені қосылды!\n\n"
        "Күн сайын таңертең сатып алу ұсыныстарын осында аласыз.\n"
        "Қазір алу үшін: /forecast"
    )


@router.message(CommandStart())
async def start(message: Message) -> None:
    stores = await asyncio.to_thread(_stores_for_chat, message.chat.id)
    if not stores:
        await message.answer(
            "Сәлеметсіз бе! 👋 Бұл бот дүкеніңіздің қорын болжап, тапсырыс ұсынады.\n\n"
            "Дүкенді қосу үшін әкімші берген арнайы сілтемені ашыңыз.\n"
            "Чтобы подключить магазин, откройте ссылку, которую выдал администратор."
        )
        return
    names = "\n".join(f"• {html.escape(name)}" for _, name in stores)
    await message.answer(f"Қосылған дүкендер:\n{names}\n\nҰсыныс алу: /forecast")


@router.message(Command("forecast"))
async def forecast(message: Message, bot: Bot) -> None:
    chat_id = message.chat.id
    stores = await asyncio.to_thread(_stores_for_chat, chat_id)
    if not stores:
        await message.answer("Бұл чатқа дүкен қосылмаған. Алдымен /start сілтемесін пайдаланыңыз.")
        return
    if chat_id in _busy_chats:
        await message.answer("⏳ Есептеу жүріп жатыр, сәл күтіңіз…")
        return

    await message.answer("⏳ Болжам есептелуде, бір минуттай уақыт алуы мүмкін…")
    _spawn(_forecast_for_chat(bot, chat_id, [store_id for store_id, _ in stores]))


async def _forecast_for_chat(bot: Bot, chat_id: int, store_ids: list[uuid.UUID]) -> None:
    # Runs detached from the update so a slow forecast never holds a webhook request open.
    _busy_chats.add(chat_id)
    try:
        for store_id in store_ids:
            try:
                await deliver_report(bot, store_id, chat_id)
            except Exception:
                logger.exception("On-demand report failed for store %s", store_id)
                await bot.send_message(chat_id, "⚠️ Болжамды есептеу мүмкін болмады. Кейінірек қайталаңыз.")
    finally:
        _busy_chats.discard(chat_id)


@router.callback_query(OrderAction.filter(F.action == "confirm"))
async def confirm(query: CallbackQuery, callback_data: OrderAction) -> None:
    message = query.message if isinstance(query.message, Message) else None
    chat_id = message.chat.id if message else query.from_user.id
    try:
        order_id = uuid.UUID(hex=callback_data.order_id)
    except ValueError:
        await query.answer("Тапсырыс табылмады", show_alert=True)
        return

    result = await asyncio.to_thread(_confirm, order_id, chat_id)
    if result is None:
        await query.answer("Тапсырыс табылмады", show_alert=True)
        return
    already_confirmed, order_text = result

    if message is not None:
        try:
            await message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass  # already removed, or message too old to edit
    if already_confirmed:
        await query.answer("Бұл тапсырыс бұрын расталған")
        return
    await query.answer("✅ Расталды")

    # Simulated hand-off to the distributor: a WhatsApp share link with the
    # order prefilled. The owner picks the distributor's chat and taps send.
    url = whatsapp_share_url(order_text)
    markup = (
        InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="📲 WhatsApp-та ашу", url=url)]])
        if url
        else None
    )
    await _with_flood_retry(lambda: query.bot.send_message(
        chat_id,
        f"✅ Тапсырыс расталды. Дистрибьюторға жіберіңіз:\n\n<pre>{html.escape(order_text)}</pre>",
        reply_markup=markup,
    ))


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

def create_bot(token: str) -> Bot:
    return Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def register_commands(bot: Bot) -> None:
    await bot.set_my_commands([
        BotCommand(command="start", description="Дүкенді қосу / Подключить магазин"),
        BotCommand(command="forecast", description="Тапсырыс ұсынысы / Рекомендация закупа"),
    ])
