import asyncio
import contextlib
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

from aiogram import Bot, Dispatcher
from aiogram.types import Update
from aiogram.utils.deep_linking import create_start_link
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.bot import create_bot, create_dispatcher, register_commands, send_morning_reports
from app.config import get_settings
from app.database import check_db_connection, engine, get_db
from app.services.telegram_link_service import issue_link_token

logger = logging.getLogger("uvicorn.error")
settings = get_settings()


# --------------------------------------------------------------------------- #
# Lifespan: database -> bot -> scheduler
# --------------------------------------------------------------------------- #

def _start_scheduler(bot: Bot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.STORE_TIMEZONE)
    scheduler.add_job(
        send_morning_reports,
        CronTrigger(hour=settings.MORNING_REPORT_HOUR, minute=settings.MORNING_REPORT_MINUTE),
        args=[bot],
        id="morning_reports",
        max_instances=1,  # never overlap if a run takes long
        coalesce=True,  # after downtime, run once, not once per missed day
        misfire_grace_time=3600,  # still send if the app was briefly down at 08:00
    )
    scheduler.start()
    logger.info(
        "Scheduler started: morning reports at %02d:%02d %s",
        settings.MORNING_REPORT_HOUR, settings.MORNING_REPORT_MINUTE, settings.STORE_TIMEZONE,
    )
    return scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: refuse to start if the database is unreachable, so a broken
    # deploy is caught by the orchestrator instead of serving 500s.
    try:
        check_db_connection()
        logger.info("Database connection OK")
    except SQLAlchemyError:
        logger.exception("Database connection failed on startup")
        raise

    bot: Bot | None = None
    dp: Dispatcher | None = None
    scheduler: AsyncIOScheduler | None = None
    polling: asyncio.Task | None = None

    if settings.TELEGRAM_BOT_TOKEN is None:
        logger.warning("TELEGRAM_BOT_TOKEN not set: Telegram bot and morning reports are disabled")
    else:
        bot = create_bot(settings.TELEGRAM_BOT_TOKEN.get_secret_value())
        dp = create_dispatcher()
        await register_commands(bot)

        if settings.TELEGRAM_WEBHOOK_URL:
            if settings.TELEGRAM_WEBHOOK_SECRET is None:
                raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is required in webhook mode")
            await bot.set_webhook(
                settings.TELEGRAM_WEBHOOK_URL,
                secret_token=settings.TELEGRAM_WEBHOOK_SECRET.get_secret_value(),
                allowed_updates=dp.resolve_used_update_types(),
            )
            logger.info("Telegram bot: webhook mode (%s)", settings.TELEGRAM_WEBHOOK_URL)
        else:
            await bot.delete_webhook()  # polling and webhook are mutually exclusive
            polling = asyncio.create_task(dp.start_polling(bot, handle_signals=False, close_bot_session=False))
            logger.info("Telegram bot: long-polling mode")

        if settings.SCHEDULER_ENABLED:
            scheduler = _start_scheduler(bot)

    app.state.bot, app.state.dispatcher = bot, dp
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        if polling is not None:
            await dp.stop_polling()
            with contextlib.suppress(asyncio.CancelledError):
                await polling
        if bot is not None:
            await bot.session.close()
        engine.dispose()


app = FastAPI(title=settings.APP_NAME, version="0.4.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["system"])
def health(request: Request) -> JSONResponse:
    bot_state = "enabled" if getattr(request.app.state, "bot", None) else "disabled"
    try:
        check_db_connection()
    except SQLAlchemyError:
        logger.exception("Health check: database unreachable")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "database": "unreachable", "bot": bot_state},
        )
    return JSONResponse(content={"status": "ok", "database": "ok", "bot": bot_state})


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #

@app.post("/telegram/webhook", include_in_schema=False)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    bot: Bot | None = request.app.state.bot
    dp: Dispatcher | None = request.app.state.dispatcher
    if bot is None or dp is None or not settings.TELEGRAM_WEBHOOK_URL:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    # Telegram echoes the secret set in set_webhook; anything else is not Telegram.
    expected = settings.TELEGRAM_WEBHOOK_SECRET.get_secret_value()
    if not x_telegram_bot_api_secret_token or not secrets.compare_digest(x_telegram_bot_api_secret_token, expected):
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    update = Update.model_validate(await request.json(), context={"bot": bot})
    await dp.feed_update(bot, update)
    return {"ok": True}


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    if settings.ADMIN_API_KEY is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin API disabled: set ADMIN_API_KEY")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.ADMIN_API_KEY.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin key")


class TelegramLinkOut(BaseModel):
    link: str
    expires_in_hours: int


@app.post(
    "/stores/{store_id}/telegram-link",
    response_model=TelegramLinkOut,
    tags=["telegram"],
    dependencies=[Depends(require_admin)],
)
async def create_telegram_link(store_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> TelegramLinkOut:
    """Issue a one-time deep link for the store owner to connect their Telegram chat."""
    bot: Bot | None = request.app.state.bot
    if bot is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telegram bot is disabled")
    ttl = timedelta(hours=settings.TELEGRAM_LINK_TTL_HOURS)
    token = await asyncio.to_thread(issue_link_token, db, store_id, ttl)
    if token is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Store not found")
    return TelegramLinkOut(
        link=await create_start_link(bot, token), expires_in_hours=settings.TELEGRAM_LINK_TTL_HOURS
    )
