from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Inventory Forecasting API"
    DEBUG: bool = False
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/inventory"

    # Local business day boundaries for daily aggregation (Taraz = UTC+5).
    STORE_TIMEZONE: str = "Asia/Almaty"

    # LLM insights. Without a key, recommendations fall back to a plain template.
    OPENAI_API_KEY: SecretStr | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT_SECONDS: float = 30.0
    REPORT_LANGUAGE: str = "Russian"

    # Telegram bot. Without a token the API runs with the bot disabled.
    TELEGRAM_BOT_TOKEN: SecretStr | None = None
    # Set to run in webhook mode (production); leave empty for long polling (dev).
    TELEGRAM_WEBHOOK_URL: str | None = None
    TELEGRAM_WEBHOOK_SECRET: SecretStr | None = None
    TELEGRAM_LINK_TTL_HOURS: int = 24

    # Daily report schedule, in STORE_TIMEZONE. Enable on exactly one process.
    SCHEDULER_ENABLED: bool = True
    MORNING_REPORT_HOUR: int = 8
    MORNING_REPORT_MINUTE: int = 0

    # Protects admin endpoints (e.g. issuing Telegram link codes). Unset = disabled.
    ADMIN_API_KEY: SecretStr | None = None

    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    @field_validator(
        "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_URL",
        "TELEGRAM_WEBHOOK_SECRET", "ADMIN_API_KEY",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # `KEY=` in .env means "not set", not "set to empty string".
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
