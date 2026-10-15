# Inventory Forecasting API

Predictive inventory for SME shops: shadow-sales reconciliation, Prophet demand
forecasts, LLM purchase recommendations delivered through a Telegram bot.

## Layout

```
app/
├── main.py                 FastAPI app: lifespan (DB → bot → scheduler), /health, webhook, admin
├── bot.py                  aiogram router: /start, /forecast, order confirmation, morning job
├── config.py               settings from .env
├── database.py             engine, session, Base
├── models.py               ORM models
└── services/
    ├── discrepancy_service.py     Phase 2: shadow sales
    ├── ai_forecasting_service.py  Phase 3: Prophet + OpenAI insights
    ├── purchase_order_service.py  order snapshot, confirmation, WhatsApp text
    └── telegram_link_service.py   store ↔ Telegram chat linking
alembic/                    migrations
tests/
```

## Setup

Requires Python 3.10+ and PostgreSQL.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Edit `.env`:

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | `postgresql+psycopg://user:pass@host:5432/db` |
| `TELEGRAM_BOT_TOKEN` | for the bot | Get it from [@BotFather](https://t.me/BotFather): `/newbot` |
| `ADMIN_API_KEY` | to link stores | Any long random string |
| `OPENAI_API_KEY` | optional | Without it, reports use a built-in template |
| `TELEGRAM_WEBHOOK_URL` / `TELEGRAM_WEBHOOK_SECRET` | production | Leave empty locally to use long polling |

Create the schema:

```bash
alembic revision --autogenerate -m "initial schema"   # first time only
alembic upgrade head
```

Optional demo data (a store with 6 products and 90 days of history; prints the STORE ID):

```bash
python -m scripts.seed_demo
```

## Run

```bash
uvicorn app.main:app --reload
```

`GET http://localhost:8000/health` → `{"status": "ok", "database": "ok", "bot": "enabled"}`.

With `TELEGRAM_WEBHOOK_URL` empty the bot uses long polling, so no public URL
is needed for local development.

## Connect a store owner

1. Issue a one-time link (valid 24 h):
   ```bash
   curl -X POST http://localhost:8000/stores/<store_uuid>/telegram-link -H "X-Admin-Key: <ADMIN_API_KEY>"
   # {"link": "https://t.me/YourBot?start=AbC...", "expires_in_hours": 24}
   ```
2. Send the link to the owner. Opening it and pressing **Start** links their chat to the store.
3. The owner can now send `/forecast` at any time, and receives a report every day at 08:00 (store time).

Under each report with something to order there is a **✅ Тапсырысты растау / WhatsApp-қа жіберу**
button. It confirms the order and returns the order text plus a WhatsApp link with the text prefilled;
the owner chooses the distributor's chat and sends it.

## Production notes

- **Webhook mode:** set `TELEGRAM_WEBHOOK_URL=https://your.domain/telegram/webhook` and
  `TELEGRAM_WEBHOOK_SECRET`. The app registers the webhook on startup and rejects requests without the secret.
- **One scheduler only:** each process with `SCHEDULER_ENABLED=true` sends its own morning reports.
  With several uvicorn workers or replicas, enable it on exactly one, or owners get duplicates.
  Likewise run long polling in a single process only.
- Reports are generated one store at a time; Prophet fitting is CPU-bound, so large tenants
  should eventually move this to a worker queue.

## Tests

```bash
pytest
```
