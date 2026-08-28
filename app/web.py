"""
FastAPI + webhook Telegram.
Assinatura flexivel — NUNCA exige 2 args obrigatorios confusos.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

try:
    from app.config import settings
except Exception:
    settings = None  # type: ignore


def create_web_app(*args, **kwargs):
    """
    Aceita:
      create_web_app(telegram_app)
      create_web_app(telegram_app=tg)
      create_web_app(None, telegram_app)
      create_web_app(bot=..., telegram_app=...)
    """
    telegram_app = kwargs.get("telegram_app")
    if telegram_app is None:
        telegram_app = kwargs.get("tg")
    if telegram_app is None:
        telegram_app = kwargs.get("bot_app")
    if telegram_app is None and args:
        # pega o primeiro objeto que tenha feed_webhook_update
        for a in args:
            if a is not None and hasattr(a, "feed_webhook_update"):
                telegram_app = a
                break
        # se so passou 1 arg e e o telegram
        if telegram_app is None and len(args) == 1 and args[0] is not None:
            telegram_app = args[0]
        if telegram_app is None and len(args) >= 2:
            telegram_app = args[1]

    if telegram_app is None:
        raise TypeError(
            "create_web_app: passe telegram_app "
            "(ex: create_web_app(telegram_app) ou create_web_app(telegram_app=...))"
        )

    app = FastAPI(title="pamela-ai")

    @app.get("/")
    async def root():
        return PlainTextResponse("pamela-ai ok", status_code=200)

    @app.get("/health")
    async def health():
        return {"ok": True, "service": "pamela-ai"}

    @app.post("/telegram/webhook")
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ):
        secret = ""
        if settings is not None:
            secret = (getattr(settings, "webhook_secret", None) or "").strip()
        if secret and secret not in ("change-me", "changeme", ""):
            if (x_telegram_bot_api_secret_token or "") != secret:
                raise HTTPException(status_code=403, detail="bad secret")

        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid json")

        try:
            await telegram_app.feed_webhook_update(data)
        except Exception as e:
            print(f"[WEB] webhook error: {e}", flush=True)
            return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=200)

        return {"ok": True}

    return app
