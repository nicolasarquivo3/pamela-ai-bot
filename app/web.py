"""
FastAPI app + webhook Telegram.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import settings


def create_web_app(telegram_app=None, *args, **kwargs):
    """
    Compat:
      create_web_app(telegram_app)
      create_web_app(app, telegram_app)  # se algum codigo antigo passar 2 args
      create_web_app(telegram_app=...)
    """
    # se passaram 2 args posicionais (algo, telegram_app)
    if args and telegram_app is not None and not hasattr(telegram_app, "feed_webhook_update"):
        # primeiro era lixo/app, segundo real
        if hasattr(args[0], "feed_webhook_update"):
            telegram_app = args[0]
    if telegram_app is None:
        telegram_app = kwargs.get("telegram_app")
    if telegram_app is None and args:
        for a in args:
            if hasattr(a, "feed_webhook_update"):
                telegram_app = a
                break

    if telegram_app is None:
        raise TypeError(
            "create_web_app() missing required telegram_app "
            "(passe o TelegramApp)"
        )

    app = FastAPI(title="pamela-ai")

    @app.get("/")
    async def root():
        return PlainTextResponse("pamela-ai ok", status_code=200)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/telegram/webhook")
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ):
        secret = (getattr(settings, "webhook_secret", None) or "").strip()
        if secret and secret != "change-me":
            if (x_telegram_bot_api_secret_token or "") != secret:
                raise HTTPException(status_code=403, detail="bad secret")

        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid json")

        try:
            await telegram_app.feed_webhook_update(data)
        except Exception as e:
            print(f"[WEB] webhook handler error: {e}", flush=True)
            # Telegram reenvia se 5xx; respondemos 200 pra nao flood
            return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=200)

        return {"ok": True}

    return app
