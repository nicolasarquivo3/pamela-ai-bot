from fastapi import FastAPI, Header, HTTPException
from app.config import settings

def create_web_app(agent, telegram_app):
    app = FastAPI(title="Telegram AI Character")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/autonomous/tick")
    async def autonomous_tick(x_autonomy_token: str | None = Header(default=None)):
        if x_autonomy_token != settings.autonomy_token:
            raise HTTPException(status_code=401, detail="unauthorized")
        return await agent.autonomous_tick()

    # Backward-compatible endpoint for an existing GitHub workflow.
    @app.post("/internal/tick")
    async def internal_tick(x_autonomy_token: str | None = Header(default=None)):
        if x_autonomy_token != settings.autonomy_token:
            raise HTTPException(status_code=401, detail="unauthorized")
        return await agent.autonomous_tick()

    @app.post("/telegram/webhook")
    async def webhook(
        update: dict,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ):
        if x_telegram_bot_api_secret_token != settings.webhook_secret:
            raise HTTPException(status_code=401, detail="unauthorized")
        await telegram_app.feed_webhook_update(update)
        return {"ok": True}

    return app
