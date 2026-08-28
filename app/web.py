"""
FastAPI + webhook + cron de tag Drive.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, Request, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

try:
    from app.config import settings
except Exception:
    settings = None  # type: ignore

# preenchido no main apos create
_DRIVE_REF = {"svc": None}


def set_drive_service(svc):
    _DRIVE_REF["svc"] = svc


def create_web_app(*args, **kwargs):
    telegram_app = kwargs.get("telegram_app")
    if telegram_app is None:
        for a in args:
            if a is not None and hasattr(a, "feed_webhook_update"):
                telegram_app = a
                break
        if telegram_app is None and args:
            telegram_app = args[0] if hasattr(args[0], "feed_webhook_update") else (
                args[1] if len(args) > 1 else None
            )
    if telegram_app is None:
        raise TypeError("create_web_app: precisa de telegram_app")

    app = FastAPI(title="pamela-ai")

    @app.get("/")
    async def root():
        return PlainTextResponse("pamela-ai ok", status_code=200)

    @app.get("/health")
    async def health():
        return {"ok": True, "service": "pamela-ai"}

    @app.get("/cron/drive_tag")
    async def cron_drive_tag(
        n: int = Query(default=8, ge=1, le=30),
        key: str | None = Query(default=None),
        force: int = Query(default=0, ge=0, le=1),
    ):
        """
        Chame a cada 5-10 min (UptimeRobot free) para:
        1) acordar o Render
        2) taguear N fotos do Drive com Gemini
        Opcional: ?key=WEBHOOK_SECRET
        """
        secret = ""
        if settings is not None:
            secret = (getattr(settings, "webhook_secret", None) or "").strip()
        if secret and secret not in ("change-me", "changeme", "") and key is not None:
            if key != secret:
                raise HTTPException(403, "bad key")

        drive = _DRIVE_REF.get("svc")
        if drive is None:
            return {"ok": False, "error": "drive_service not set"}
        try:
            # por padrao so tagueia no ocio (nao atrasa conversa)
            force = False
            try:
                from fastapi import Request as _R
            except Exception:
                pass
            # query force ja via n; check idle
            try:
                from app.runtime_flags import telegram_busy, recently_active
                if not force and (telegram_busy() or recently_active(90)):
                    print("[CRON] skip tag — usuario em conversa", flush=True)
                    return {
                        "ok": True,
                        "tagged": 0,
                        "skipped": "user_active",
                        "hint": "tag so no ocio; ?force=1 para forcar",
                    }
            except Exception:
                pass
            if hasattr(drive, "_ensure_caption_fn"):
                drive._ensure_caption_fn()
            done = await drive.backfill_captions(limit=n)
            st = {}
            if hasattr(drive, "stats"):
                try:
                    st = await drive.stats()
                except Exception:
                    pass
            print(f"[CRON] drive_tag done={done} stats={st}", flush=True)
            return {"ok": True, "tagged": done, "stats": st}
        except Exception as e:
            print(f"[CRON] drive_tag err: {e}", flush=True)
            return {"ok": False, "error": str(e)[:300]}

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
