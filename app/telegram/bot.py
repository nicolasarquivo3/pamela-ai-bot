"""
Telegram bot.
- 1 mensagem do usuario = 1 processamento (lock por user + dedup)
- Foto SEM legenda; texto em mensagem separada
- channel_post: indexa fotos do ALBUM_CHANNEL_ID
"""
from __future__ import annotations

import asyncio
from collections import deque

from aiogram import Bot, Dispatcher
from aiogram.types import Message, BufferedInputFile, Update
import httpx

from app.config import settings


class TelegramApp:

    def __init__(self, agent, album_service=None):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.agent = agent
        self.album_service = album_service

        self._user_locks: dict[int, asyncio.Lock] = {}
        self._seen_updates: deque[int] = deque(maxlen=800)
        self._seen_update_set: set[int] = set()
        self._seen_messages: deque[str] = deque(maxlen=800)
        self._seen_message_set: set[str] = set()

        @self.dp.channel_post()
        async def on_channel_post(message: Message):
            await self._handle_channel_post(message)

        @self.dp.message()
        async def handler(message: Message):
            if not message.from_user:
                return

            user_id = message.from_user.id
            msg_key = f"{user_id}:{message.message_id}"

            if msg_key in self._seen_message_set:
                print(
                    f"[TelegramApp] skip duplicata message_id={message.message_id}",
                    flush=True,
                )
                return
            self._mark_message(msg_key)

            lock = self._user_locks.setdefault(user_id, asyncio.Lock())
            async with lock:
                await self._handle_one(message)

    def _mark_message(self, key: str) -> None:
        if key in self._seen_message_set:
            return
        if len(self._seen_messages) >= 800:
            old = self._seen_messages.popleft()
            self._seen_message_set.discard(old)
        self._seen_messages.append(key)
        self._seen_message_set.add(key)

    def _mark_update(self, update_id: int | None) -> bool:
        if update_id is None:
            return False
        if update_id in self._seen_update_set:
            return True
        if len(self._seen_updates) >= 800:
            old = self._seen_updates.popleft()
            self._seen_update_set.discard(old)
        self._seen_updates.append(update_id)
        self._seen_update_set.add(update_id)
        return False

    async def _handle_channel_post(self, message: Message) -> None:
        if not self.album_service:
            return
        chat = message.chat
        chat_id = chat.id if chat else None
        if not self.album_service.is_album_channel(chat_id):
            print(
                f"[ALBUM] channel_post ignorado chat_id={chat_id} "
                f"(esperado {self.album_service.channel_id})",
                flush=True,
            )
            return

        photos = message.photo or []
        if not photos:
            # document image?
            doc = message.document
            if doc and (doc.mime_type or "").startswith("image/"):
                file_id = doc.file_id
                fuid = doc.file_unique_id
                w = h = None
            else:
                return
        else:
            # maior resolucao
            best = photos[-1]
            file_id = best.file_id
            fuid = best.file_unique_id
            w = best.width
            h = best.height

        gen_cap = bool(getattr(settings, "album_caption_on_ingest", False))
        ok = await self.album_service.ingest_telegram_photo(
            file_id=file_id,
            file_unique_id=fuid,
            caption_hint=(message.caption or ""),
            width=w,
            height=h,
            generate_caption=gen_cap,
        )
        n = await self.album_service.count()
        print(f"[ALBUM] channel_post ok={ok} total={n}", flush=True)

    async def _handle_one(self, message: Message) -> None:
        session = self.agent.context_manager.session
        try:
            text = message.text or message.caption or ""
            # comando album stats
            low = text.strip().lower()
            if low in ("/album", "/album_stats"):
                if self.album_service:
                    n = await self.album_service.count()
                    await message.answer(f"Album: {n} foto(s) indexada(s).")
                else:
                    await message.answer("Album desativado.")
                await session.commit()
                return
            if low.startswith("/album_tag"):
                # /album_tag ou /album_tag 30 — gera caption IA nas fotos sem tag
                if not self.album_service:
                    await message.answer("Album desativado.")
                    await session.commit()
                    return
                parts = low.split()
                lim = 20
                if len(parts) > 1 and parts[1].isdigit():
                    lim = min(int(parts[1]), 50)
                await message.answer(
                    f"Tagueando ate {lim} fotos sem legenda com IA... aguarde."
                )
                n = await self.album_service.backfill_captions(limit=lim)
                await message.answer(f"Pronto: {n} foto(s) com tag automatica.")
                await session.commit()
                return

            print(
                f"[TelegramApp] handle user={message.from_user.id} "
                f"msg={message.message_id} text={text[:80]!r}",
                flush=True,
            )

            result = await self.agent.receive_message(
                message.from_user.id,
                text,
            )

            await session.commit()

            if not isinstance(result, dict):
                return

            # texto primeiro (mensagem separada)
            reply = (result.get("text") or result.get("reply") or "").strip()
            if reply:
                await message.answer(reply)

            # foto: agent pode devolver type=image no root ou em "image"
            img = None
            if result.get("type") == "image" or result.get("telegram_file_id") or result.get("bytes") or result.get("url"):
                img = {
                    "success": result.get("success", True),
                    "image_url": result.get("url") or result.get("image_url"),
                    "image_bytes": result.get("bytes") or result.get("image_bytes"),
                    "telegram_file_id": result.get("telegram_file_id"),
                    "provider": result.get("provider"),
                    "caption": None,
                }
            elif result.get("image") or result.get("image_result"):
                raw = result.get("image") or result.get("image_result")
                if isinstance(raw, dict):
                    img = {
                        "success": raw.get("success", True),
                        "image_url": raw.get("url") or raw.get("image_url"),
                        "image_bytes": raw.get("bytes") or raw.get("image_bytes"),
                        "telegram_file_id": raw.get("telegram_file_id"),
                        "provider": raw.get("provider"),
                        "caption": None,
                    }
                else:
                    img = {
                        "success": getattr(raw, "success", False),
                        "image_url": getattr(raw, "image_url", None),
                        "image_bytes": getattr(raw, "image_bytes", None),
                        "telegram_file_id": getattr(raw, "telegram_file_id", None),
                        "provider": getattr(raw, "provider", None),
                        "caption": None,
                    }

            if img and (img.get("telegram_file_id") or img.get("image_bytes") or img.get("image_url")):
                await self._send_image_result(message, img)

        except Exception as e:
            print(f"[TelegramApp] handle error: {e}", flush=True)
            try:
                await session.rollback()
            except Exception:
                pass
            try:
                await message.answer("Amor, deu um probleminha aqui agora. Tenta de novo? ❤️")
            except Exception:
                pass

    async def _send_image_result(self, message: Message, result: dict) -> bool:
        if not result or not result.get("success"):
            return False

        # SEM legenda
        caption = None
        file_id = result.get("telegram_file_id")
        image_bytes = result.get("image_bytes")
        image_url = result.get("image_url")

        print(
            f"[TelegramApp] image payload: file_id={'yes' if file_id else 'no'} "
            f"url={'yes' if image_url else 'no'} bytes={len(image_bytes or b'')} "
            f"caption=False provider={result.get('provider')}",
            flush=True,
        )

        try:
            if file_id:
                await message.answer_photo(file_id, caption=caption)
                return True
        except Exception as e:
            print(f"[TelegramApp] send file_id failed: {e}", flush=True)

        try:
            if image_bytes:
                bio = BufferedInputFile(image_bytes, filename="pamela.jpg")
                await message.answer_photo(bio, caption=caption)
                return True
        except Exception as e:
            print(f"[TelegramApp] send bytes failed: {e}", flush=True)

        if image_url:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(image_url)
                if r.status_code == 200 and len(r.content) > 1000:
                    bio = BufferedInputFile(r.content, filename="pamela.jpg")
                    await message.answer_photo(bio, caption=caption)
                    return True
            except Exception as e:
                print(f"[TelegramApp] download/send url failed: {e}", flush=True)
                try:
                    await message.answer_photo(image_url, caption=caption)
                    return True
                except Exception as e2:
                    print(f"[TelegramApp] answer_photo url failed: {e2}", flush=True)

        print("[TelegramApp] sem dados de imagem utilizáveis", flush=True)
        return False

    async def feed_webhook_update(self, update):
        raw = update
        if isinstance(update, dict):
            uid = update.get("update_id")
        else:
            uid = getattr(update, "update_id", None)

        if self._mark_update(uid):
            print(f"[TelegramApp] skip update_id duplicado={uid}", flush=True)
            return

        await self.dp.feed_update(
            self.bot,
            Update.model_validate(update) if not isinstance(update, Update) else update,
        )

    async def set_webhook(self):
        if settings.webhook_base_url:
            url = (
                settings.webhook_base_url.rstrip("/")
                + "/telegram/webhook"
            )
            await self.bot.set_webhook(
                url,
                secret_token=settings.webhook_secret,
                drop_pending_updates=True,
                allowed_updates=[
                    "message",
                    "channel_post",
                    "edited_channel_post",
                    "callback_query",
                ],
            )
            print(
                f"[TelegramApp] webhook set url={url} "
                f"allowed_updates=message,channel_post",
                flush=True,
            )
