"""
Telegram bot.
- 1 mensagem do usuario = 1 processamento (lock por user + dedup update/message)
- Foto SEM legenda; texto em mensagem separada
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

from aiogram import Bot, Dispatcher
from aiogram.types import Message, BufferedInputFile, Update
import httpx

from app.config import settings


class TelegramApp:

    def __init__(self, agent):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.agent = agent

        # evita 2 respostas quando Telegram reenvia webhook (foto demora)
        self._user_locks: dict[int, asyncio.Lock] = {}
        self._seen_updates: deque[int] = deque(maxlen=500)
        self._seen_update_set: set[int] = set()
        self._seen_messages: deque[str] = deque(maxlen=500)
        self._seen_message_set: set[str] = set()

        @self.dp.message()
        async def handler(message: Message):
            if not message.from_user:
                return

            user_id = message.from_user.id
            msg_key = f"{user_id}:{message.message_id}"
            update_id = getattr(message, "message_id", None)

            # dedup por message_id (retry do Telegram)
            if msg_key in self._seen_message_set:
                print(
                    f"[TelegramApp] skip duplicata message_id={message.message_id}",
                    flush=True,
                )
                return
            self._mark_message(msg_key)

            lock = self._user_locks.setdefault(user_id, asyncio.Lock())
            if lock.locked():
                print(
                    f"[TelegramApp] user={user_id} ainda processando â€” enfileira/espera",
                    flush=True,
                )

            async with lock:
                await self._handle_one(message)

    def _mark_message(self, key: str) -> None:
        if key in self._seen_message_set:
            return
        if len(self._seen_messages) >= 500:
            old = self._seen_messages.popleft()
            self._seen_message_set.discard(old)
        self._seen_messages.append(key)
        self._seen_message_set.add(key)

    def _mark_update(self, update_id: int | None) -> bool:
        """True se ja visto (duplicata)."""
        if update_id is None:
            return False
        if update_id in self._seen_update_set:
            return True
        if len(self._seen_updates) >= 500:
            old = self._seen_updates.popleft()
            self._seen_update_set.discard(old)
        self._seen_updates.append(update_id)
        self._seen_update_set.add(update_id)
        return False

    async def _handle_one(self, message: Message) -> None:
        session = self.agent.context_manager.session
        try:
            text = message.text or ""
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
                print(
                    f"[TelegramApp] result inesperado: {type(result)} {result!r}",
                    flush=True,
                )
                await message.answer(
                    "Amor, tive um probleminha agora. Tenta de novo? â¤ï¸"
                )
                return

            if result.get("type") == "text":
                await message.answer(
                    result.get(
                        "text",
                        "Desculpa, tive um probleminha agora. â¤ï¸",
                    )
                )
                return

            if result.get("type") == "image":
                # Texto (se houver) separado; foto SEM legenda
                reply_text = (result.get("text") or "").strip()
                if reply_text:
                    if "Ã¢" in reply_text or "Ã¯Â¸" in reply_text:
                        reply_text = "Olha eu aqui â¤ï¸"
                    await message.answer(reply_text)

                photo_result = dict(result)
                photo_result["caption"] = None
                photo_result["text"] = None

                ok = await self._send_image_result(message, photo_result)
                if not ok and not reply_text:
                    await message.answer(
                        "Amor, a foto nao saiu agora. Tenta de novo? â¤ï¸"
                    )
                return

            await message.answer(
                "Amor, tive um probleminha para processar "
                "sua mensagem agora. â¤ï¸"
            )

        except Exception as exc:
            try:
                await session.rollback()
            except Exception:
                pass

            print(
                f"[TelegramApp] Erro: {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                await message.answer(
                    "Amor, deu um probleminha aqui agora. "
                    "Tenta mandar de novo em alguns segundos? â¤ï¸"
                )
            except Exception as send_exc:
                print(
                    f"[TelegramApp] Erro ao avisar usuÃ¡rio: "
                    f"{type(send_exc).__name__}: {send_exc}",
                    flush=True,
                )

    async def _send_image_result(self, message: Message, result: dict) -> bool:
        """Prefere bytes (face swap). Foto SEMPRE sem caption."""
        image_url = result.get("url")
        image_bytes = result.get("bytes")
        caption = None

        print(
            f"[TelegramApp] image payload: "
            f"url={'yes' if image_url else 'no'} "
            f"bytes={len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else type(image_bytes)} "
            f"caption={bool(caption)}",
            flush=True,
        )

        if isinstance(image_bytes, (bytes, bytearray)) and len(image_bytes) > 100:
            try:
                photo = BufferedInputFile(
                    bytes(image_bytes),
                    filename="pamela.jpg",
                )
                await message.answer_photo(photo, caption=caption)
                return True
            except Exception as e:
                print(f"[TelegramApp] send bytes failed: {e}", flush=True)

        if image_url and isinstance(image_url, str) and image_url.startswith("http"):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(image_url)
                    r.raise_for_status()
                    data = r.content
                if len(data) > 100:
                    photo = BufferedInputFile(data, filename="pamela.jpg")
                    await message.answer_photo(photo, caption=caption)
                    return True
            except Exception as e:
                print(f"[TelegramApp] download/send url failed: {e}", flush=True)
                try:
                    await message.answer_photo(image_url, caption=caption)
                    return True
                except Exception as e2:
                    print(f"[TelegramApp] answer_photo url failed: {e2}", flush=True)

        print("[TelegramApp] sem dados de imagem utilizÃ¡veis", flush=True)
        return False

    async def feed_webhook_update(self, update):
        # dedup por update_id (Telegram retry)
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
            # drop_pending evita flood de updates antigos no redeploy
            await self.bot.set_webhook(
                url,
                secret_token=settings.webhook_secret,
                drop_pending_updates=True,
            )
