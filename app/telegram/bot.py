from aiogram import Bot, Dispatcher
from aiogram.types import Message, BufferedInputFile
import httpx

from app.config import settings


class TelegramApp:

    def __init__(self, agent):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.agent = agent

        @self.dp.message()
        async def handler(message: Message):
            if not message.from_user:
                return

            session = self.agent.context_manager.session

            try:
                text = message.text or ""

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
                    # SEMPRE manda o texto da personagem primeiro
                    # (a legenda da foto some fÃ¡cil no celular)
                    reply_text = (
                        (result.get("text") or "").strip()
                        or (result.get("caption") or "").strip()
                    )
                    if reply_text:
                        # Telegram caption max 1024; texto completo vai na mensagem
                        await message.answer(reply_text)

                    # Foto com legenda curta (nÃ£o repetir o monÃ³logo)
                    photo_result = dict(result)
                    short = (result.get("photo_caption") or "â¤ï¸").strip()
                    if len(short) > 200:
                        short = short[:197] + "..."
                    photo_result["caption"] = short
                    # Evita reenviar o texto longo como caption se falhar
                    photo_result["text"] = short

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
        """Prefere bytes (face swap). Se sÃ³ URL, baixa e envia como arquivo."""
        image_url = result.get("url")
        image_bytes = result.get("bytes")
        caption = (result.get("caption") or "").strip() or None

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
        from aiogram.types import Update

        await self.dp.feed_update(
            self.bot,
            Update.model_validate(update),
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
            )
