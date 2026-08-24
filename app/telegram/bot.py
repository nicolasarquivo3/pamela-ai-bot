from aiogram import Bot, Dispatcher
from aiogram.types import Message, BufferedInputFile

from app.config import settings


class TelegramApp:

    def __init__(self, agent):

        self.bot = Bot(
            token=settings.telegram_bot_token
        )

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

                # -------------------------------------------------
                # IMPORTANTE:
                # Salva tudo antes de enviar a resposta.
                # -------------------------------------------------

                await session.commit()

                # -------------------------------------------------
                # RESPOSTA DE TEXTO
                # -------------------------------------------------

                if result.get("type") == "text":

                    await message.answer(
                        result.get(
                            "text",
                            "Desculpa, tive um probleminha agora. ❤️",
                        )
                    )

                    return

                # -------------------------------------------------
                # RESPOSTA DE IMAGEM
                # -------------------------------------------------

                if result.get("type") == "image":

                    image_url = result.get("url")
                    image_bytes = result.get("bytes")

                    if image_url:

                        await message.answer_photo(
                            image_url
                        )

                        return

                    if image_bytes:

                        photo = BufferedInputFile(
                            image_bytes,
                            filename="pamera_generated.png",
                        )

                        await message.answer_photo(
                            photo
                        )

                        return

                    # Se chegou aqui, o Agent disse que gerou
                    # uma imagem, mas não forneceu os dados.
                    await message.answer(
                        "Amor, a foto foi gerada, mas tive "
                        "um probleminha para te enviar agora. ❤️"
                    )

                    return

                # -------------------------------------------------
                # TIPO DESCONHECIDO
                # -------------------------------------------------

                await message.answer(
                    "Amor, tive um probleminha para processar "
                    "sua mensagem agora. ❤️"
                )

            except Exception as exc:

                # -------------------------------------------------
                # MUITO IMPORTANTE:
                # Nunca tente commit depois de uma exceção.
                # Primeiro fazemos rollback.
                # -------------------------------------------------

                try:
                    await session.rollback()
                except Exception:
                    pass

                print(
                    f"[TelegramApp] Erro ao processar mensagem: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Tentamos avisar o usuário sem derrubar o webhook.
                try:

                    await message.answer(
                        "Amor, deu um probleminha aqui agora. "
                        "Tenta mandar de novo em alguns segundos? ❤️"
                    )

                except Exception as send_exc:

                    print(
                        "[TelegramApp] "
                        f"Erro ao enviar mensagem de erro: "
                        f"{type(send_exc).__name__}: {send_exc}"
                    )

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
