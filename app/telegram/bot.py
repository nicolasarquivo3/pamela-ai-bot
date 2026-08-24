from aiogram import Bot, Dispatcher
from aiogram.types import Message
from app.config import settings

class TelegramApp:
    def __init__(self, agent):
        self.bot = Bot(settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.agent = agent

        @self.dp.message()
        async def handler(message: Message):
            if not message.from_user:
                return
            try:
                result = await self.agent.receive_message(message.from_user.id, message.text or "")
                if result["type"] == "text":
                    await message.answer(result["text"])
                elif result["type"] == "image":
                    if result.get("url"):
                        await message.answer_photo(result["url"])
                    elif result.get("bytes"):
                        from aiogram.types import BufferedInputFile
                        await message.answer_photo(BufferedInputFile(result["bytes"], filename="generated.png"))
            finally:
                await self.agent.context_manager.session.commit()

    async def feed_webhook_update(self, update):
        from aiogram.types import Update
        await self.dp.feed_update(self.bot, Update.model_validate(update))

    async def set_webhook(self):
        if settings.webhook_base_url:
            url = settings.webhook_base_url.rstrip("/") + "/telegram/webhook"
            await self.bot.set_webhook(url, secret_token=settings.webhook_secret)
