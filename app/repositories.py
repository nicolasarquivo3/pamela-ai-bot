from sqlalchemy import select

from app.database.models import Character, User


class CharacterRepository:
    def __init__(self, session):
        self.session = session

    async def get(self, character_id: int):
        result = await self.session.execute(
            select(Character).where(
                Character.id == character_id
            )
        )

        return result.scalar_one_or_none()

    async def get_default(self):
        result = await self.session.execute(
            select(Character)
            .order_by(Character.id.asc())
            .limit(1)
        )

        return result.scalar_one_or_none()


class UserRepository:
    def __init__(self, session):
        self.session = session

    async def get_by_telegram_id(self, telegram_id: int):
        result = await self.session.execute(
            select(User).where(
                User.telegram_id == telegram_id
            )
        )

        return result.scalar_one_or_none()

    async def get_or_create(self, telegram_id: int):
        user = await self.get_by_telegram_id(telegram_id)

        if user:
            return user

        user = User(
            telegram_id=telegram_id,
            active=True,
            character_id=1,
        )

        self.session.add(user)

        await self.session.flush()

        return user
