from sqlalchemy import select
from app.database.models import Character, User

class CharacterRepository:
    def __init__(self, session):
        self.session = session
    async def get(self, character_id):
        result = await self.session.execute(select(Character).where(Character.id == character_id))
        return result.scalar_one_or_none()

class UserRepository:
    def __init__(self, session):
        self.session = session
    async def get_or_create(self, telegram_id):
        result = await self.session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if user:
            return user
        user = User(telegram_id=telegram_id, active=True, character_id=1)
        self.session.add(user)
        await self.session.flush()
        return user
    async def active_users(self):
        result = await self.session.execute(select(User).where(User.active.is_(True)))
        return list(result.scalars())
