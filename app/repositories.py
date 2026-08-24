from datetime import date, datetime, timezone

from sqlalchemy import select

from app.database.models import (
    Character,
    User,
    ImageGeneration,
    ImageStatus,
    ImageUsage,
)


class CharacterRepository:

    def __init__(self, session):
        self.session = session

    async def get_by_id(self, character_id):
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

    async def get_or_create_default(self):
        character = await self.get_default()

        if character:
            return character

        character = Character(
            name="Pâmela",
            image_identity={},
            personality_profile={
                "description": (
                    "Mulher adulta, carinhosa, espontânea, "
                    "afetuosa e conversacional."
                )
            },
        )

        self.session.add(character)

        await self.session.flush()

        return character


class UserRepository:

    def __init__(self, session):
        self.session = session

    async def get_by_telegram_id(self, telegram_id):
        result = await self.session.execute(
            select(User).where(
                User.telegram_id == telegram_id
            )
        )

        return result.scalar_one_or_none()

    async def get_or_create(self, telegram_id):
        user = await self.get_by_telegram_id(
            telegram_id
        )

        if user:
            return user

        character = await self._get_default_character()

        user = User(
            telegram_id=telegram_id,
            active=True,
            character_id=character.id if character else 1,
        )

        self.session.add(user)

        await self.session.flush()

        return user

    async def _get_default_character(self):
        result = await self.session.execute(
            select(Character)
            .order_by(Character.id.asc())
            .limit(1)
        )

        return result.scalar_one_or_none()


class ImageRepository:

    def __init__(self, session):
        self.session = session

    async def create(
        self,
        request,
        prompt,
        negative,
    ):
        row = ImageGeneration(
            user_id=request.user_id,
            character_id=request.character_id,
            scene=request.scene,
            prompt=prompt,
            negative_prompt=negative,
            status=ImageStatus.PROCESSING,
            metadata_json={
                "width": request.width,
                "height": request.height,
                "style": request.style,
            },
        )

        self.session.add(row)

        await self.session.flush()

        return row

    async def complete(
        self,
        row,
        result,
    ):
        row.status = ImageStatus.COMPLETED

        row.provider = result.provider

        row.job_id = result.job_id

        row.image_url = result.image_url

        row.completed_at = datetime.now(
            timezone.utc
        )

        row.metadata_json = {
            **(row.metadata_json or {}),
            "face_swapped": bool(
                result.face_swapped
            ),
        }

        await self.session.flush()

    async def fail(
        self,
        row,
        error,
    ):
        row.status = ImageStatus.FAILED

        row.error = str(error)

        await self.session.flush()


class ImageQuota:

    def __init__(
        self,
        session,
        daily_limit=5,
        monthly_limit=100,
    ):
        self.session = session
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit

    async def allowed(
        self,
        user_id,
    ):
        today = date.today()

        month_start = today.replace(
            day=1
        )

        result = await self.session.execute(
            select(ImageUsage).where(
                ImageUsage.user_id == user_id,
                ImageUsage.period_date >= month_start,
            )
        )

        rows = list(
            result.scalars()
        )

        daily = sum(
            row.count
            for row in rows
            if row.period_date == today
        )

        monthly = sum(
            row.count
            for row in rows
        )

        return (
            daily < self.daily_limit
            and monthly < self.monthly_limit
        )

    async def consume(
        self,
        user_id,
        provider,
    ):
        today = date.today()

        result = await self.session.execute(
            select(ImageUsage).where(
                ImageUsage.user_id == user_id,
                ImageUsage.provider == provider,
                ImageUsage.period_date == today,
            )
        )

        row = result.scalar_one_or_none()

        if row:
            row.count += 1
        else:
            self.session.add(
                ImageUsage(
                    user_id=user_id,
                    provider=provider,
                    period_date=today,
                    count=1,
                )
            )

        await self.session.flush()
