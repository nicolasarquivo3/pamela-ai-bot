from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.database.models import (
    ImageGeneration,
    ImageStatus,
    ImageUsage,
)


class ImageRepository:

    def __init__(self, session):
        self.session = session

    async def create(
        self,
        request,
        prompt,
        negative,
    ):
        try:
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

        except SQLAlchemyError:
            await self.session.rollback()
            raise

    async def complete(
        self,
        row,
        result,
    ):
        try:
            row.status = ImageStatus.COMPLETED
            row.provider = getattr(
                result,
                "provider",
                None,
            )
            row.job_id = getattr(
                result,
                "job_id",
                None,
            )
            row.image_url = getattr(
                result,
                "image_url",
                None,
            )

            row.completed_at = datetime.now(
                timezone.utc
            )

            row.metadata_json = {
                **(row.metadata_json or {}),
                "face_swapped": bool(
                    getattr(
                        result,
                        "face_swapped",
                        False,
                    )
                ),
            }

            await self.session.flush()

        except SQLAlchemyError:
            await self.session.rollback()
            raise

    async def fail(
        self,
        row,
        error,
    ):
        try:
            row.status = ImageStatus.FAILED
            row.error = str(error)

            await self.session.flush()

        except SQLAlchemyError:
            await self.session.rollback()
            raise


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

        rows = list(result.scalars())

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
        try:
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

        except SQLAlchemyError:
            await self.session.rollback()
            raise
