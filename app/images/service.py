from app.images.models import ImageResult
from app.images.prompt_builder import PromptBuilder


class ImageService:
    def __init__(self, character_repository, router, image_repository, quota, face_swap_service=None):
        self.character_repository = character_repository
        self.router = router
        self.image_repository = image_repository
        self.quota = quota
        self.face_swap_service = face_swap_service
        self.prompt_builder = PromptBuilder()

    async def generate(self, request):
        if not await self.quota.allowed(request.user_id):
            return ImageResult(False, error="daily_or_monthly_image_limit")
        character = await self.character_repository.get(request.character_id)
        if not character:
            return ImageResult(False, error="character_not_found")
        prompt = self.prompt_builder.build(character, request)
        record = await self.image_repository.create(
            request, prompt, self.prompt_builder.NEGATIVE_PROMPT
        )
        try:
            result = await self.router.generate(request, prompt)
            if not result.success:
                await self.image_repository.fail(record, result.error or "generation_failed")
                return result

            # The generation provider creates the scene; the face-swap layer is
            # a separate post-processing step that enforces the configured face
            # reference. If face swapping is required, the raw generated image
            # is never returned to the user after a failed swap.
            if self.face_swap_service:
                result = await self.face_swap_service.apply(result)
                if not result.success:
                    await self.image_repository.fail(record, result.error or "face_swap_failed")
                    return result

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or "unknown")
            return result
        except Exception as exc:
            await self.image_repository.fail(record, str(exc))
            return ImageResult(False, error=str(exc))
