from app.images.models import ImageResult, ImageRequest
from app.images.prompt_builder import PromptBuilder


class ImageService:
    """
    Ordem (Pexels por ultimo):
      1. DuckDuckGo  2. Bing  3. Reddit  4. Pixabay  5. Gelbooru  6. Pexels  7. AI
    Padrao: micro saia / micro vestido.
    """

    def __init__(
        self,
        character_repository,
        router,
        image_repository,
        quota,
        face_swap_service=None,
        pexels_service=None,
        web_image_service=None,
        bing_image_service=None,
        reddit_image_service=None,
        pixabay_service=None,
        gelbooru_service=None,
        prefer_real_photos: bool = True,
    ):
        self.character_repository = character_repository
        self.router = router
        self.image_repository = image_repository
        self.quota = quota
        self.face_swap_service = face_swap_service
        self.pexels_service = pexels_service
        self.web_image_service = web_image_service
        self.bing_image_service = bing_image_service
        self.reddit_image_service = reddit_image_service
        self.pixabay_service = pixabay_service
        self.gelbooru_service = gelbooru_service
        self.prefer_real_photos = bool(prefer_real_photos)
        self.prompt_builder = PromptBuilder()

    async def generate(self, request: ImageRequest) -> ImageResult:
        if not await self.quota.allowed(request.user_id):
            return ImageResult(False, error="daily_or_monthly_image_limit")

        character = await self.character_repository.get(request.character_id)
        if not character:
            return ImageResult(False, error="character_not_found")

        scene = (request.scene or "").strip() or (
            "sexy woman micro mini dress micro mini skirt fashion selfie"
        )
        print(f"[IMAGE] generate scene={scene[:100]!r}", flush=True)

        if self.prefer_real_photos:
            chain = [
                ("duckduckgo", self.web_image_service),
                ("bing", self.bing_image_service),
                ("reddit", self.reddit_image_service),
                ("pixabay", self.pixabay_service),
                ("gelbooru", self.gelbooru_service),
                ("pexels", self.pexels_service),
            ]
            for name, svc in chain:
                if not svc:
                    continue
                result = await self._try_source(name, svc, request, scene)
                if result and result.success:
                    return result

        return await self._try_ai_generation(request, character, scene)

    async def _try_source(self, provider_name, service, request, scene):
        try:
            if hasattr(service, "available"):
                if not await service.available():
                    return None

            photo = await service.search(scene)
            if not photo:
                print(f"[IMAGE] {provider_name}: nenhuma foto", flush=True)
                return None

            print(
                f"[IMAGE] {provider_name} ok: query={photo.get('query')} "
                f"bytes={len(photo.get('bytes') or b'')}",
                flush=True,
            )

            record = await self.image_repository.create(
                request,
                f"{provider_name.upper()}:{photo.get('query')}",
                "real_photo",
            )

            result = ImageResult(
                success=True,
                provider=provider_name,
                image_url=photo.get("url"),
                image_bytes=photo.get("bytes"),
            )

            if self.face_swap_service:
                result = await self.face_swap_service.apply(result)
                if not result.success:
                    await self.image_repository.fail(
                        record, result.error or "face_swap_failed"
                    )
                    print(
                        f"[IMAGE] Face swap falhou no {provider_name}: {result.error}",
                        flush=True,
                    )
                    return None

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or provider_name)
            return result
        except Exception as e:
            print(f"[IMAGE] {provider_name} error: {e}", flush=True)
            return None

    async def _try_ai_generation(self, request, character, scene):
        prompt = self.prompt_builder.build(character, request)
        negative_prompt = self.prompt_builder.NEGATIVE_PROMPT

        record = await self.image_repository.create(request, prompt, negative_prompt)

        try:
            result = await self.router.generate(request, prompt)
            if not result.success:
                await self.image_repository.fail(
                    record, result.error or "generation_failed"
                )
                return result

            if self.face_swap_service:
                result = await self.face_swap_service.apply(result)
                if not result.success:
                    await self.image_repository.fail(
                        record, result.error or "face_swap_failed"
                    )
                    return result

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or "unknown")
            return result
        except Exception as exc:
            await self.image_repository.fail(record, str(exc))
            return ImageResult(False, error=str(exc))
