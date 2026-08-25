from app.images.models import ImageResult, ImageRequest
from app.images.prompt_builder import PromptBuilder


class ImageService:
    """
    Fluxo:
      1. DuckDuckGo
      2. Reddit
      3. Pexels
      4. AI
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
        reddit_image_service=None,
        prefer_real_photos: bool = True,
    ):
        self.character_repository = character_repository
        self.router = router
        self.image_repository = image_repository
        self.quota = quota
        self.face_swap_service = face_swap_service
        self.pexels_service = pexels_service
        self.web_image_service = web_image_service
        self.reddit_image_service = reddit_image_service
        self.prefer_real_photos = bool(prefer_real_photos)
        self.prompt_builder = PromptBuilder()

    async def generate(self, request: ImageRequest) -> ImageResult:
        if not await self.quota.allowed(request.user_id):
            return ImageResult(False, error="daily_or_monthly_image_limit")

        character = await self.character_repository.get(request.character_id)
        if not character:
            return ImageResult(False, error="character_not_found")

        scene = (request.scene or "").strip() or "sexy woman fashion mirror selfie"
        print(f"[IMAGE] generate scene={scene[:100]!r}", flush=True)

        if self.prefer_real_photos:
            if self.web_image_service:
                web = await self._try_source(
                    "duckduckgo", self.web_image_service, request, scene
                )
                if web and web.success:
                    return web

            if self.reddit_image_service:
                reddit = await self._try_source(
                    "reddit", self.reddit_image_service, request, scene
                )
                if reddit and reddit.success:
                    return reddit

            if self.pexels_service:
                real = await self._try_real_photo(request, scene)
                if real and real.success:
                    return real

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

    async def _try_real_photo(self, request, scene):
        try:
            photo = await self.pexels_service.search(scene)
            if not photo or not photo.get("url"):
                print("[IMAGE] Pexels: nenhuma foto encontrada", flush=True)
                return None

            print(
                f"[IMAGE] Pexels ok: id={photo.get('photo_id')} "
                f"by {photo.get('photographer')} query={photo.get('query')}",
                flush=True,
            )

            record = await self.image_repository.create(
                request, f"PEXELS:{photo.get('query')}", "real_photo"
            )

            result = ImageResult(
                success=True,
                provider="pexels",
                image_url=photo["url"],
                image_bytes=photo.get("bytes"),
            )

            if self.face_swap_service:
                result = await self.face_swap_service.apply(result)
                if not result.success:
                    await self.image_repository.fail(
                        record, result.error or "face_swap_failed"
                    )
                    print(
                        f"[IMAGE] Face swap falhou no Pexels: {result.error}",
                        flush=True,
                    )
                    return None

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or "pexels")
            return result
        except Exception as e:
            print(f"[IMAGE] real_photo error: {e}", flush=True)
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
