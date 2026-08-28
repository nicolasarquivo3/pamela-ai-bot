"""
Busca web por outfit/contexto da conversa + face swap no final.

Ordem:
  1) AI (Flux) + seed + face swap
  2) Fallback: DDG → Bing → Reddit → Pixabay → Gelbooru → Pexels + face swap
"""
from __future__ import annotations

import random
import time

from app.images.models import ImageResult, ImageRequest
from app.images.prompt_builder import PromptBuilder

try:
    from app.images.recent_guard import RECENT
except Exception:
    RECENT = None  # type: ignore


class ImageService:
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
        prefer_ai_first: bool = True,
        album_service=None,
        drive_album_service=None,
        album_first: bool = True,
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
        self.prefer_ai_first = bool(prefer_ai_first)
        self.album_service = album_service
        self.drive_album_service = drive_album_service
        self.album_first = bool(album_first)
        self.prompt_builder = PromptBuilder()

    def _web_search_query(self, scene: str) -> str:
        """Query internet a partir da roupa/cena da conversa."""
        try:
            from app.images.web_search_images import build_web_query
            return build_web_query(scene)
        except Exception:
            pass
        s = (scene or "").lower()
        parts = ["sexy young woman"]
        if "black" in s or "preto" in s:
            parts.append("black")
        if "mini" in s or "curt" in s:
            parts.append("mini dress")
        if "dress" in s or "vestido" in s:
            parts.append("dress")
        if "heel" in s or "salto" in s:
            parts.append("high heels")
        if "arrum" in s or "ready" in s or "mirror" in s or "espelho" in s:
            parts.append("getting ready mirror home")
        if "club" in s or "balada" in s or "party" in s:
            parts.append("going out night")
        if "outfit:" in s:
            try:
                o = s.split("outfit:")[1].split("|")[0].strip()
                parts.append(o[:80])
            except Exception:
                pass
        q = " ".join(parts)
        print(f"[IMAGE] web query from scene: {q[:100]!r}", flush=True)
        return q[:120]

    def _fresh_seed(self) -> int:

        return (int(time.time() * 1000) + random.randint(0, 1_000_000)) % 2_147_483_647

    def _face_swap_required(self) -> bool:
        fs = self.face_swap_service
        if fs is None:
            return False
        return bool(getattr(fs, "required", True))


    async def _download_telegram_file(self, file_id: str):
        """Baixa bytes de um file_id do Telegram (album)."""
        try:
            from app.config import settings
            import httpx

            token = (settings.telegram_bot_token or "").strip()
            if not token or not file_id:
                return None
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(
                    f"https://api.telegram.org/bot{token}/getFile",
                    params={"file_id": file_id},
                )
                if r.status_code != 200:
                    print(f"[ALBUM] getFile HTTP {r.status_code}", flush=True)
                    return None
                data = r.json()
                if not data.get("ok"):
                    print(f"[ALBUM] getFile not ok: {data}", flush=True)
                    return None
                path = (data.get("result") or {}).get("file_path")
                if not path:
                    return None
                r2 = await client.get(
                    f"https://api.telegram.org/file/bot{token}/{path}"
                )
                if r2.status_code != 200 or len(r2.content) < 500:
                    print(
                        f"[ALBUM] download HTTP {r2.status_code} "
                        f"len={len(r2.content)}",
                        flush=True,
                    )
                    return None
                print(
                    f"[ALBUM] downloaded bytes={len(r2.content)} path={path}",
                    flush=True,
                )
                return r2.content
        except Exception as e:
            print(f"[ALBUM] download error: {e}", flush=True)
            return None


    async def generate(self, request: ImageRequest) -> ImageResult:
        if not await self.quota.allowed(request.user_id):
            return ImageResult(False, error="daily_or_monthly_image_limit")

        character = await self.character_repository.get(request.character_id)
        if not character:
            return ImageResult(False, error="character_not_found")

        scene = (request.scene or "").strip() or (
            "OUTFIT: micro mini dress high heels | PEDIDO: selfie"
        )
        request.seed = self._fresh_seed()
        request.randomize_seed = True

        print(f"[IMAGE] generate seed={request.seed} scene={scene[:120]!r}", flush=True)

        # 0) ALBUM do canal Telegram (fotos reais suas)
        # Baixa file_id -> face swap (rosto da Pamela) -> envia bytes

        # 0a) Google Drive album (nuvem)
        if self.drive_album_service is not None:
            try:
                if await self.drive_album_service.available():
                    picked = await self.drive_album_service.pick_best(scene)
                    if picked and picked.get("image_bytes"):
                        print(
                            f"[IMAGE] DRIVE hit name={picked.get('name')!r} "
                            f"bytes={len(picked['image_bytes'])}",
                            flush=True,
                        )
                        result = ImageResult(
                            success=True,
                            provider="album:drive",
                            image_bytes=picked["image_bytes"],
                        )
                        if self.face_swap_service:
                            print("[IMAGE] DRIVE -> face swap...", flush=True)
                            result = await self.face_swap_service.apply(result)
                            if result.success:
                                result.provider = "album:drive+faceswap"
                                print(
                                    f"[IMAGE] DRIVE+swap ok bytes="
                                    f"{len(result.image_bytes or b'')}",
                                    flush=True,
                                )
                                return result
                            print(
                                f"[IMAGE] DRIVE face swap falhou: {result.error}",
                                flush=True,
                            )
                            if not self._face_swap_required():
                                return ImageResult(
                                    success=True,
                                    provider="album:drive",
                                    image_bytes=picked["image_bytes"],
                                )
                        else:
                            return result
            except Exception as e:
                print(f"[IMAGE] drive album error: {e}", flush=True)

        if self.album_first and self.album_service:
            try:
                if await self.album_service.available():
                    picked = await self.album_service.pick_best(scene)
                    if picked and picked.get("file_id"):
                        fid = picked["file_id"]
                        print(
                            f"[IMAGE] ALBUM hit file_id={str(fid)[:24]}...",
                            flush=True,
                        )
                        raw = await self._download_telegram_file(fid)
                        if not raw:
                            print(
                                "[IMAGE] ALBUM download falhou — "
                                "reenvia file_id sem swap",
                                flush=True,
                            )
                            return ImageResult(
                                success=True,
                                provider="album:telegram",
                                telegram_file_id=fid,
                            )

                        result = ImageResult(
                            success=True,
                            provider="album:telegram",
                            image_bytes=raw,
                            telegram_file_id=None,
                        )

                        if self.face_swap_service:
                            print("[IMAGE] ALBUM -> face swap...", flush=True)
                            result = await self.face_swap_service.apply(result)
                            if not result.success:
                                print(
                                    f"[IMAGE] ALBUM face swap falhou: "
                                    f"{result.error} — tenta proximo provider",
                                    flush=True,
                                )
                                # nao devolve foto sem rosto se required
                                if self._face_swap_required():
                                    pass  # cai nos providers abaixo
                                else:
                                    # devolve original do album
                                    return ImageResult(
                                        success=True,
                                        provider="album:telegram",
                                        telegram_file_id=fid,
                                    )
                            else:
                                result.provider = (
                                    "album:telegram+faceswap"
                                )
                                print(
                                    f"[IMAGE] ALBUM+swap ok "
                                    f"bytes={len(result.image_bytes or b'')}",
                                    flush=True,
                                )
                                return result
                        else:
                            # sem face swap configurado: manda original
                            return ImageResult(
                                success=True,
                                provider="album:telegram",
                                telegram_file_id=fid,
                            )
            except Exception as e:
                print(f"[IMAGE] album error: {e}", flush=True)

        if self.prefer_ai_first:
            ai = await self._try_ai_generation(request, character, scene)
            if ai and ai.success:
                print(
                    f"[IMAGE] AI ok provider={ai.provider} "
                    f"face_swapped={getattr(ai, 'face_swapped', False)} "
                    f"bytes={len(ai.image_bytes or b'')}",
                    flush=True,
                )
                return ai
            print(
                f"[IMAGE] AI falhou ({getattr(ai, 'error', None)}) -> fallback fotos reais",
                flush=True,
            )

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
                    print(f"[IMAGE] fallback ok provider={name}", flush=True)
                    return result

        if not self.prefer_ai_first:
            return await self._try_ai_generation(request, character, scene)

        return ImageResult(False, error="all_image_providers_failed")

    async def _try_ai_generation(self, request, character, scene) -> ImageResult:
        if not (request.scene or "").strip():
            request.scene = scene

        prompt = self.prompt_builder.build(character, request)
        negative_prompt = getattr(
            self.prompt_builder,
            "NEGATIVE_PROMPT",
            "deformed, bad anatomy, blurry, low quality, child, minor",
        )
        print(f"[IMAGE] AI prompt[:220]={prompt[:220]!r}", flush=True)

        record = await self.image_repository.create(request, prompt, negative_prompt)

        try:
            result = await self.router.generate(request, prompt)
            if not result or not result.success:
                err = getattr(result, "error", None) or "generation_failed"
                await self.image_repository.fail(record, err)
                return result or ImageResult(False, error=err)

            if self.face_swap_service:
                swapped = await self.face_swap_service.apply(result)
                if swapped and swapped.success:
                    result = swapped
                else:
                    err = getattr(swapped, "error", None) or "face_swap_failed"
                    print(f"[IMAGE] Face swap falhou no AI: {err}", flush=True)
                    if self._face_swap_required():
                        await self.image_repository.fail(record, err)
                        return ImageResult(False, error=err, provider="ai+faceswap")
                    print("[IMAGE] face_swap nao obrigatorio — usando AI sem swap", flush=True)

            # anti-repeat final (pos face swap)
            if RECENT is not None:
                if RECENT.seen(content=result.image_bytes):
                    print("[IMAGE] AI result DUPLICATA (hash) — descarta", flush=True)
                    await self.image_repository.fail(record, "duplicate_image")
                    return ImageResult(False, error="duplicate_image", provider=result.provider)
                RECENT.remember(
                    url=result.image_url,
                    content=result.image_bytes,
                )

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or "ai")
            return result
        except Exception as exc:
            await self.image_repository.fail(record, str(exc))
            print(f"[IMAGE] AI exception: {exc}", flush=True)
            return ImageResult(False, error=str(exc))

    async def _try_source(self, provider_name, service, request, scene):
        try:
            if hasattr(service, "available"):
                ok = service.available
                if callable(ok):
                    ok = ok()
                    if hasattr(ok, "__await__"):
                        ok = await ok
                if not ok:
                    return None

            photo = await service.search(scene)
            if not photo:
                print(f"[IMAGE] {provider_name}: nenhuma foto", flush=True)
                return None

            if RECENT is not None and RECENT.seen(
                url=photo.get("url"),
                photo_id=photo.get("photo_id"),
                content=photo.get("bytes"),
            ):
                photo = await service.search(scene + " alternative pose angle")
                if not photo or RECENT.seen(
                    url=photo.get("url"),
                    photo_id=photo.get("photo_id"),
                    content=photo.get("bytes"),
                ):
                    return None

            print(f"[IMAGE] {provider_name} ok: query={photo.get('query')}", flush=True)

            # anti-repeat: marca original antes do face swap
            if RECENT is not None:
                RECENT.remember(
                    url=photo.get("url"),
                    photo_id=photo.get("photo_id"),
                    content=photo.get("bytes"),
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
                swapped = await self.face_swap_service.apply(result)
                if swapped and swapped.success:
                    result = swapped
                else:
                    err = getattr(swapped, "error", None) or "face_swap_failed"
                    print(f"[IMAGE] Face swap falhou no {provider_name}: {err}", flush=True)
                    if self._face_swap_required():
                        await self.image_repository.fail(record, err)
                        return None

            # rejeita se o RESULTADO final (pos face swap) ja foi usado
            if RECENT is not None and RECENT.seen(content=result.image_bytes):
                print(f"[IMAGE] {provider_name} resultado final DUPLICATA — tenta outra", flush=True)
                await self.image_repository.fail(record, "duplicate_final")
                return None

            await self.image_repository.complete(record, result)
            await self.quota.consume(request.user_id, result.provider or provider_name)

            if RECENT is not None:
                RECENT.remember(
                    url=result.image_url or photo.get("url"),
                    photo_id=photo.get("photo_id"),
                    content=result.image_bytes or photo.get("bytes"),
                )
                RECENT.remember(content=result.image_bytes)
            return result
        except Exception as e:
            print(f"[IMAGE] {provider_name} error: {e}", flush=True)
            return None
