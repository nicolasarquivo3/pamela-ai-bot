import asyncio
import os

import uvicorn

from app.config import settings
from app.database.session import SessionLocal

from app.images.repository import ImageQuota, ImageRepository
from app.images.router import ImageProviderRouter
from app.images.service import ImageService
try:
    from app.images.album_service import AlbumService
except Exception:
    AlbumService = None  # type: ignore
from app.images.face_swap import FaceSwapService
from app.images.pexels import PexelsSearchService
from app.images.web_search_images import WebImageSearchService
from app.images.reddit_images import RedditImageSearchService
from app.images.bing_images import BingImageSearchService
from app.images.pixabay_images import PixabaySearchService
from app.images.gelbooru_images import GelbooruSearchService

from app.providers.huggingface_image import HuggingFaceImageProvider
from app.providers.pollinations_image import PollinationsImageProvider
from app.providers.stable_horde_image import StableHordeImageProvider
from app.providers.perchance_image import PerchanceImageProvider

from app.repositories import CharacterRepository, UserRepository

from app.brain.agent import AgentBrain
from app.brain.memory_extractor import MemoryExtractor
from app.brain.deduplicator import Deduplicator
from app.brain.memory_manager import MemoryManager
from app.brain.semantic_memory import SemanticMemoryManager
from app.brain.context_manager import ContextManager
from app.brain.emotion_engine import EmotionEngine
from app.brain.relationship_engine import RelationshipEngine
from app.brain.autonomy import AutonomyService
from app.brain.autonomy_loop import start_autonomy_loop

from app.telegram.bot import TelegramApp
from app.llm.gemini import GeminiLLM
from app.llm.openrouter import OpenRouterLLM
from app.llm.llm_router import LLMRouter
from app.web import create_web_app


def make_brain_components(session):
    extractor = MemoryExtractor()
    memory_manager = MemoryManager(session, extractor, Deduplicator())
    semantic_manager = SemanticMemoryManager(session)
    emotion_engine = EmotionEngine(session)
    relationship_engine = RelationshipEngine(session)
    context_manager = ContextManager(
        session,
        memory_manager,
        emotion_engine,
        relationship_engine,
        semantic_manager,
    )
    return (
        memory_manager,
        semantic_manager,
        emotion_engine,
        relationship_engine,
        context_manager,
    )


async def main():
    session = SessionLocal()
    chars = CharacterRepository(session)
    users = UserRepository(session)

    providers = []

    # 1) Perchance (PRIORIDADE) — estilo do seu gerador
    # Se a key cair: log perchance:invalid_key → usa Horde/HF/etc até você trocar a key
    # Channel: https://perchance.org/5yf90s8rdo
    import os as _os
    pc_key = (
        getattr(settings, "perchance_user_key", None)
        or _os.getenv("PERCHANCE_USER_KEY")
    )
    pc_channel = (
        getattr(settings, "perchance_channel", None)
        or _os.getenv("PERCHANCE_CHANNEL")
        or "5yf90s8rdo"
    )
    if pc_key:
        # lista de geradores (mesma API, channels diferentes)
        raw_chs = (
            getattr(settings, "perchance_channels", None)
            or _os.getenv("PERCHANCE_CHANNELS")
            or ""
        )
        extra_channels = [c.strip() for c in str(raw_chs).split(",") if c.strip()]
        providers.append(
            PerchanceImageProvider(
                user_key=pc_key.strip(),
                cookies=(
                    getattr(settings, "perchance_cookies", None)
                    or _os.getenv("PERCHANCE_COOKIES")
                ),
                channel=(pc_channel or "5yf90s8rdo").strip(),
                channels=extra_channels or None,
                timeout=int(getattr(settings, "image_timeout_seconds", 180) or 180),
                resolution=getattr(settings, "perchance_resolution", None) or "512x768",
            )
        )
        print(
            f"[IMAGE] Perchance #1 key={pc_key[:8]}... "
            f"primary={pc_channel!r} extras={extra_channels}",
            flush=True,
        )
    else:
        print(
            "[IMAGE] Perchance SEM KEY — defina PERCHANCE_USER_KEY "
            "(cai direto no Stable Horde)",
            flush=True,
        )

    # 2) Stable Horde FREE (fallback se Perchance falhar / key morta)
    # Key gratis: https://stablehorde.net/register
    horde = StableHordeImageProvider(
        api_key=getattr(settings, "stable_horde_api_key", None),
        timeout=int(getattr(settings, "image_timeout_seconds", 180) or 180),
        width=512,
        height=768,
        nsfw=True,
    )
    providers.append(horde)

    # 3) HuggingFace Space Flux (gratis, mas SSE falha muito)
    huggingface_provider = HuggingFaceImageProvider(
        space_url="https://xurxowsky-flux2-klein-4b-playground.hf.space",
        timeout=settings.image_timeout_seconds,
    )
    providers.append(huggingface_provider)

    # 4) Pollinations — so se tiver key com pollen (402 = sem credito)
    pol_key = getattr(settings, "pollinations_api_key", None)
    if pol_key:
        pollinations = PollinationsImageProvider(
            api_key=pol_key,
            model=getattr(settings, "pollinations_image_model", None) or "flux",
            timeout=int(getattr(settings, "image_timeout_seconds", 120) or 120),
            width=768,
            height=1024,
            max_retries=2,
        )
        providers.append(pollinations)
        print("[IMAGE] Pollinations ativo (key presente)", flush=True)
    else:
        print(
            "[IMAGE] Pollinations desligado (sem key / free acabou — use Stable Horde)",
            flush=True,
        )

    print(
        "[IMAGE] Providers AI: " + ", ".join(p.name for p in providers),
        flush=True,
    )

    face_swap_service = None
    if settings.face_swap_enabled:
        reference_path = settings.face_reference_image_path
        if reference_path and not reference_path.startswith("/"):
            reference_path = f"/app/{reference_path}"
        face_swap_service = FaceSwapService(
            reference_path=reference_path,
            required=settings.face_swap_required,
            provider=settings.face_swap_provider,
            hf_space=settings.hf_face_swap_space,
            hf_api_name=settings.hf_face_swap_api_name,
            hf_token=settings.hf_token,
            hf_swap_model=settings.hf_face_swap_model,
            hf_target_index=settings.hf_face_swap_target_index,
            hf_restore_model=settings.hf_face_restore_model,
            hf_restore_strength=settings.hf_face_restore_strength,
            replicate_token=settings.replicate_api_token,
            replicate_version=settings.replicate_face_swap_version,
            timeout=settings.face_swap_timeout_seconds,
        )

    web_image_service = WebImageSearchService(timeout=45, max_results=15)
    print("[IMAGE] DuckDuckGo ativo (fallback)", flush=True)

    bing_image_service = BingImageSearchService(timeout=45, max_results=15)
    print("[IMAGE] Bing ativo (fallback)", flush=True)

    reddit_image_service = RedditImageSearchService(timeout=40, limit=30)
    print("[IMAGE] Reddit ativo (fallback)", flush=True)

    pixabay_key = getattr(settings, "pixabay_api_key", None) or os.getenv(
        "PIXABAY_API_KEY"
    )
    pixabay_service = None
    if pixabay_key:
        pixabay_service = PixabaySearchService(api_key=pixabay_key)
        print("[IMAGE] Pixabay ativo (fallback)", flush=True)
    else:
        print("[IMAGE] PIXABAY_API_KEY ausente", flush=True)

    gelbooru_service = GelbooruSearchService(timeout=40, limit=30)
    print("[IMAGE] Gelbooru ativo (fallback)", flush=True)

    pexels_service = None
    if settings.pexels_api_key:
        pexels_service = PexelsSearchService(
            api_key=settings.pexels_api_key,
            timeout=30,
            per_page=40,
            orientation="portrait",
        )
        print("[IMAGE] Pexels ativo (ultimo fallback)", flush=True)

    # Album Telegram (opcional — so ativa se album_service.py existir)
    album_service = None
    if AlbumService is not None and bool(getattr(settings, "album_enabled", True)):
        _album_ch = getattr(settings, "album_channel_id", None) or "-1004349291324"
        try:
            _album_ch_int = int(str(_album_ch).strip())
        except Exception:
            _album_ch_int = None
        album_service = AlbumService(
            session=session,
            channel_id=_album_ch_int,
            enabled=True,
            use_llm_match=bool(getattr(settings, "album_use_llm_match", True)),
            llm=None,
        )
        print(
            f"[ALBUM] enabled={album_service.enabled} channel={album_service.channel_id}",
            flush=True,
        )
    else:
        print("[ALBUM] desligado (sem modulo ou album_enabled=false)", flush=True)

    image_service = ImageService(

        chars,
        ImageProviderRouter(providers),
        ImageRepository(session),
        ImageQuota(
            session,
            settings.image_daily_limit,
            settings.image_monthly_limit,
        ),
        face_swap_service=face_swap_service,
        pexels_service=pexels_service,
        web_image_service=web_image_service,
        bing_image_service=bing_image_service,
        reddit_image_service=reddit_image_service,
        pixabay_service=pixabay_service,
        gelbooru_service=gelbooru_service,
        album_service=album_service,
        album_first=bool(getattr(settings, "album_first", True)),
        prefer_real_photos=settings.prefer_real_photos,
        prefer_ai_first=True,
    )

    (
        memory_manager,
        semantic_manager,
        emotion_engine,
        relationship_engine,
        context_manager,
    ) = make_brain_components(session)

    # LLM: Gemini -> se bloquear/falhar -> OpenRouter Venice
    gemini = GeminiLLM(
        api_key=getattr(settings, "gemini_api_key", None),
        api_keys=getattr(settings, "gemini_api_keys", None),
        model=settings.gemini_model,
        timeout=int(getattr(settings, "llm_timeout_seconds", 60) or 60),
        max_output_tokens=int(getattr(settings, "llm_max_output_tokens", 1000) or 1000),
    )
    openrouter = OpenRouterLLM(
        api_key=getattr(settings, "openrouter_api_key", None),
        model=getattr(
            settings,
            "openrouter_model",
            "openrouter/free",
        ),
        timeout=int(getattr(settings, "llm_timeout_seconds", 90) or 90),
        max_output_tokens=int(
            getattr(settings, "llm_max_output_tokens", 1000) or 1000
        ),
    )
    llm = LLMRouter(primary=gemini, fallback=openrouter)
    print(
        f"[LLM] Router: primary=Gemini(keys={len(getattr(gemini, 'keys', []) or [])}) "
        f"fallback=OpenRouter "
        f"(key={'sim' if openrouter.api_key else 'NAO'})",
        flush=True,
    )

    agent = AgentBrain(
        image_service,
        users,
        context_manager,
        memory_manager,
        emotion_engine,
        relationship_engine,
        semantic_manager,
        llm,
    )

    # liga LLM no album (match opcional)
    if album_service is not None:
        album_service.llm = llm
    try:
        telegram = TelegramApp(agent, album_service=album_service)
    except TypeError:
        telegram = TelegramApp(agent)

    await telegram.set_webhook()

    def memory_factory(tick_session):
        mm, sm, ee, re, cm = make_brain_components(tick_session)
        return mm, sm, cm

    agent.autonomy_service = AutonomyService(
        SessionLocal,
        telegram.bot,
        llm,
        memory_factory,
        image_service=image_service,
        min_interval_minutes=getattr(settings, "autonomy_min_interval_minutes", 30),
        max_daily_messages=getattr(settings, "autonomy_max_daily_messages", 12),
        photo_chance=0.55,
    )

    app = create_web_app(agent, telegram)

    # Loop interno: mensagens proativas
    autonomy_interval = int(
        getattr(settings, "autonomy_tick_seconds", None)
        or os.getenv("AUTONOMY_TICK_SECONDS", "600")
    )
    start_autonomy_loop(agent, interval_seconds=autonomy_interval)
    print(f"[Autonomy] loop a cada {autonomy_interval}s", flush=True)

    port = int(os.getenv("PORT", "8000"))
    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    )
    try:
        print(f"[WEB] Starting server on port {port}", flush=True)
        await server.serve()
    finally:
        try:
            await telegram.bot.session.close()
        except Exception:
            pass
        try:
            await session.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
