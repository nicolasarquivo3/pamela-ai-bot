import asyncio
import os

import uvicorn

from app.config import settings
from app.database.session import SessionLocal

from app.images.repository import ImageQuota, ImageRepository
from app.images.router import ImageProviderRouter
from app.images.service import ImageService
from app.images.face_swap import FaceSwapService
from app.images.pexels import PexelsSearchService
from app.images.web_search_images import WebImageSearchService
from app.images.reddit_images import RedditImageSearchService
from app.images.bing_images import BingImageSearchService
from app.images.pixabay_images import PixabaySearchService
from app.images.gelbooru_images import GelbooruSearchService

from app.providers.huggingface_image import HuggingFaceImageProvider

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

from app.telegram.bot import TelegramApp
from app.llm.gemini import GeminiLLM
from app.web import create_web_app


def make_brain_components(session):
    extractor = MemoryExtractor()
    memory_manager = MemoryManager(session, extractor, Deduplicator())
    semantic_manager = SemanticMemoryManager(session)
    emotion_engine = EmotionEngine(session)
    relationship_engine = RelationshipEngine(session)
    context_manager = ContextManager(
        session, memory_manager, emotion_engine, relationship_engine, semantic_manager
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
    huggingface_provider = HuggingFaceImageProvider(
        space_url="https://xurxowsky-flux2-klein-4b-playground.hf.space",
        timeout=settings.image_timeout_seconds,
    )
    providers.append(huggingface_provider)
    print(
        "[IMAGE] Providers AI (fallback): "
        + ", ".join(p.name for p in providers),
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
    print("[IMAGE] DuckDuckGo ativo (micro saia/vestido)", flush=True)

    bing_image_service = BingImageSearchService(timeout=45, max_results=15)
    print("[IMAGE] Bing ativo (micro saia/vestido)", flush=True)

    reddit_image_service = RedditImageSearchService(timeout=40, limit=30)
    print("[IMAGE] Reddit ativo", flush=True)

    pixabay_key = getattr(settings, "pixabay_api_key", None) or os.getenv(
        "PIXABAY_API_KEY"
    )
    pixabay_service = None
    if pixabay_key:
        pixabay_service = PixabaySearchService(api_key=pixabay_key)
        print("[IMAGE] Pixabay ativo", flush=True)
    else:
        print("[IMAGE] PIXABAY_API_KEY ausente — Pixabay off", flush=True)

    gelbooru_service = GelbooruSearchService(timeout=40, limit=30)
    print("[IMAGE] Gelbooru ativo (anime/NSFW)", flush=True)

    pexels_service = None
    if settings.pexels_api_key:
        pexels_service = PexelsSearchService(
            api_key=settings.pexels_api_key,
            timeout=30,
            per_page=15,
            orientation="portrait",
        )
        print("[IMAGE] Pexels ativo (ULTIMO fallback)", flush=True)
    else:
        print("[IMAGE] PEXELS_API_KEY ausente", flush=True)

    image_service = ImageService(
        chars,
        ImageProviderRouter(providers),
        ImageRepository(session),
        ImageQuota(session, settings.image_daily_limit, settings.image_monthly_limit),
        face_swap_service=face_swap_service,
        pexels_service=pexels_service,
        web_image_service=web_image_service,
        bing_image_service=bing_image_service,
        reddit_image_service=reddit_image_service,
        pixabay_service=pixabay_service,
        gelbooru_service=gelbooru_service,
        prefer_real_photos=settings.prefer_real_photos,
    )

    (
        memory_manager,
        semantic_manager,
        emotion_engine,
        relationship_engine,
        context_manager,
    ) = make_brain_components(session)

    llm = GeminiLLM(
        settings.gemini_api_key,
        settings.gemini_model,
        settings.llm_timeout_seconds,
        settings.llm_max_output_tokens,
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
        min_interval_minutes=settings.autonomy_min_interval_minutes,
        max_daily_messages=settings.autonomy_max_daily_messages,
        photo_chance=0.45,
    )

    app = create_web_app(agent, telegram)
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
