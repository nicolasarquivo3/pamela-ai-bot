from app.providers.huggingface_image import (
    HuggingFaceImageProvider,
)
import asyncio
import os
import uvicorn
from app.config import settings
from app.database.session import SessionLocal
from app.images.repository import ImageQuota, ImageRepository
from app.images.router import ImageProviderRouter
from app.images.service import ImageService
from app.images.face_swap import FaceSwapService
from app.providers.cloudflare_image import CloudflareImageProvider
from app.providers.pollinations_image import PollinationsImageProvider
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
        session,
        memory_manager,
        emotion_engine,
        relationship_engine,
        semantic_manager,
    )
    return memory_manager, semantic_manager, emotion_engine, relationship_engine, context_manager

async def main():
    # This session is used by the webhook path. Autonomous ticks create their
    # own short-lived sessions through SessionLocal.
    session = SessionLocal()
    chars = CharacterRepository(session)
    users = UserRepository(session)

    providers = []

providers.append(
    HuggingFaceImageProvider(
        "https://xurxowsky-flux2-klein-4b-playground.hf.space",
        settings.image_timeout_seconds,
    )
)
    if settings.pollinations_api_key:
        providers.append(
            PollinationsImageProvider(
                settings.pollinations_api_key,
                settings.pollinations_image_model,
                settings.image_timeout_seconds,
            )
        )

    face_swap_service = None
    if settings.face_swap_enabled:
        reference_path = settings.face_reference_image_path
        if not reference_path.startswith("/"):
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
    )

    memory_manager, semantic_manager, emotion_engine, relationship_engine, context_manager = (
        make_brain_components(session)
    )

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
        settings.autonomy_min_interval_minutes,
        settings.autonomy_max_daily_messages,
    )

    app = create_web_app(agent, telegram)
    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
    )
    try:
        await server.serve()
    finally:
        await telegram.bot.session.close()
        await session.close()

if __name__ == "__main__":
    asyncio.run(main())
