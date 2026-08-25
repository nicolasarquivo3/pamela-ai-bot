import asyncio
import os

import uvicorn

from app.config import settings
from app.database.session import SessionLocal

from app.images.repository import ImageQuota, ImageRepository
from app.images.router import ImageProviderRouter
from app.images.service import ImageService
from app.images.face_swap import FaceSwapService

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

    memory_manager = MemoryManager(
        session,
        extractor,
        Deduplicator(),
    )

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

    # =========================================================
    # BANCO DE DADOS
    # =========================================================

    session = SessionLocal()

    chars = CharacterRepository(session)
    users = UserRepository(session)

    # =========================================================
    # PROVIDER DE IMAGEM
    # =========================================================
    #
    # Neste momento usamos somente o Hugging Face.
    #
    # Pollinations permanece desativado para evitar o erro
    # 402 Payment Required.
    #
    # =========================================================

    providers = []

    huggingface_space_url = (
        "https://xurxowsky-flux2-klein-4b-playground.hf.space"
    )

    huggingface_provider = HuggingFaceImageProvider(
        space_url=huggingface_space_url,
        timeout=settings.image_timeout_seconds,
    )

    providers.append(huggingface_provider)

    print(
        "[IMAGE] Providers configurados: "
        + ", ".join(
            provider.name
            for provider in providers
        ),
        flush=True,
    )

    # =========================================================
    # FACE SWAP
    # =========================================================

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

    # =========================================================
    # IMAGE SERVICE
    # =========================================================

    image_service = ImageService(
        chars,
        ImageProviderRouter(providers),
        ImageRepository(session),
        ImageQuota(
            session,
            settings.image_daily_limit,
            settings.image_monthly_limit,
        ),
        face_swap_service=face
