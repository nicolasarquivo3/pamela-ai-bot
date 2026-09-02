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
try:
    from app.images.drive_album import DriveAlbumService
except Exception:
    DriveAlbumService = None  # type: ignore
try:
    from app.images.drive_sync_loop import DriveSyncLoop
except Exception:
    DriveSyncLoop = None  # type: ignore
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
try:
    from app.brain.event_memory import EventMemoryService
except Exception:
    try:
        from app.images.event_memory import EventMemoryService
    except Exception:
        EventMemoryService = None  # type: ignore

try:
    from app.brain.story_phase import StoryPhaseService
except Exception:
    try:
        from app.story_phase import StoryPhaseService
    except Exception:
        StoryPhaseService = None  # type: ignore

try:
    from app.brain.long_term_memory import LongTermMemoryService
except Exception:
    try:
        from app.long_term_memory import LongTermMemoryService
    except Exception:
        LongTermMemoryService = None  # type: ignore
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
try:
    from app.web import set_drive_service
except Exception:
    set_drive_service = None  # type: ignore


def make_brain_components(session):
    extractor = MemoryExtractor()
    memory_manager = MemoryManager(session, extractor, Deduplicator())
    semantic_manager = SemanticMemoryManager(session)
    emotion_engine = EmotionEngine(session)
    relationship_engine = RelationshipEngine(session)
    event_memory_service = None
    if EventMemoryService is not None:
        try:
            event_memory_service = EventMemoryService(session)
        except Exception as e:
            print(f"[EVENT-MEM] init fail: {e}", flush=True)

    story_phase_service = None
    if StoryPhaseService is not None:
        try:
            import os as _os
            story_phase_service = StoryPhaseService(
                session,
                min_days_between_advance=int(
                    getattr(settings, "story_min_days", None)
                    or _os.getenv("STORY_MIN_DAYS")
                    or 30
                ),
            )
        except Exception as e:
            print(f"[STORY] init fail: {e}", flush=True)

    long_term_memory_service = None
    if LongTermMemoryService is not None:
        try:
            long_term_memory_service = LongTermMemoryService(session)
        except Exception as e:
            print(f"[LTM] init fail: {e}", flush=True)

    context_manager = ContextManager(
        session,
        memory_manager,
        emotion_engine,
        relationship_engine,
        semantic_manager,
        event_memory_service=event_memory_service,
        story_phase_service=story_phase_service,
        long_term_memory_service=long_term_memory_service,
        max_messages=40,
        max_memories=16,
        max_semantic_memories=10,
        max_event_memories=8,
    )
    return (
        memory_manager,
        semantic_manager,
        emotion_engine,
        relationship_engine,
        context_manager,
        event_memory_service,
        story_phase_service,
        long_term_memory_service,
    )



# --- TEXT ONLY MODE: desliga imagem/drive por padrao (acelera chat) ---
import os as _os_text_only
if _os_text_only.getenv("TEXT_ONLY_MODE", "true").lower() in ("1", "true", "yes", "on"):
    _os_text_only.environ.setdefault("IMAGE_DISABLED", "true")
    _os_text_only.environ.setdefault("DRIVE_ALBUM_ENABLED", "false")
    _os_text_only.environ.setdefault("DRIVE_AUTO_SYNC", "false")
    _os_text_only.environ.setdefault("ALBUM_ENABLED", "false")
    _os_text_only.environ.setdefault("FACE_SWAP_REQUIRED", "false")
    print("[BOOT] TEXT_ONLY_MODE=true — pipeline de imagem/drive desligado", flush=True)

async def main():
    session = SessionLocal()
    chars = CharacterRepository(session)
    users = UserRepository(session)

    
    # TEXT_ONLY: nao instancia face swap / providers pesados
    _img_off = (
        os.getenv("IMAGE_DISABLED", "true").lower() in ("1", "true", "yes", "on")
        or os.getenv("TEXT_ONLY_MODE", "true").lower() in ("1", "true", "yes", "on")
    )
    providers = []
    face_swap_service = None
    web_image_service = None
    bing_image_service = None
    reddit_image_service = None
    pixabay_service = None
    gelbooru_service = None
    pexels_service = None

    if _img_off:
        print("[IMAGE] TEXT_ONLY — providers de imagem NAO carregados", flush=True)
    else:
        # 1) Perchance
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
            raw_chs = (
                getattr(settings, "perchance_channels", None)
                or _os.getenv("PERCHANCE_CHANNELS")
                or ""
            )
            extra_channels = [c.strip() for c in str(raw_chs).split(",") if c.strip()]
            import inspect as _inspect
            _pc_kwargs = {
                "user_key": pc_key.strip(),
                "cookies": (
                    getattr(settings, "perchance_cookies", None)
                    or _os.getenv("PERCHANCE_COOKIES")
                ),
                "channel": (pc_channel or "5yf90s8rdo").strip(),
                "channels": extra_channels or None,
                "timeout": int(getattr(settings, "image_timeout_seconds", 180) or 180),
                "resolution": getattr(settings, "perchance_resolution", None) or "512x768",
            }
            try:
                _sig = _inspect.signature(PerchanceImageProvider.__init__)
                _allowed = set(_sig.parameters.keys()) - {"self"}
                if any(
                    p.kind == _inspect.Parameter.VAR_KEYWORD
                    for p in _sig.parameters.values()
                ):
                    _final = _pc_kwargs
                else:
                    _final = {k: v for k, v in _pc_kwargs.items() if k in _allowed}
            except Exception:
                _final = {
                    "user_key": pc_key.strip(),
                    "timeout": int(getattr(settings, "image_timeout_seconds", 180) or 180),
                }
            providers.append(PerchanceImageProvider(**_final))
            print(
                f"[IMAGE] Perchance #1 key={pc_key[:8]}... primary={pc_channel!r}",
                flush=True,
            )
        else:
            print("[IMAGE] Perchance SEM KEY", flush=True)

        horde = StableHordeImageProvider(
            api_key=getattr(settings, "stable_horde_api_key", None),
            timeout=int(getattr(settings, "image_timeout_seconds", 180) or 180),
            width=512,
            height=768,
            nsfw=True,
        )
        providers.append(horde)

        huggingface_provider = HuggingFaceImageProvider(
            space_url="https://xurxowsky-flux2-klein-4b-playground.hf.space",
            timeout=settings.image_timeout_seconds,
        )
        providers.append(huggingface_provider)

        pol_key = getattr(settings, "pollinations_api_key", None)
        if pol_key:
            providers.append(
                PollinationsImageProvider(
                    api_key=pol_key,
                    model=getattr(settings, "pollinations_image_model", None) or "flux",
                    timeout=int(getattr(settings, "image_timeout_seconds", 120) or 120),
                    width=768,
                    height=1024,
                    max_retries=2,
                )
            )
            print("[IMAGE] Pollinations ativo", flush=True)

        print(
            "[IMAGE] Providers AI: " + ", ".join(p.name for p in providers),
            flush=True,
        )

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
        bing_image_service = BingImageSearchService(timeout=45, max_results=15)
        reddit_image_service = RedditImageSearchService(timeout=40, limit=30)
        pixabay_key = getattr(settings, "pixabay_api_key", None) or os.getenv("PIXABAY_API_KEY")
        if pixabay_key:
            pixabay_service = PixabaySearchService(api_key=pixabay_key)
        gelbooru_service = GelbooruSearchService(timeout=40, limit=30)
        if settings.pexels_api_key:
            pexels_service = PexelsSearchService(
                api_key=settings.pexels_api_key,
                timeout=30,
                per_page=40,
                orientation="portrait",
            )

    # Album Telegram (opcional — so ativa se album_service.py existir)
    album_service = None
    if (not _img_off) and AlbumService is not None and bool(getattr(settings, "album_enabled", True)):
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
            use_vision_match=bool(getattr(settings, "album_use_vision_match", True)),
            llm=None,
        )
        print(
            f"[ALBUM] enabled={album_service.enabled} channel={album_service.channel_id}",
            flush=True,
        )
    else:
        print("[ALBUM] desligado (sem modulo ou album_enabled=false)", flush=True)

    # Google Drive album
    drive_album_service = None
    try:
        import os as _os
        _df = (
            getattr(settings, "drive_folder_id", None)
            or _os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            or _os.getenv("DRIVE_FOLDER_ID")
        )
        _sj = (
            getattr(settings, "google_service_account_json", None)
            or _os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        )
        _de_raw = _os.getenv("DRIVE_ALBUM_ENABLED", "")
        _de = bool(getattr(settings, "drive_album_enabled", False)) or (
            str(_de_raw).lower() in ("1", "true", "yes")
        )
        print(
            f"[DRIVE] check enabled={_de} folder={bool(_df)} json={bool(_sj)} "
            f"cls={DriveAlbumService is not None} "
            f"folder_id={str(_df)[:24] if _df else None}...",
            flush=True,
        )
        if DriveAlbumService is None:
            print("[DRIVE] desligado: import DriveAlbumService falhou", flush=True)
        elif not _de:
            print("[DRIVE] desligado: DRIVE_ALBUM_ENABLED nao true", flush=True)
        elif not _df:
            print("[DRIVE] desligado: falta GOOGLE_DRIVE_FOLDER_ID", flush=True)
        elif not _sj:
            print("[DRIVE] desligado: falta GOOGLE_SERVICE_ACCOUNT_JSON", flush=True)
        else:
            _kw = dict(
                session=session,
                folder_id=str(_df).strip(),
                sa_json=_sj if isinstance(_sj, str) else str(_sj),
                enabled=True,
                use_vision_caption=True,
                caption_fn=None,  # wired abaixo apos album
            )
            try:
                import inspect as _ins
                if "session_factory" in _ins.signature(DriveAlbumService.__init__).parameters:
                    _kw["session_factory"] = SessionLocal
            except Exception:
                pass

            drive_album_service = DriveAlbumService(**_kw)
            print(f"[DRIVE] enabled folder={str(_df)[:20]}...", flush=True)
            # WIRE CAPTION: Gemini vision do AlbumService
            if album_service is not None and hasattr(album_service, "_auto_caption_vision"):
                drive_album_service._caption_fn = album_service._auto_caption_vision
                drive_album_service.use_vision_caption = True
                print("[DRIVE] caption_fn=album_service._auto_caption_vision", flush=True)
            else:
                # fallback: caption via album se criar mini wrapper
                print(
                    "[DRIVE] WARN: album_service sem _auto_caption_vision — "
                    "tag auto pode falhar. Use /album_drive_tag apos criar album.",
                    flush=True,
                )
    except Exception as _e:
        print(f"[DRIVE] init fail: {_e}", flush=True)
        import traceback
        traceback.print_exc()
        drive_album_service = None



    # ensure caption_fn after both album+drive exist
    if drive_album_service is not None and album_service is not None:
        if not getattr(drive_album_service, "_caption_fn", None):
            if hasattr(album_service, "_auto_caption_vision"):
                drive_album_service._caption_fn = album_service._auto_caption_vision
                drive_album_service.use_vision_caption = True
                print("[DRIVE] caption_fn ensure wired", flush=True)
        else:
            print("[DRIVE] caption_fn OK (Gemini Vision tags ON)", flush=True)
    elif drive_album_service is not None:
        print("[DRIVE] WARN: sem album_service", flush=True)
        # cria caption_fn minima se album nao existe
        try:
            from app.images.album_service import AlbumService as _AS
            _tmp = _AS(session, enabled=False)
            if hasattr(_tmp, "_auto_caption_vision"):
                drive_album_service._caption_fn = _tmp._auto_caption_vision
                drive_album_service.use_vision_caption = True
                print("[DRIVE] caption_fn via AlbumService stub", flush=True)
        except Exception as _ce:
            print(f"[DRIVE] caption stub fail: {_ce}", flush=True)


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
        drive_album_service=drive_album_service,
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
        event_memory_service,
        story_phase_service,
        long_term_memory_service,
    ) = make_brain_components(session)

    # re-cria context se settings pedem mais memoria
    try:
        if event_memory_service is not None:
            context_manager = ContextManager(
                session,
                memory_manager,
                emotion_engine,
                relationship_engine,
                semantic_manager,
                event_memory_service=event_memory_service,
                story_phase_service=story_phase_service,
                long_term_memory_service=long_term_memory_service,
                max_messages=int(getattr(settings, "memory_max_messages", 40) or 40),
                max_memories=int(getattr(settings, "memory_max_facts", 16) or 16),
                max_semantic_memories=int(
                    getattr(settings, "memory_max_semantic", 10) or 10
                ),
                max_event_memories=int(getattr(settings, "memory_max_events", 8) or 8),
            )
            print(
                f"[EVENT-MEM] on messages={getattr(settings,'memory_max_messages',40)}",
                flush=True,
            )
        print(
            f"[STORY] on={story_phase_service is not None} "
            f"[LTM] on={long_term_memory_service is not None}",
            flush=True,
        )
    except Exception as _em:
        print(f"[EVENT-MEM] setup fail: {_em}", flush=True)


    # ---------- LLM (Gemini + NSFW free + OpenRouter free) ----------
    gemini_keys = (
        getattr(settings, "gemini_api_keys", None)
        or os.getenv("GEMINI_API_KEYS")
        or ""
    )
    gemini_key = (
        getattr(settings, "gemini_api_key", None)
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    gemini_model = (
        getattr(settings, "gemini_model", None)
        or os.getenv("GEMINI_MODEL")
        or "gemini-3.5-flash-lite"
    )
    gemini = GeminiLLM(
        api_key=gemini_key or None,
        api_keys=gemini_keys or None,
        model=gemini_model,
        timeout=int(getattr(settings, "llm_timeout_seconds", 90) or 90),
        max_output_tokens=int(getattr(settings, "llm_max_output_tokens", 1000) or 1000),
    )

    try:
        from app.providers.openrouter import NSFW_FREE_MODELS, DEFAULT_FREE_MODELS
    except Exception:
        try:
            from app.llm.openrouter import NSFW_FREE_MODELS, DEFAULT_FREE_MODELS
        except Exception:
            try:
                from app.openrouter import NSFW_FREE_MODELS, DEFAULT_FREE_MODELS
            except Exception:
                NSFW_FREE_MODELS = [
                    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "google/gemma-3-27b-it:free",
                ]
                DEFAULT_FREE_MODELS = [
                    "openrouter/free",
                    "liquid/lfm-2.5-2.6b:free",
                ]

    _or_key = (
        getattr(settings, "openrouter_api_key", None)
        or os.getenv("OPENROUTER_API_KEY")
    )
    _or_timeout = int(getattr(settings, "llm_timeout_seconds", 90) or 90)
    _or_max = int(getattr(settings, "llm_max_output_tokens", 1000) or 1000)

    def _make_openrouter(model_list, label):
        import inspect as _ins
        kwargs = {
            "api_key": _or_key,
            "timeout": _or_timeout,
            "max_output_tokens": _or_max,
        }
        try:
            params = _ins.signature(OpenRouterLLM.__init__).parameters
        except Exception:
            params = {}
        ml = list(model_list or [])
        if "models" in params:
            kwargs["models"] = ml
        else:
            kwargs["model"] = ml[0] if ml else "openrouter/free"
            if "extra_models" in params:
                kwargs["extra_models"] = ml[1:] if len(ml) > 1 else []
        if "label" in params:
            kwargs["label"] = label
        return OpenRouterLLM(**kwargs)

    openrouter_nsfw = _make_openrouter(NSFW_FREE_MODELS, "OpenRouter-NSFW")
    openrouter = _make_openrouter(DEFAULT_FREE_MODELS, "OpenRouter-FREE")

    try:
        import inspect as _ins2
        rp = _ins2.signature(LLMRouter.__init__).parameters
    except Exception:
        rp = {}
    if "nsfw_fallback" in rp:
        llm = LLMRouter(
            primary=gemini,
            nsfw_fallback=openrouter_nsfw,
            free_fallback=openrouter,
        )
    else:
        llm = LLMRouter(primary=gemini, fallback=openrouter_nsfw)
        print("[LLM] LLMRouter sem nsfw_fallback — use llm_router.py novo", flush=True)

    n_keys = len(getattr(gemini, "keys", []) or [])
    print(
        f"[LLM] Router: primary=Gemini(keys={n_keys}) "
        f"fallback=OpenRouter (key={'sim' if _or_key else 'nao'})",
        flush=True,
    )

    # ---------- Agent ----------
    # TAGUEAMENTO AUTO CHECK
    if drive_album_service is not None:
        cfn = getattr(drive_album_service, "_caption_fn", None)
        print(
            f"[DRIVE] TAG check: caption_fn={'SIM' if cfn else 'NAO'} "
            f"vision={getattr(drive_album_service, 'use_vision_caption', None)}",
            flush=True,
        )
        if not cfn:
            # ultima tentativa
            try:
                if album_service is not None and hasattr(album_service, "_auto_caption_vision"):
                    drive_album_service._caption_fn = album_service._auto_caption_vision
                    print("[DRIVE] TAG check: re-wired caption_fn", flush=True)
                else:
                    from app.images.album_service import AlbumService as _AS2
                    _tmp2 = _AS2(session, enabled=False)
                    drive_album_service._caption_fn = _tmp2._auto_caption_vision
                    print("[DRIVE] TAG check: stub caption_fn", flush=True)
            except Exception as _e:
                print(f"[DRIVE] TAG check FAIL: {_e}", flush=True)

    agent = AgentBrain(

        image_service=image_service,
        user_repository=users,
        context_manager=context_manager,
        memory_manager=memory_manager,
        emotion_engine=emotion_engine,
        relationship_engine=relationship_engine,
        semantic_memory_manager=semantic_manager,
        llm=llm,
    )
    if event_memory_service is not None:
        agent.event_memory_service = event_memory_service
        # LLM opcional no event memory
        try:
            event_memory_service.llm = llm
        except Exception:
            pass
    if story_phase_service is not None:
        agent.story_phase_service = story_phase_service
    if long_term_memory_service is not None:
        agent.long_term_memory_service = long_term_memory_service

    # ---------- Telegram ----------
    telegram_app = TelegramApp(
        agent,
        album_service=album_service,
        drive_album_service=drive_album_service,
    )

    # ---------- Autonomy ----------
    try:
        autonomy = AutonomyService(
            session_factory=SessionLocal,
            telegram_bot=telegram_app.bot,
            llm=llm,
            memory_manager_factory=lambda s: MemoryManager(
                s, MemoryExtractor(), Deduplicator()
            ),
            image_service=image_service,
            min_interval_minutes=int(
                getattr(settings, "autonomy_interval_minutes", None)
                or os.getenv("AUTONOMY_INTERVAL_MINUTES")
                or 30
            ),
            max_daily_messages=int(
                getattr(settings, "autonomy_max_daily", None)
                or os.getenv("AUTONOMY_MAX_DAILY")
                or 12
            ),
        )
        agent.autonomy_service = autonomy
    except Exception as e:
        print(f"[Autonomy] init fail: {e}", flush=True)
        autonomy = None

    # ---------- Drive auto sync (background, low priority) ----------
    if (
        (not _img_off)
        and DriveSyncLoop is not None
        and drive_album_service is not None
        and bool(
            getattr(settings, "drive_auto_sync", None)
            if getattr(settings, "drive_auto_sync", None) is not None
            else (os.getenv("DRIVE_AUTO_SYNC", "true").lower() in ("1", "true", "yes"))
        )
    ):
        try:
            _iv = int(
                getattr(settings, "drive_sync_interval_seconds", None)
                or os.getenv("DRIVE_SYNC_INTERVAL_SECONDS")
                or 600
            )
            _batch = int(
                getattr(settings, "drive_sync_batch", None)
                or os.getenv("DRIVE_SYNC_BATCH")
                or 25
            )
            _tag = int(
                getattr(settings, "drive_tag_batch", None)
                or os.getenv("DRIVE_TAG_BATCH")
                or 8
            )
            drive_loop = DriveSyncLoop(
                drive_album_service,
                interval_seconds=_iv,
                batch=_batch,
                tag_batch=_tag,
                enabled=True,
            )
            await drive_loop.start()
            # boot: tagueia ja algumas (nao espera 12h)
            try:
                if hasattr(drive_album_service, "_ensure_caption_fn"):
                    drive_album_service._ensure_caption_fn()
                nboot = int(__import__("os").getenv("DRIVE_TAG_ON_BOOT", "3") or 5)
                if nboot > 0:
                    print(f"[DRIVE] boot tag {nboot} (background)...", flush=True)
                    async def _boot_tag():
                        try:
                            nb = await drive_album_service.backfill_captions(limit=nboot)
                            print(f"[DRIVE] boot tagged={nb}", flush=True)
                        except Exception as __e:
                            print(f"[DRIVE] boot tag fail: {__e}", flush=True)
                    import asyncio as _aio
                    _aio.create_task(_boot_tag())
            except Exception as _bt:
                print(f"[DRIVE] boot tag fail: {_bt}", flush=True)
        except Exception as e:
            print(f"[DriveSync] start fail: {e}", flush=True)

    # ---------- Autonomy loop ----------
    try:
        if autonomy is not None:
            start_autonomy_loop(
                agent,
                interval_seconds=int(
                    getattr(settings, "autonomy_loop_seconds", None)
                    or os.getenv("AUTONOMY_LOOP_SECONDS")
                    or 600
                ),
            )
            print("[Autonomy] loop a cada 600s", flush=True)
    except Exception as e:
        print(f"[Autonomy] loop fail: {e}", flush=True)


    # ---------- Webhook + HTTP (OBRIGATORIO para Render detectar porta) ----------
    try:
        await telegram_app.set_webhook()
    except Exception as e:
        print(f"[TelegramApp] set_webhook fail: {e}", flush=True)

    app = None
    try:
        app = create_web_app(telegram_app)
    except TypeError as e1:
        print(f"[WEB] create_web_app(telegram_app) fail: {e1}", flush=True)
        try:
            app = create_web_app(telegram_app=telegram_app)
        except TypeError as e2:
            print(f"[WEB] create_web_app(telegram_app=) fail: {e2}", flush=True)
            try:
                app = create_web_app(None, telegram_app)
            except TypeError as e3:
                print(f"[WEB] create_web_app(None, tg) fail: {e3}", flush=True)
                app = None

    if app is None:
        # fallback total: FastAPI minimo inline (nao depende de web.py do repo)
        print("[WEB] usando FastAPI inline (fallback)", flush=True)
        from fastapi import FastAPI, Request, Header, HTTPException
        from fastapi.responses import PlainTextResponse, JSONResponse

        app = FastAPI(title="pamela-ai")

        @app.get("/")
        async def _root():
            return PlainTextResponse("pamela-ai ok")

        @app.get("/health")
        async def _health():
            return {"ok": True}

        @app.post("/telegram/webhook")
        async def _wh(
            request: Request,
            x_telegram_bot_api_secret_token: str | None = Header(default=None),
        ):
            secret = (getattr(settings, "webhook_secret", None) or "").strip()
            if secret and secret not in ("change-me", "changeme", ""):
                if (x_telegram_bot_api_secret_token or "") != secret:
                    raise HTTPException(403, "bad secret")
            data = await request.json()
            try:
                await telegram_app.feed_webhook_update(data)
            except Exception as e:
                print(f"[WEB] inline webhook error: {e}", flush=True)
            return {"ok": True}

    if set_drive_service and drive_album_service is not None:
        try:
            set_drive_service(drive_album_service)
            print("[DRIVE] cron /cron/drive_tag pronto", flush=True)
        except Exception as _sd:
            print(f"[DRIVE] set_drive_service: {_sd}", flush=True)

    port = int(os.getenv("PORT") or "10000")
    print(f"[WEB] Starting server on 0.0.0.0:{port}", flush=True)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=30,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
