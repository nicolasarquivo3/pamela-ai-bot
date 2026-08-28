from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str
    database_url: str

    webhook_base_url: str | None = None
    webhook_secret: str = "change-me"

    # GEMINI (uma key ou varias)
    # GEMINI_API_KEY=key1
    # GEMINI_API_KEYS=key1,key2,key3   <- rodizio de cota
    gemini_api_key: str | None = None
    gemini_api_keys: str | None = None
    gemini_model: str = "gemini-flash-lite-latest"

    # OPENROUTER (fallback quando Gemini SAFETY; free rota openrouter/free)
    openrouter_api_key: str | None = None
    openrouter_model: str = "openrouter/free"
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 1000

    # AUTONOMIA
    autonomy_token: str
    autonomy_min_interval_minutes: int = 30
    autonomy_max_daily_messages: int = 12
    autonomy_tick_seconds: int = 600

    # IMAGENS
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-2-klein-4b"

    pollinations_api_key: str | None = None
    pollinations_image_model: str = "flux"
    # Stable Horde (grátis): crie key em https://stablehorde.net/register
    # ou deixe None para usar anonimo 0000000000 (mais lento)
    stable_horde_api_key: str | None = None

    # Perchance PRIORIDADE (https://perchance.org/5yf90s8rdo)
    # Renove PERCHANCE_USER_KEY quando log disser invalid_key
    perchance_user_key: str | None = None
    # Cookie header do MESMO request generate (cf_clearance=...; ...)
    perchance_cookies: str | None = None
    perchance_channel: str = "5yf90s8rdo"
    # varios geradores (mesma API): separados por virgula
    perchance_channels: str = (
        "ai-photo-generator,image-generator-professional,"
        "ai-text-to-image-generator,5yf90s8rdo"
    )
    perchance_resolution: str = "512x768" 



    image_daily_limit: int = 9999
    image_monthly_limit: int = 99999
    image_timeout_seconds: int = 120

    pexels_api_key: str | None = None
    pixabay_api_key: str | None = None
    prefer_real_photos: bool = True
    # False = NÃO manda foto em toda mensagem
    photo_every_message: bool = False
    # True = foto só quando a cena pede (provocar / look / flerte)
    photo_contextual: bool = True
    # Chance de surpresa espontânea (0.0 a 1.0). 0.08 ≈ 8%
    photo_surprise_chance: float = 0.03

    # FACE SWAP
    face_swap_enabled: bool = True
    face_swap_required: bool = True
    face_swap_provider: str = "huggingface"

    face_reference_image_path: str = "assets/pamela_face_reference.jpg"
    face_swap_timeout_seconds: int = 180

    hf_face_swap_space: str = "V0pr0S/FaceFusion-Face-Swap-Hyperswap"
    hf_face_swap_api_name: str = "/generate_image"
    hf_token: str | None = None
    hf_face_swap_model: str = "hyperswap_1b_256.onnx"
    hf_face_swap_target_index: int = 0
    hf_face_restore_model: str = "none"
    hf_face_restore_strength: float = 0.7

    replicate_api_token: str | None = None
    replicate_face_swap_version: str = (
        "codeplugtech/face-swap:"
        "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"
    )

    # ALBUM Telegram (canal privado com fotos)
    # ID: -1004349291324
    album_channel_id: str | None = "-1004349291324"
    album_enabled: bool = True
    album_first: bool = True  # tenta album antes de IA
    album_use_llm_match: bool = True
    album_use_vision_match: bool = True  # grátis; se false so tags
    # no bulk de milhares de fotos: false = nao chama LLM por foto
    album_caption_on_ingest: bool = True  # Gemini Vision tag automatica

    # ALBUM Google Drive (fotos na nuvem — libera espaco do celular)
    drive_album_enabled: bool = False
    drive_album_first: bool = True  # Drive antes do canal Telegram
    drive_folder_id: str | None = Field(default=None, validation_alias=AliasChoices("GOOGLE_DRIVE_FOLDER_ID", "DRIVE_FOLDER_ID", "drive_folder_id"))
    # JSON da service account (uma linha). Alternativa: arquivo via secret.
    google_service_account_json: str | None = Field(default=None, validation_alias=AliasChoices("GOOGLE_SERVICE_ACCOUNT_JSON", "google_service_account_json"))
    drive_auto_sync: bool = True  # tag automatica de fotos novas no Drive
    drive_sync_interval_seconds: int = 900  # 15 min
    drive_sync_batch: int = 50  # fotos por ciclo (cota Gemini)



    # MEMORIA
    memory_max_messages: int = 40  # historico recente no contexto (antes ~20)
    memory_max_facts: int = 16
    memory_max_semantic: int = 10
    memory_max_events: int = 8
    event_memory_enabled: bool = True

    timezone: str = "America/Sao_Paulo"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
