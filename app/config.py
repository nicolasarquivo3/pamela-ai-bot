from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str
    database_url: str

    webhook_base_url: str | None = None
    webhook_secret: str = "change-me"

    # GEMINI
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 1000

    # AUTONOMIA
    autonomy_token: str
    autonomy_min_interval_minutes: int = 90
    autonomy_max_daily_messages: int = 3

    # IMAGENS (AI fallback)
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-2-klein-4b"

    pollinations_api_key: str | None = None
    pollinations_image_model: str = "flux"

    image_daily_limit: int = 5
    image_monthly_limit: int = 100
    image_timeout_seconds: int = 120

    # PEXELS (foto real — preferido)
    pexels_api_key: str | None = None
    prefer_real_photos: bool = True

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

    # REPLICATE (opcional)
    replicate_api_token: str | None = None
    replicate_face_swap_version: str = (
        "codeplugtech/face-swap:"
        "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"
    )

    timezone: str = "America/Sao_Paulo"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
