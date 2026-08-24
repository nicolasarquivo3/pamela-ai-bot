from dataclasses import dataclass

@dataclass
class ImageRequest:
    user_id: int
    character_id: int
    scene: str
    style: str = "photorealistic"
    width: int = 1024
    height: int = 1024
    reference_images: list[bytes] | None = None

@dataclass
class ImageResult:
    success: bool
    provider: str | None = None
    job_id: str | None = None
    image_url: str | None = None
    image_bytes: bytes | None = None
    error: str | None = None
    face_swapped: bool = False
