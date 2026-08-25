from dataclasses import dataclass
import random
import time


@dataclass
class ImageRequest:
    user_id: int
    character_id: int
    scene: str
    style: str = "photorealistic"
    width: int = 1024
    height: int = 1024
    reference_images: list[bytes] | None = None
    seed: int | None = None
    randomize_seed: bool = True

    def __post_init__(self):
        if self.randomize_seed or self.seed is None:
            self.seed = int(time.time() * 1000) % 2_147_483_647
            self.seed = (self.seed + random.randint(0, 999_999)) % 2_147_483_647


@dataclass
class ImageResult:
    success: bool
    provider: str | None = None
    job_id: str | None = None
    image_url: str | None = None
    image_bytes: bytes | None = None
    error: str | None = None
    face_swapped: bool = False
