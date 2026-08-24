from abc import ABC, abstractmethod
from app.images.models import ImageRequest, ImageResult

class ImageProvider(ABC):
    name = "unknown"
    @abstractmethod
    async def available(self) -> bool: ...
    @abstractmethod
    async def generate(self, request: ImageRequest, prompt: str) -> ImageResult: ...
