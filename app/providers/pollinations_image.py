import httpx
from app.images.models import ImageResult
from app.providers.image import ImageProvider

class PollinationsImageProvider(ImageProvider):
    name = "pollinations"
    def __init__(self, api_key, model="flux", timeout=120):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.url = "https://gen.pollinations.ai/v1/images/generations"
    async def available(self):
        return bool(self.api_key)
    async def generate(self, request, prompt):
        if not await self.available():
            return ImageResult(False, self.name, error="not_configured")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt, "size": f"{request.width}x{request.height}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, headers=headers, json=payload)
            if r.status_code == 429:
                return ImageResult(False, self.name, error="quota")
            r.raise_for_status()
            data = r.json().get("data") or []
            url = data[0].get("url") if data else None
            return ImageResult(True, self.name, image_url=url) if url else ImageResult(False, self.name, error="no_url")
        except httpx.TimeoutException:
            return ImageResult(False, self.name, error="timeout")
        except Exception as exc:
            return ImageResult(False, self.name, error=str(exc))
