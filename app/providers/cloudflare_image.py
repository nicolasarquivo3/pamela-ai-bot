import base64
import httpx
from app.images.models import ImageResult
from app.providers.image import ImageProvider

class CloudflareImageProvider(ImageProvider):
    name = "cloudflare"
    def __init__(self, account_id, api_token, model, timeout=120):
        self.account_id, self.api_token, self.model, self.timeout = account_id, api_token, model, timeout
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    async def available(self):
        return bool(self.account_id and self.api_token and self.model)
    async def generate(self, request, prompt):
        if not await self.available():
            return ImageResult(False, self.name, error="not_configured")
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = {"prompt": prompt, "width": str(request.width), "height": str(request.height)}
        files = {}
        for i, image in enumerate((request.reference_images or [])[:4]):
            files[f"input_image_{i}"] = (f"reference_{i}.png", image, "image/png")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, headers=headers, data=data, files=files or None)
            if r.status_code == 429:
                return ImageResult(False, self.name, error="quota")
            if r.status_code in (401, 403):
                return ImageResult(False, self.name, error="authorization")
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("image/"):
                return ImageResult(True, self.name, image_bytes=r.content)
            image_b64 = (r.json().get("result") or {}).get("image")
            if image_b64:
                return ImageResult(True, self.name, image_bytes=base64.b64decode(image_b64))
            return ImageResult(False, self.name, error="no_image_in_response")
        except httpx.TimeoutException:
            return ImageResult(False, self.name, error="timeout")
        except Exception as exc:
            return ImageResult(False, self.name, error=str(exc))
