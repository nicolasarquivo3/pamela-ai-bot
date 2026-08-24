import asyncio
import json
import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, space_url, timeout=180):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout

    async def available(self):
        return bool(self.space_url)

    async def generate(self, request, prompt):
        if not await self.available():
            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        payload = {
            "data": [
                prompt,
                "",
                None,
                0.0,
                "Photoreal / Fotorrealista",
                "Portrait 4:5 / Retrato 4:5",
                28,
                4.0,
                -1,
            ]
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    f"{self.space_url}/gradio_api/call/generate_images",
                    json=payload,
                )

                if response.status_code != 200:
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"http_{response.status_code}: "
                            f"{response.text[:500]}"
                        ),
                    )

                data = response.json()
                event_id = data.get("event_id")

                if not event_id:
                    return ImageResult(
                        False,
                        self.name,
                        error="no_event_id",
                    )

                result_response = await client.get(
                    f"{self.space_url}/gradio_api/call/"
                    f"generate_images/{event_id}"
                )

                if result_response.status_code != 200:
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"result_http_"
                            f"{result_response.status_code}"
                        ),
                    )

                image_url = None

                for line in result_response.text.splitlines():
                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()

                    if not raw:
                        continue

                    try:
                        result_data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(result_data, list):
                        continue

                    if not result_data:
                        continue

                    gallery = result_data[0]

                    if not isinstance(gallery, list):
                        continue

                    if not gallery:
                        continue

                    first = gallery[0]

                    if isinstance(first, dict):
                        image = first.get("image", first)

                        if isinstance(image, dict):
                            image_url = (
                                image.get("url")
                                or image.get("path")
                            )

                    if image_url:
                        break

                if not image_url:
                    return ImageResult(
                        False,
                        self.name,
                        error="no_image_url",
                    )

                if image_url.startswith("/"):
                    image_url = (
                        f"{self.space_url}{image_url}"
                    )

                image_response = await client.get(
                    image_url
                )

                image_response.raise_for_status()

                return ImageResult(
                    True,
                    self.name,
                    image_bytes=image_response.content,
                )

        except httpx.TimeoutException:
            return ImageResult(
                False,
                self.name,
                error="timeout",
            )

        except Exception as exc:
            return ImageResult(
                False,
                self.name,
                error=str(exc),
            )
