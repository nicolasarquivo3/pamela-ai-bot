import json
import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, space_url, timeout=180, hf_token=None):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout
        self.hf_token = hf_token

    async def available(self):
        return bool(self.space_url)

    def _headers(self):
        headers = {
            "Accept": "application/json",
        }

        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        return headers

    async def generate(self, request, prompt):
        if not await self.available():
            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # API real do Space:
        #
        # generate_images(
        #     mode,
        #     t2i_prompt,
        #     i2i_prompt,
        #     i2i_image,
        #     strength,
        #     steps,
        #     guidance_scale,
        #     seed
        # )

        payload = {
            "data": [
                "t2i",
                prompt,
                "",
                None,
                0.0,
                4,
                4.0,
                -1,
            ]
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._headers(),
                follow_redirects=True,
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
                            f"{response.text[:1000]}"
                        ),
                    )

                data = response.json()
                event_id = data.get("event_id")

                if not event_id:
                    return ImageResult(
                        False,
                        self.name,
                        error=f"no_event_id: {response.text[:1000]}",
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
                            f"result_http_{result_response.status_code}: "
                            f"{result_response.text[:1000]}"
                        ),
                    )

                image_url = None
                final_error = None

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

                    if isinstance(result_data, dict):
                        final_error = result_data.get("error")
                        continue

                    if not isinstance(result_data, list):
                        continue

                    if len(result_data) < 1:
                        continue

                    gallery = result_data[0]

                    if not isinstance(gallery, list):
                        continue

                    for item in gallery:

                        if not isinstance(item, dict):
                            continue

                        image = item.get("image", item)

                        if not isinstance(image, dict):
                            continue

                        image_url = (
                            image.get("url")
                            or image.get("path")
                        )

                        if image_url:
                            break

                    if image_url:
                        break

                if not image_url:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "no_image_url_in_sse_response"
                            + (
                                f": {final_error}"
                                if final_error
                                else ""
                            )
                        ),
                    )

                if image_url.startswith("/"):
                    image_url = (
                        f"{self.space_url}{image_url}"
                    )

                image_response = await client.get(
                    image_url,
                    headers=self._headers(),
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

        except httpx.HTTPError as exc:
            return ImageResult(
                False,
                self.name,
                error=f"http_error: {exc}",
            )

        except Exception as exc:
            return ImageResult(
                False,
                self.name,
                error=f"{type(exc).__name__}: {exc}",
            )
