import json
from typing import Any

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, space_url: str, timeout: int = 180, hf_token: str | None = None):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout
        self.hf_token = hf_token

    async def available(self) -> bool:
        return bool(self.space_url)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        return headers

    def _aspect_ratio(self, request) -> str:
        width = getattr(request, "width", 1024) or 1024
        height = getattr(request, "height", 1024) or 1024

        ratio = width / height

        if 0.95 <= ratio <= 1.05:
            return "Square 1:1 / Cuadrado 1:1"

        if ratio < 0.9:
            return "Portrait 4:5 / Retrato 4:5"

        if ratio > 1.5:
            return "Landscape 16:9 / Paisaje 16:9"

        return "Portrait 4:5 / Retrato 4:5"

    def _style_preset(self, request) -> str:
        style = (getattr(request, "style", None) or "").lower()

        if "fantasy" in style or "fantasia" in style:
            return "Fantasy / Fantasia"

        return "Photoreal / Fotorrealista"

    def _find_image_reference(self, value: Any) -> str | None:
        """
        Procura recursivamente por uma URL/path de imagem dentro
        da resposta do Gradio.
        """

        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

            if (
                value.startswith("http://")
                or value.startswith("https://")
                or value.startswith("/")
            ):
                return value

            return None

        if isinstance(value, dict):

            # Estruturas comuns do Gradio FileData/ImageData
            for key in (
                "url",
                "path",
                "image",
                "file",
                "src",
            ):
                if key in value:
                    result = self._find_image_reference(value[key])

                    if result:
                        return result

            # Procura também em qualquer outro campo
            for child in value.values():
                result = self._find_image_reference(child)

                if result:
                    return result

            return None

        if isinstance(value, (list, tuple)):

            for child in value:
                result = self._find_image_reference(child)

                if result:
                    return result

        return None

    def _parse_sse(self, text: str) -> tuple[str | None, str | None]:
        """
        Retorna:
            (image_reference, error_message)
        """

        last_data = None
        sse_error = None

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("event:"):
                event_name = line[6:].strip()

                if event_name == "error":
                    sse_error = "gradio_event_error"

                continue

            if not line.startswith("data:"):
                continue

            raw = line[5:].strip()

            if not raw:
                continue

            if raw == "[DONE]":
                continue

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            last_data = parsed

            image_reference = self._find_image_reference(parsed)

            if image_reference:
                return image_reference, None

        if sse_error:
            return None, sse_error

        return None, None

    async def generate(self, request, prompt: str) -> ImageResult:

        if not await self.available():
            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # =====================================================
        # PAYLOAD CORRETO PARA O SPACE
        # =====================================================

        payload = {
            "data": {
                "t2i_prompt": prompt,
                "t2i_image": None,
                "style_preset": self._style_preset(request),
                "aspect_ratio": self._aspect_ratio(request),
                "steps": 28,
                "guidance_scale": 4.0,
                "seed": -1,
            }
        }

        endpoint = (
            f"{self.space_url}"
            "/gradio_api/call/generate_images"
        )

        headers = self._headers()

        try:

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=self.timeout,
                    write=30.0,
                    pool=30.0,
                ),
                follow_redirects=True,
            ) as client:

                # =================================================
                # 1. CRIA O JOB
                # =================================================

                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
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

                try:
                    data = response.json()
                except Exception:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "invalid_json_from_gradio: "
                            f"{response.text[:1000]}"
                        ),
                    )

                event_id = data.get("event_id")

                if not event_id:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "no_event_id: "
                            f"{data}"
                        ),
                    )

                # =================================================
                # 2. AGUARDA O RESULTADO SSE
                # =================================================

                result_endpoint = (
                    f"{self.space_url}"
                    "/gradio_api/call/generate_images/"
                    f"{event_id}"
                )

                result_response = await client.get(
                    result_endpoint,
                    headers={
                        **headers,
                        "Accept": "text/event-stream",
                    },
                )

                if result_response.status_code != 200:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"result_http_"
                            f"{result_response.status_code}: "
                            f"{result_response.text[:1000]}"
                        ),
                    )

                image_reference, sse_error = self._parse_sse(
                    result_response.text
                )

                if sse_error:

                    return ImageResult(
                        False,
                        self.name,
                        job_id=event_id,
                        error=sse_error,
                    )

                if not image_reference:

                    return ImageResult(
                        False,
                        self.name,
                        job_id=event_id,
                        error=(
                            "no_image_url_in_sse_response: "
                            f"{result_response.text[:2000]}"
                        ),
                    )

                # =================================================
                # 3. CONVERTE PATH RELATIVO EM URL
                # =================================================

                if image_reference.startswith("/"):
                    image_url = (
                        f"{self.space_url}"
                        f"{image_reference}"
                    )
                else:
                    image_url = image_reference

                # =================================================
                # 4. BAIXA A IMAGEM
                # =================================================

                image_response = await client.get(
                    image_url,
                    headers=headers,
                )

                if image_response.status_code != 200:

                    return ImageResult(
                        False,
                        self.name,
                        job_id=event_id,
                        image_url=image_url,
                        error=(
                            f"image_http_"
                            f"{image_response.status_code}: "
                            f"{image_response.text[:500]}"
                        ),
                    )

                content_type = (
                    image_response.headers
                    .get("content-type", "")
                    .lower()
                )

                if not content_type.startswith("image/"):

                    return ImageResult(
                        False,
                        self.name,
                        job_id=event_id,
                        image_url=image_url,
                        error=(
                            "downloaded_file_is_not_image: "
                            f"{content_type}"
                        ),
                    )

                # =================================================
                # 5. SUCESSO
                # =================================================

                return ImageResult(
                    True,
                    self.name,
                    job_id=event_id,
                    image_url=image_url,
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
                error=f"httpx_error: {exc}",
            )

        except Exception as exc:

            return ImageResult(
                False,
                self.name,
                error=f"unexpected_error: {exc}",
            )
