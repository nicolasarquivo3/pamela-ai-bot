import base64
import urllib.parse

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

        # =====================================================
        # PARÂMETROS DO SPACE
        # =====================================================

        width = getattr(request, "width", 512)
        height = getattr(request, "height", 512)

        # O Space trabalha com aspect ratios próprios.
        if height > width:
            if height / max(width, 1) >= 1.6:
                aspect_ratio = "Tall 9:16 / Vertical 9:16"
            else:
                aspect_ratio = "Portrait 4:5 / Retrato 4:5"

        elif width / max(height, 1) >= 1.6:
            aspect_ratio = "Wide 16:9 / Panoramico 16:9"

        else:
            aspect_ratio = "Square 1:1 / Cuadrado 1:1"

        # =====================================================
        # IMAGEM DE REFERÊNCIA
        # =====================================================

        i2i_image = None
        i2i_prompt = ""

        reference_images = (
            getattr(request, "reference_images", None)
            or []
        )

        if reference_images:

            reference = reference_images[0]

            if isinstance(reference, bytes):

                encoded = base64.b64encode(
                    reference
                ).decode("utf-8")

                i2i_image = {
                    "url": (
                        "data:image/png;base64,"
                        + encoded
                    ),
                    "meta": {
                        "_type": "gradio.FileData"
                    },
                }

                i2i_prompt = prompt

        # =====================================================
        # PAYLOAD COMPATÍVEL COM O OPENAPI DO SPACE
        # =====================================================

        payload = {
            "t2i_prompt": prompt,
            "i2i_prompt": i2i_prompt,
            "i2i_image": i2i_image,
            "strength": 0.0,
            "style_preset": (
                "Photoreal / Fotorrealista"
            ),
            "aspect_ratio": aspect_ratio,
            "steps": 28,
            "guidance_scale": 4.0,
            "seed": -1,
        }

        endpoint = (
            f"{self.space_url}"
            "/gradio_api/run/generate_images"
        )

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:

                # =================================================
                # CHAMADA AO SPACE
                # =================================================

                response = await client.post(
                    endpoint,
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

                try:
                    data = response.json()

                except Exception:

                    return ImageResult(
                        False,
                        self.name,
                        error="invalid_json_response",
                    )

                # =================================================
                # RESULTADO DO GRADIO
                # =================================================

                output = data.get("output")

                if not output:
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            data.get("output_1")
                            or "no_output"
                        ),
                    )

                if not isinstance(output, list):
                    return ImageResult(
                        False,
                        self.name,
                        error="invalid_output_format",
                    )

                if not output:
                    return ImageResult(
                        False,
                        self.name,
                        error="empty_gallery",
                    )

                first = output[0]

                # =================================================
                # GALERIA -> IMAGE DATA
                # =================================================

                image_data = None

                if isinstance(first, dict):

                    image_data = first.get(
                        "image"
                    )

                if not isinstance(
                    image_data,
                    dict,
                ):

                    return ImageResult(
                        False,
                        self.name,
                        error="no_image_data",
                    )

                image_url = image_data.get("url")

                image_path = image_data.get(
                    "path"
                )

                # =================================================
                # URL DISPONÍVEL
                # =================================================

                if image_url:

                    image_response = (
                        await client.get(
                            image_url
                        )
                    )

                    image_response.raise_for_status()

                    return ImageResult(
                        True,
                        self.name,
                        image_bytes=(
                            image_response.content
                        ),
                    )

                # =================================================
                # FALLBACK: PATH DO GRADIO
                # =================================================

                if image_path:

                    encoded_path = (
                        urllib.parse.quote(
                            image_path,
                            safe="",
                        )
                    )

                    possible_urls = [
                        (
                            f"{self.space_url}"
                            f"/gradio_api/file="
                            f"{encoded_path}"
                        ),
                        (
                            f"{self.space_url}"
                            f"/file="
                            f"{encoded_path}"
                        ),
                    ]

                    for file_url in possible_urls:

                        try:

                            image_response = (
                                await client.get(
                                    file_url
                                )
                            )

                            if (
                                image_response.status_code
                                == 200
                            ):

                                return ImageResult(
                                    True,
                                    self.name,
                                    image_bytes=(
                                        image_response.content
                                    ),
                                )

                        except Exception:
                            continue

                return ImageResult(
                    False,
                    self.name,
                    error="image_url_not_available",
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
                error=(
                    f"http_error:{str(exc)}"
                ),
            )

        except Exception as exc:

            return ImageResult(
                False,
                self.name,
                error=str(exc),
            )
