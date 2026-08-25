import base64
import json

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):

    name = "huggingface"

    def __init__(
        self,
        space_url,
        timeout=180,
        hf_token=None,
    ):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout
        self.hf_token = hf_token

    async def available(self):
        return bool(self.space_url)

    async def generate(self, request, prompt):

        if not await self.available():
            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        headers = {
            "Content-Type": "application/json",
        }

        if self.hf_token:
            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        # =====================================================
        # CONFIGURAÇÃO DA IMAGEM
        # =====================================================

        width = getattr(request, "width", 1024)
        height = getattr(request, "height", 1024)

        if width == height:
            aspect_ratio = (
                "Square 1:1 / Cuadrado 1:1"
            )

        elif height > width:
            if height / width >= 1.6:
                aspect_ratio = (
                    "Portrait 9:16 / Retrato 9:16"
                )
            else:
                aspect_ratio = (
                    "Portrait 4:5 / Retrato 4:5"
                )

        else:
            if width / height >= 1.6:
                aspect_ratio = (
                    "Landscape 16:9 / Paisaje 16:9"
                )
            else:
                aspect_ratio = (
                    "Landscape 16:9 / Paisaje 16:9"
                )

        # =====================================================
        # IMAGE-TO-IMAGE
        # =====================================================

        reference_images = (
            getattr(
                request,
                "reference_images",
                None,
            )
            or []
        )

        i2i_image = None
        i2i_prompt = ""

        if reference_images:

            reference = reference_images[0]

            if isinstance(reference, bytes):

                encoded = base64.b64encode(
                    reference
                ).decode("utf-8")

                i2i_image = {
                    "path": None,
                    "url": (
                        "data:image/png;base64,"
                        + encoded
                    ),
                    "size": len(reference),
                    "orig_name": "reference.png",
                    "mime_type": "image/png",
                    "is_stream": False,
                    "meta": {
                        "_type": "gradio.FileData"
                    },
                }

                i2i_prompt = prompt

        # =====================================================
        # PAYLOAD DO OPENAPI DO SEU SPACE
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
                # CHAMADA AO ENDPOINT /run
                # =================================================

                response = await client.post(
                    endpoint,
                    headers=headers,
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

                # =================================================
                # RESPOSTA JSON
                # =================================================

                try:
                    data = response.json()

                except json.JSONDecodeError:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "invalid_json_response: "
                            f"{response.text[:1000]}"
                        ),
                    )

                # =================================================
                # LOG ÚTIL PARA DEBUG
                # =================================================

                print(
                    "[HF] Response:",
                    str(data)[:2000],
                    flush=True,
                )

                # =================================================
                # POSSÍVEIS FORMATOS DE RESPOSTA
                # =================================================

                outputs = None

                if isinstance(data, dict):

                    # Formato:
                    # {"data": [...]}

                    if isinstance(
                        data.get("data"),
                        list,
                    ):
                        outputs = data["data"]

                    # Formato:
                    # {"output": [...]}

                    elif isinstance(
                        data.get("output"),
                        list,
                    ):
                        outputs = data["output"]

                    # Formato:
                    # {"outputs": [...]}

                    elif isinstance(
                        data.get("outputs"),
                        list,
                    ):
                        outputs = data["outputs"]

                    # Caso o /run devolva um event_id,
                    # tratamos abaixo.

                    event_id = data.get(
                        "event_id"
                    )

                    if event_id:

                        return await self._read_event(
                            client,
                            event_id,
                            headers,
                        )

                # =================================================
                # EXTRAI IMAGEM
                # =================================================

                image_url = self._extract_image_url(
                    outputs
                )

                if not image_url:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "no_image_in_response: "
                            f"{str(data)[:2000]}"
                        ),
                    )

                # =================================================
                # DOWNLOAD
                # =================================================

                image_response = await client.get(
                    image_url,
                    headers=headers,
                )

                if (
                    image_response.status_code
                    != 200
                ):

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "image_download_http_"
                            f"{image_response.status_code}: "
                            f"{image_response.text[:500]}"
                        ),
                    )

                if not image_response.content:

                    return ImageResult(
                        False,
                        self.name,
                        error="empty_image_response",
                    )

                return ImageResult(
                    True,
                    self.name,
                    image_url=image_url,
                    image_bytes=(
                        image_response.content
                    ),
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
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    # =========================================================
    # EVENTO SSE — FALLBACK
    # =========================================================

    async def _read_event(
        self,
        client,
        event_id,
        headers,
    ):

        endpoint = (
            f"{self.space_url}"
            "/gradio_api/call/generate_images/"
            f"{event_id}"
        )

        response = await client.get(
            endpoint,
            headers=headers,
        )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_{response.status_code}: "
                    f"{response.text[:1000]}"
                ),
            )

        for block in response.text.split(
            "\n\n"
        ):

            event_name = None
            event_data = None

            for line in block.splitlines():

                if line.startswith(
                    "event:"
                ):

                    event_name = (
                        line[6:].strip()
                    )

                elif line.startswith(
                    "data:"
                ):

                    event_data = (
                        line[5:].strip()
                    )

            if event_name == "error":

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        "huggingface_event_error: "
                        + (
                            event_data
                            or "unknown_error"
                        )
                    ),
                )

            if event_name != "complete":
                continue

            if not event_data:
                continue

            try:

                outputs = json.loads(
                    event_data
                )

            except json.JSONDecodeError:

                continue

            image_url = self._extract_image_url(
                outputs
            )

            if not image_url:

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        "complete_without_image: "
                        f"{event_data[:2000]}"
                    ),
                )

            image_response = await client.get(
                image_url,
                headers=headers,
            )

            if image_response.status_code != 200:

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        "image_download_http_"
                        f"{image_response.status_code}"
                    ),
                )

            return ImageResult(
                True,
                self.name,
                image_url=image_url,
                image_bytes=(
                    image_response.content
                ),
            )

        return ImageResult(
            False,
            self.name,
            error=(
                "sse_finished_without_complete"
            ),
        )

    # =========================================================
    # EXTRATOR DE IMAGEM
    # =========================================================

    @staticmethod
    def _extract_image_url(outputs):

        if not outputs:
            return None

        # -----------------------------------------------------
        # Procura recursivamente por FileData
        # -----------------------------------------------------

        def find_url(value):

            if isinstance(
                value,
                dict,
            ):

                # FileData padrão do Gradio
                url = value.get("url")

                if (
                    isinstance(url, str)
                    and url.startswith(
                        (
                            "http://",
                            "https://",
                        )
                    )
                ):
                    return url

                # Alguns resultados podem ter
                # image dentro de outro objeto.

                for key in (
                    "image",
                    "file",
                    "data",
                    "output",
                ):

                    if key in value:

                        result = find_url(
                            value[key]
                        )

                        if result:
                            return result

                return None

            if isinstance(
                value,
                list,
            ):

                for item in value:

                    result = find_url(item)

                    if result:
                        return result

            return None

        return find_url(outputs)
