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

    async def generate(
        self,
        request,
        prompt,
    ):

        if not await self.available():

            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # =====================================================
        # HEADERS
        # =====================================================

        headers = {}

        if self.hf_token:

            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        # =====================================================
        # PAYLOAD DO SPACE
        # =====================================================

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
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:

                # =================================================
                # CRIA JOB
                # =================================================

                response = await client.post(
                    (
                        f"{self.space_url}"
                        "/gradio_api/call/generate_images"
                    ),
                    headers=headers,
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
                        error=(
                            "invalid_json_from_huggingface: "
                            f"{response.text[:500]}"
                        ),
                    )

                event_id = data.get(
                    "event_id"
                )

                if not event_id:

                    return ImageResult(
                        False,
                        self.name,
                        error="no_event_id",
                    )

                # =================================================
                # AGUARDA RESULTADO SSE
                # =================================================

                result_response = await client.get(
                    (
                        f"{self.space_url}"
                        "/gradio_api/call/generate_images/"
                        f"{event_id}"
                    ),
                    headers=headers,
                )

                if result_response.status_code != 200:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"result_http_"
                            f"{result_response.status_code}: "
                            f"{result_response.text[:500]}"
                        ),
                    )

                # =================================================
                # ANALISA SSE
                # =================================================

                image_url = None
                image_path = None

                for line in result_response.text.splitlines():

                    line = line.strip()

                    if not line.startswith(
                        "data:"
                    ):
                        continue

                    raw = line[5:].strip()

                    if not raw:
                        continue

                    # ---------------------------------------------
                    # ERRO DO GRADIO
                    # ---------------------------------------------

                    if raw == "null":
                        continue

                    try:

                        result_data = json.loads(
                            raw
                        )

                    except json.JSONDecodeError:

                        continue

                    # ---------------------------------------------
                    # RESULTADO FINAL
                    # ---------------------------------------------

                    if not isinstance(
                        result_data,
                        list,
                    ):
                        continue

                    if not result_data:
                        continue

                    # O primeiro output é a Gallery.
                    gallery = result_data[0]

                    if not isinstance(
                        gallery,
                        list,
                    ):
                        continue

                    if not gallery:
                        continue

                    first = gallery[0]

                    # -------------------------------------------------
                    # GalleryImage
                    # -------------------------------------------------

                    if isinstance(
                        first,
                        dict,
                    ):

                        image = first.get(
                            "image"
                        )

                        if isinstance(
                            image,
                            dict,
                        ):

                            image_url = image.get(
                                "url"
                            )

                            image_path = image.get(
                                "path"
                            )

                        # Alguns retornos podem vir diretamente
                        # como FileData.
                        if not image_url:

                            image_url = first.get(
                                "url"
                            )

                        if not image_path:

                            image_path = first.get(
                                "path"
                            )

                    if image_url:

                        break

                # =================================================
                # SEM URL — TENTA USAR PATH
                # =================================================

                if not image_url and image_path:

                    # Se o Gradio devolveu uma URL completa,
                    # usamos diretamente.

                    if image_path.startswith(
                        "http://"
                    ) or image_path.startswith(
                        "https://"
                    ):

                        image_url = image_path

                    else:

                        image_url = (
                            f"{self.space_url}"
                            "/gradio_api/file="
                            f"{image_path}"
                        )

                # =================================================
                # NENHUMA IMAGEM
                # =================================================

                if not image_url:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "no_image_url_in_sse_response: "
                            f"{result_response.text[:1000]}"
                        ),
                    )

                # =================================================
                # DOWNLOAD DA IMAGEM
                # =================================================

                image_response = await client.get(
                    image_url,
                    headers=headers,
                )

                if image_response.status_code != 200:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"image_download_http_"
                            f"{image_response.status_code}"
                        ),
                    )

                if not image_response.content:

                    return ImageResult(
                        False,
                        self.name,
                        error="empty_image_response",
                    )

                # =================================================
                # SUCESSO
                # =================================================

                return ImageResult(
                    True,
                    self.name,
                    image_url=image_url,
                    image_bytes=image_response.content,
                )

        # =========================================================
        # TIMEOUT
        # =========================================================

        except httpx.TimeoutException:

            return ImageResult(
                False,
                self.name,
                error="timeout",
            )

        # =========================================================
        # ERRO GERAL
        # =========================================================

        except Exception as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
