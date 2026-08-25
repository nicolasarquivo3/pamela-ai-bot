import asyncio
import json
import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, space_url: str, timeout: int = 180):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout

    async def available(self) -> bool:
        return bool(self.space_url)

    async def generate(self, request, prompt: str) -> ImageResult:
        if not await self.available():
            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # O Space FLUX.2 Klein 4B espera exatamente:
        #
        # mode
        # t2i_prompt
        # i2i_prompt
        # i2i_image
        # strength
        # steps
        # guidance_scale
        # seed
        #
        # Como nosso fluxo principal gera uma imagem a partir
        # de texto, usamos o modo "t2i".

        payload = {
            "data": [
                "t2i",
                prompt,
                "",
                None,
                0.8,
                4,
                4.0,
                -1,
            ]
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=30,
                )
            ) as client:

                # -------------------------------------------------
                # 1. Inicia a geração
                # -------------------------------------------------

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

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "invalid_json_from_huggingface: "
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
                            f"{response.text[:1000]}"
                        ),
                    )

                # -------------------------------------------------
                # 2. Aguarda o resultado SSE
                # -------------------------------------------------

                result_url = (
                    f"{self.space_url}/gradio_api/call/"
                    f"generate_images/{event_id}"
                )

                result_response = await client.get(
                    result_url
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

                image_url = None
                last_event = None

                for line in result_response.text.splitlines():

                    line = line.strip()

                    if not line:
                        continue

                    # Guarda o tipo de evento para diagnóstico
                    if line.startswith("event:"):
                        last_event = line[6:].strip()
                        continue

                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()

                    if not raw:
                        continue

                    # Se o Space retornar erro
                    if raw == "null" and last_event == "error":
                        return ImageResult(
                            False,
                            self.name,
                            error=(
                                "huggingface_generation_error: "
                                "event:error data:null"
                            ),
                        )

                    try:
                        result_data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # O resultado do Gallery normalmente chega
                    # como uma lista contendo os resultados.
                    if isinstance(result_data, list):

                        # Procura recursivamente por uma URL/path
                        # de imagem.
                        image_url = self._find_image_url(
                            result_data
                        )

                        if image_url:
                            break

                if not image_url:
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "no_image_url_in_sse_response: "
                            f"{result_response.text[:2000]}"
                        ),
                    )

                # -------------------------------------------------
                # 3. Normaliza URL
                # -------------------------------------------------

                if image_url.startswith("/"):
                    image_url = (
                        f"{self.space_url}{image_url}"
                    )

                elif image_url.startswith("./"):
                    image_url = (
                        f"{self.space_url}/"
                        f"{image_url[2:]}"
                    )

                # -------------------------------------------------
                # 4. Baixa a imagem
                # -------------------------------------------------

                image_response = await client.get(
                    image_url
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

                content_type = (
                    image_response.headers
                    .get("content-type", "")
                    .lower()
                )

                if not (
                    content_type.startswith("image/")
                    or image_response.content.startswith(b"\x89PNG")
                    or image_response.content.startswith(b"\xff\xd8")
                    or image_response.content.startswith(b"RIFF")
                ):
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "downloaded_content_is_not_image: "
                            f"{content_type}"
                        ),
                    )

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
                error=f"http_client_error: {exc}",
            )

        except Exception as exc:
            return ImageResult(
                False,
                self.name,
                error=f"unexpected_error: {exc}",
            )

    @staticmethod
    def _find_image_url(value):
        """
        Procura uma URL/path de imagem dentro da estrutura
        retornada pelo Gradio.

        Isso deixa o provider mais resistente a pequenas
        diferenças no formato da resposta do Gallery.
        """

        if isinstance(value, dict):

            # Formatos comuns do Gradio Gallery/FileData
            for key in (
                "url",
                "path",
            ):
                candidate = value.get(key)

                if isinstance(candidate, str):
                    if (
                        candidate.startswith("http://")
                        or candidate.startswith("https://")
                        or candidate.startswith("/")
                        or candidate.startswith("./")
                    ):
                        return candidate

            # Algumas respostas podem colocar a imagem
            # dentro de "image".
            if "image" in value:
                result = (
                    HuggingFaceImageProvider
                    ._find_image_url(
                        value["image"]
                    )
                )

                if result:
                    return result

            # Procura em outros campos
            for nested in value.values():
                result = (
                    HuggingFaceImageProvider
                    ._find_image_url(nested)
                )

                if result:
                    return result

            return None

        if isinstance(value, list):
            for item in value:
                result = (
                    HuggingFaceImageProvider
                    ._find_image_url(item)
                )

                if result:
                    return result

        return None
