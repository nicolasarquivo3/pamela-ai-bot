"app/providers/huggingface_image.py"

import json
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider

class HuggingFaceImageProvider(ImageProvider):
name = "huggingface"

def __init__(
    self,
    space_url: str,
    timeout: int = 180,
    hf_token: str | None = None,
):
    self.space_url = space_url.rstrip("/")
    self.timeout = timeout
    self.hf_token = hf_token

async def available(self) -> bool:
    return bool(self.space_url)

def _headers(self) -> dict:
    headers = {
        "Accept": "application/json",
    }

    if self.hf_token:
        headers["Authorization"] = f"Bearer {self.hf_token}"

    return headers

# =========================================================
# EXTRAÇÃO DE IMAGEM
# =========================================================

@classmethod
def _extract_image_info(cls, data):
    """
    Procura recursivamente uma imagem dentro da resposta
    retornada pelo Gradio.

    O Space informado pelo usuário retorna:

        output -> GalleryData
        output_1 -> string

    GalleryData normalmente possui:

        [
            {
                "image": {
                    "path": "...",
                    "url": "...",
                    ...
                },
                "caption": ...
            }
        ]
    """

    if data is None:
        return None

    # -----------------------------------------------------
    # Lista
    # -----------------------------------------------------

    if isinstance(data, list):
        for item in data:
            result = cls._extract_image_info(item)

            if result:
                return result

        return None

    # -----------------------------------------------------
    # String
    # -----------------------------------------------------

    if isinstance(data, str):
        value = data.strip()

        if (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("/")
        ):
            return {
                "url": (
                    value
                    if value.startswith(("http://", "https://"))
                    else None
                ),
                "path": (
                    value
                    if value.startswith("/")
                    else None
                ),
            }

        return None

    # -----------------------------------------------------
    # Dicionário
    # -----------------------------------------------------

    if not isinstance(data, dict):
        return None

    # Caso seja diretamente um objeto ImageData.

    image_url = data.get("url")
    image_path = data.get("path")

    if image_url or image_path:
        return {
            "url": image_url,
            "path": image_path,
        }

    # Caso seja GalleryImage.

    image = data.get("image")

    if isinstance(image, dict):
        image_url = image.get("url")
        image_path = image.get("path")

        if image_url or image_path:
            return {
                "url": image_url,
                "path": image_path,
            }

    # Caso exista output.

    if "output" in data:
        result = cls._extract_image_info(
            data.get("output")
        )

        if result:
            return result

    # Alguns wrappers podem utilizar outros nomes.

    for key in (
        "outputs",
        "gallery",
        "result",
        "results",
        "data",
    ):
        if key not in data:
            continue

        result = cls._extract_image_info(
            data.get(key)
        )

        if result:
            return result

    return None

# =========================================================
# DOWNLOAD DA IMAGEM
# =========================================================

async def _download_image(
    self,
    client: httpx.AsyncClient,
    image_url: str | None,
    image_path: str | None,
) -> bytes | None:

    candidates: list[str] = []

    # -----------------------------------------------------
    # URL direta
    # -----------------------------------------------------

    if image_url:
        if image_url.startswith(
            (
                "http://",
                "https://",
            )
        ):
            candidates.append(image_url)

        elif image_url.startswith("/"):
            candidates.append(
                f"{self.space_url}{image_url}"
            )

    # -----------------------------------------------------
    # Path do Gradio
    # -----------------------------------------------------

    if image_path:

        if image_path.startswith(
            (
                "http://",
                "https://",
            )
        ):
            candidates.append(image_path)

        else:
            clean_path = image_path

            if clean_path.startswith("/"):
                clean_path = clean_path[1:]

            encoded_path = quote(
                clean_path,
                safe="",
            )

            # Gradio moderno.

            candidates.append(
                f"{self.space_url}/gradio_api/file="
                f"{encoded_path}"
            )

            # Compatibilidade com versões anteriores.

            candidates.append(
                f"{self.space_url}/file="
                f"{encoded_path}"
            )

            # Caso o path já seja uma rota do Space.

            if image_path.startswith("/"):
                candidates.append(
                    f"{self.space_url}{image_path}"
                )

    # -----------------------------------------------------
    # Remover duplicados
    # -----------------------------------------------------

    candidates = list(
        dict.fromkeys(candidates)
    )

    if not candidates:
        return None

    # -----------------------------------------------------
    # Tentar cada URL
    # -----------------------------------------------------

    for candidate in candidates:

        try:
            response = await client.get(
                candidate,
                headers=self._headers(),
            )

            if response.status_code != 200:
                continue

            content = response.content

            if not content:
                continue

            content_type = (
                response.headers.get(
                    "content-type",
                    "",
                )
                .lower()
            )

            # -------------------------------------------------
            # MIME type
            # -------------------------------------------------

            if content_type.startswith(
                "image/"
            ):
                return content

            # -------------------------------------------------
            # PNG
            # -------------------------------------------------

            if content.startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                return content

            # -------------------------------------------------
            # JPEG
            # -------------------------------------------------

            if content.startswith(
                b"\xff\xd8\xff"
            ):
                return content

            # -------------------------------------------------
            # WEBP
            # -------------------------------------------------

            if (
                content.startswith(b"RIFF")
                and b"WEBP" in content[:16]
            ):
                return content

        except (
            httpx.HTTPError,
            OSError,
        ):
            continue

    return None

# =========================================================
# PARÂMETROS DO SPACE
# =========================================================

@staticmethod
def _build_generation_parameters(
    request=None,
) -> dict:

    style_preset = (
        "Photoreal / Fotorrealista"
    )

    aspect_ratio = (
        "Portrait 4:5 / Retrato 4:5"
    )

    steps = 28
    guidance_scale = 4.0
    seed = -1
    strength = 0.0

    # -----------------------------------------------------
    # Ajustar proporção
    # -----------------------------------------------------

    if request is not None:

        width = getattr(
            request,
            "width",
            1024,
        )

        height = getattr(
            request,
            "height",
            1024,
        )

        try:
            width = int(width)
            height = int(height)
        except (
            TypeError,
            ValueError,
        ):
            width = 1024
            height = 1024

        if width <= 0:
            width = 1024

        if height <= 0:
            height = 1024

        ratio = width / height

        if ratio >= 1.6:
            aspect_ratio = (
                "Wide 16:9 / Panoramico 16:9"
            )

        elif ratio > 1.05:
            aspect_ratio = (
                "Landscape 5:4 / Horizontal 5:4"
            )

        elif ratio <= 0.625:
            aspect_ratio = (
                "Tall 9:16 / Vertical 9:16"
            )

        elif ratio < 0.95:
            aspect_ratio = (
                "Portrait 4:5 / Retrato 4:5"
            )

        else:
            aspect_ratio = (
                "Square 1:1 / Cuadrado 1:1"
            )

        # -------------------------------------------------
        # Estilo
        # -------------------------------------------------

        requested_style = getattr(
            request,
            "style",
            None,
        )

        if requested_style:

            normalized_style = str(
                requested_style
            ).lower()

            if "cinematic" in normalized_style:
                style_preset = (
                    "Cinematic / Cinematografico"
                )

            elif "illustration" in normalized_style:
                style_preset = (
                    "Illustration / Ilustracion"
                )

            elif "editorial" in normalized_style:
                style_preset = (
                    "Editorial / Editorial"
                )

            elif "fantasy" in normalized_style:
                style_preset = (
                    "Fantasy / Fantasia"
                )

            else:
                style_preset = (
                    "Photoreal / Fotorrealista"
                )

        # -------------------------------------------------
        # Steps
        # -------------------------------------------------

        requested_steps = getattr(
            request,
            "steps",
            None,
        )

        if requested_steps is not None:

            try:
                steps = int(
                    requested_steps
                )
            except (
                TypeError,
                ValueError,
            ):
                steps = 28

        steps = max(
            1,
            min(50, steps),
        )

        # -------------------------------------------------
        # Guidance
        # -------------------------------------------------

        requested_guidance = getattr(
            request,
            "guidance_scale",
            None,
        )

        if requested_guidance is not None:

            try:
                guidance_scale = float(
                    requested_guidance
                )
            except (
                TypeError,
                ValueError,
            ):
                guidance_scale = 4.0

        guidance_scale = max(
            1.0,
            min(10.0, guidance_scale),
        )

        # -------------------------------------------------
        # Seed
        # -------------------------------------------------

        requested_seed = getattr(
            request,
            "seed",
            None,
        )

        if requested_seed is not None:

            try:
                seed = int(
                    requested_seed
                )
            except (
                TypeError,
                ValueError,
            ):
                seed = -1

    return {
        "style_preset": style_preset,
        "aspect_ratio": aspect_ratio,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "strength": strength,
    }

# =========================================================
# SSE / GRADIO
# =========================================================

async def _generate_sse(
    self,
    client: httpx.AsyncClient,
    prompt: str,
    request=None,
) -> ImageResult:

    """
    Usa o endpoint oficial de eventos do Gradio:

        POST /gradio_api/call/generate_images

    seguido de:

        GET /gradio_api/call/generate_images/{event_id}

    O Space fornecido pelo usuário documenta
    generate_images com exatamente 9 argumentos:

        1. t2i_prompt
        2. i2i_prompt
        3. i2i_image
        4. strength
        5. style_preset
        6. aspect_ratio
        7. steps
        8. guidance_scale
        9. seed
    """

    parameters = (
        self._build_generation_parameters(
            request
        )
    )

    # -----------------------------------------------------
    # IMPORTANTE
    #
    # Para /gradio_api/call/... o Gradio espera:
    #
    # {
    #     "data": [
    #         ...
    #     ]
    # }
    #
    # A ordem precisa ser exatamente a ordem
    # definida pelo Space.
    # -----------------------------------------------------

    payload = {
        "data": [
            prompt,
            "",
            None,
            parameters["strength"],
            parameters["style_preset"],
            parameters["aspect_ratio"],
            parameters["steps"],
            parameters["guidance_scale"],
            parameters["seed"],
        ]
    }

    call_url = (
        f"{self.space_url}"
        "/gradio_api/call/generate_images"
    )

    try:

        response = await client.post(
            call_url,
            json=payload,
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
        )

    except httpx.HTTPError as exc:

        return ImageResult(
            False,
            self.name,
            error=(
                "sse_call_http_error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    # -----------------------------------------------------
    # Erro HTTP inicial
    # -----------------------------------------------------

    if response.status_code != 200:

        return ImageResult(
            False,
            self.name,
            error=(
                f"sse_call_http_"
                f"{response.status_code}: "
                f"{response.text[:1500]}"
            ),
        )

    # -----------------------------------------------------
    # Obter event_id
    # -----------------------------------------------------

    try:
        initial_data = response.json()

    except json.JSONDecodeError:

        return ImageResult(
            False,
            self.name,
            error=(
                "sse_invalid_initial_response: "
                f"{response.text[:1000]}"
            ),
        )

    event_id = initial_data.get(
        "event_id"
    )

    if not event_id:

        return ImageResult(
            False,
            self.name,
            error=(
                "sse_no_event_id: "
                f"{json.dumps(initial_data, ensure_ascii=False)[:1500]}"
            ),
        )

    # -----------------------------------------------------
    # Buscar resultado
    # -----------------------------------------------------

    result_url = (
        f"{self.space_url}"
        "/gradio_api/call/generate_images/"
        f"{event_id}"
    )

    try:

        result_response = await client.get(
            result_url,
            headers={
                "Accept": "text/event-stream",
                **(
                    {
                        "Authorization": (
                            f"Bearer {self.hf_token}"
                        )
                    }
                    if self.hf_token
                    else {}
                ),
            },
        )

    except httpx.HTTPError as exc:

        return ImageResult(
            False,
            self.name,
            error=(
                "sse_result_http_error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    if result_response.status_code != 200:

        return ImageResult(
            False,
            self.name,
            error=(
                f"sse_result_http_"
                f"{result_response.status_code}: "
                f"{result_response.text[:1500]}"
            ),
        )

    # -----------------------------------------------------
    # Processar SSE
    # -----------------------------------------------------

    last_event = None
    last_data = None

    for raw_line in result_response.text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # -------------------------------------------------
        # event:
        # -------------------------------------------------

        if line.startswith("event:"):

            last_event = line[
                len("event:"):
            ].strip()

            continue

        # -------------------------------------------------
        # data:
        # -------------------------------------------------

        if not line.startswith("data:"):
            continue

        raw_data = line[
            len("data:"):
        ].strip()

        if not raw_data:
            continue

        # -------------------------------------------------
        # Alguns eventos podem devolver JSON.
        # -------------------------------------------------

        try:
            event_data = json.loads(
                raw_data
            )
        except json.JSONDecodeError:

            # Pode ser texto simples.
            event_data = raw_data

        last_data = event_data

        # -------------------------------------------------
        # Evento de erro
        # -------------------------------------------------

        if last_event == "error":

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_generation_error: "
                    f"{event_data}"
                ),
            )

        if isinstance(event_data, dict):

            error_value = event_data.get(
                "error"
            )

            if error_value:

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        "sse_generation_error: "
                        f"{error_value}"
                    ),
                )

        # -------------------------------------------------
        # Procurar imagem
        # -------------------------------------------------

        image_info = (
            self._extract_image_info(
                event_data
            )
        )

        if not image_info:
            continue

        image_url = image_info.get(
            "url"
        )

        image_path = image_info.get(
            "path"
        )

        image_bytes = (
            await self._download_image(
                client,
                image_url,
                image_path,
            )
        )

        if not image_bytes:
            continue

        return ImageResult(
            True,
            self.name,
            image_url=image_url,
            image_bytes=image_bytes,
        )

    # -----------------------------------------------------
    # Não encontrou imagem
    # -----------------------------------------------------

    debug_data = ""

    if last_data is not None:

        try:
            debug_data = json.dumps(
                last_data,
                ensure_ascii=False,
            )[:1500]

        except Exception:
            debug_data = str(
                last_data
            )[:1500]

    return ImageResult(
        False,
        self.name,
        error=(
            "no_image_in_sse_response: "
            f"last_event={last_event!r}; "
            f"last_data={debug_data}"
        ),
    )

# =========================================================
# MÉTODO PRINCIPAL
# =========================================================

async def generate(
    self,
    request,
    prompt: str,
) -> ImageResult:

    if not await self.available():

        return ImageResult(
            False,
            self.name,
            error="not_configured",
        )

    if not prompt or not prompt.strip():

        return ImageResult(
            False,
            self.name,
            error="empty_prompt",
        )

    try:

        timeout = httpx.Timeout(
            connect=30.0,
            read=self.timeout,
            write=self.timeout,
            pool=30.0,
        )

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:

            # -------------------------------------------------
            # O Space fornecido documenta /run/generate_images,
            # mas o endpoint /run está retornando 405 no
            # ambiente atual.
            #
            # Portanto usamos diretamente o mecanismo oficial
            # de chamada assíncrona do Gradio.
            # -------------------------------------------------

            result = await self._generate_sse(
                client,
                prompt,
                request,
            )

            return result

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
                "http_client_error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    except Exception as exc:

        return ImageResult(
            False,
            self.name,
            error=(
                "unexpected_error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
