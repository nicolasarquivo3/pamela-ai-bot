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

    # =========================================================
    # HEADERS
    # =========================================================

    def _headers(self) -> dict[str, str]:
        """
        Cabeçalhos usados nas chamadas ao Gradio/Hugging Face.

        O Gradio Client atual utiliza x-hf-authorization para
        transportar a autorização do Hugging Face e também
        identifica chamadas de API com x-gradio-user=api.

        Mantemos Authorization também para compatibilidade com
        a API REST documentada do Hugging Face.
        """

        headers = {
            "Accept": "application/json",
            "User-Agent": "pamela-ai/1.0",
            "x-gradio-user": "api",
        }

        if self.hf_token:
            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

            headers["x-hf-authorization"] = (
                f"Bearer {self.hf_token}"
            )

        return headers

    def _sse_headers(self) -> dict[str, str]:
        """
        Cabeçalhos específicos para a conexão SSE.
        """

        headers = self._headers()

        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"

        return headers

    # =========================================================
    # UTILITÁRIOS
    # =========================================================

    @staticmethod
    def _is_image_dict(value) -> bool:
        if not isinstance(value, dict):
            return False

        return bool(
            value.get("url")
            or value.get("path")
        )

    @classmethod
    def _extract_image_info(cls, value):
        """
        Procura recursivamente uma imagem dentro da resposta.

        O Space pode retornar estruturas como:

            [
                {
                    "image": {
                        "path": "...",
                        "url": "..."
                    }
                }
            ]

        ou:

            [
                [
                    {
                        "image": {
                            "path": "...",
                            "url": "..."
                        }
                    }
                ],
                "status"
            ]

        ou diretamente:

            {
                "image": {
                    "path": "...",
                    "url": "..."
                }
            }

        Esta função trata todos esses formatos.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # Dicionário
        # -----------------------------------------------------

        if isinstance(value, dict):

            # Objeto de imagem direto
            if cls._is_image_dict(value):
                return {
                    "url": value.get("url"),
                    "path": value.get("path"),
                }

            # Campo "image"
            image = value.get("image")

            if isinstance(image, dict):
                if cls._is_image_dict(image):
                    return {
                        "url": image.get("url"),
                        "path": image.get("path"),
                    }

                nested = cls._extract_image_info(
                    image
                )

                if nested:
                    return nested

            # Campo "output"
            if "output" in value:
                nested = cls._extract_image_info(
                    value.get("output")
                )

                if nested:
                    return nested

            # Outros campos aninhados
            for key in (
                "data",
                "result",
                "results",
                "gallery",
                "images",
            ):
                if key not in value:
                    continue

                nested = cls._extract_image_info(
                    value.get(key)
                )

                if nested:
                    return nested

            return None

        # -----------------------------------------------------
        # Lista
        # -----------------------------------------------------

        if isinstance(value, list):

            for item in value:

                nested = cls._extract_image_info(
                    item
                )

                if nested:
                    return nested

        return None

    @staticmethod
    def _extract_error_message(value) -> str | None:
        """
        Extrai mensagens de erro de diferentes formatos.
        """

        if value is None:
            return None

        if isinstance(value, str):

            text = value.strip()

            if not text:
                return None

            return text

        if isinstance(value, dict):

            for key in (
                "error",
                "message",
                "detail",
                "msg",
            ):
                candidate = value.get(key)

                if candidate is None:
                    continue

                if isinstance(candidate, str):
                    if candidate.strip():
                        return candidate.strip()

                else:
                    try:
                        return json.dumps(
                            candidate,
                            ensure_ascii=False,
                        )
                    except Exception:
                        return str(candidate)

            return None

        if isinstance(value, list):

            for item in value:

                message = (
                    HuggingFaceImageProvider
                    ._extract_error_message(item)
                )

                if message:
                    return message

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
        # PATH do Gradio
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

                encoded_path = quote(
                    image_path,
                    safe="",
                )

                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file={encoded_path}"
                )

                candidates.append(
                    f"{self.space_url}"
                    f"/file={encoded_path}"
                )

                if image_path.startswith("/"):
                    candidates.append(
                        f"{self.space_url}{image_path}"
                    )

        # -----------------------------------------------------
        # Remove duplicados
        # -----------------------------------------------------

        candidates = list(
            dict.fromkeys(candidates)
        )

        for candidate in candidates:

            try:

                response = await client.get(
                    candidate,
                    headers=self._headers(),
                )

                if response.status_code != 200:
                    continue

                content = response.content

                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()

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
                    b"\xff\xd8"
                ):
                    return content

                # -------------------------------------------------
                # WEBP
                # -------------------------------------------------

                if (
                    len(content) >= 12
                    and content[:4] == b"RIFF"
                    and content[8:12] == b"WEBP"
                ):
                    return content

                # -------------------------------------------------
                # MIME informado corretamente
                # -------------------------------------------------

                if content_type.startswith(
                    "image/"
                ):
                    return content

            except (
                httpx.HTTPError,
                OSError,
            ):
                continue

        return None

    # =========================================================
    # TENTATIVA VIA ENDPOINT DIRETO
    # =========================================================

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

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
        # Dimensões
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

            if width > height:

                ratio = (
                    width / max(height, 1)
                )

                if ratio >= 1.6:
                    aspect_ratio = (
                        "Wide 16:9 / Panoramico 16:9"
                    )
                else:
                    aspect_ratio = (
                        "Landscape 5:4 / Horizontal 5:4"
                    )

            elif height > width:

                ratio = (
                    height / max(width, 1)
                )

                if ratio >= 1.6:
                    aspect_ratio = (
                        "Tall 9:16 / Vertical 9:16"
                    )
                else:
                    aspect_ratio = (
                        "Portrait 4:5 / Retrato 4:5"
                    )

            else:
                aspect_ratio = (
                    "Square 1:1 / Cuadrado 1:1"
                )

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

        # -----------------------------------------------------
        # Payload conforme ao OpenAPI fornecido
        # -----------------------------------------------------

        payload = {
            "t2i_prompt": prompt,
            "i2i_prompt": "",
            "i2i_image": None,
            "strength": strength,
            "style_preset": style_preset,
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        try:

            response = await client.post(
                f"{self.space_url}"
                "/run/generate_images",
                json=payload,
                headers=self._headers(),
            )

        except httpx.HTTPError as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "direct_http_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"http_{response.status_code}: "
                    f"{response.text[:1500]}"
                ),
            )

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

        image_info = (
            self._extract_image_info(data)
        )

        if not image_info:

            return ImageResult(
                False,
                self.name,
                error=(
                    "no_image_output: "
                    f"{json.dumps("
                    f"data, "
                    f"ensure_ascii=False"
                    f")[:1500]}"
                ),
            )

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

            return ImageResult(
                False,
                self.name,
                image_url=image_url,
                error=(
                    "image_download_failed: "
                    f"url={image_url!r} "
                    f"path={image_path!r}"
                ),
            )

        return ImageResult(
            True,
            self.name,
            image_url=image_url,
            image_bytes=image_bytes,
        )

    # =========================================================
    # SSE
    # =========================================================

    async def _generate_sse(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        print(
            "[IMAGE] Hugging Face: "
            "starting SSE generation",
            flush=True,
        )

        # -----------------------------------------------------
        # Os valores abaixo seguem o OpenAPI fornecido pelo
        # usuário para o endpoint generate_images.
        # -----------------------------------------------------

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
        # Ajuste de proporção
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

            if width > height:

                ratio = (
                    width / max(height, 1)
                )

                if ratio >= 1.6:
                    aspect_ratio = (
                        "Wide 16:9 / Panoramico 16:9"
                    )
                else:
                    aspect_ratio = (
                        "Landscape 5:4 / Horizontal 5:4"
                    )

            elif height > width:

                ratio = (
                    height / max(width, 1)
                )

                if ratio >= 1.6:
                    aspect_ratio = (
                        "Tall 9:16 / Vertical 9:16"
                    )
                else:
                    aspect_ratio = (
                        "Portrait 4:5 / Retrato 4:5"
                    )

            else:

                aspect_ratio = (
                    "Square 1:1 / Cuadrado 1:1"
                )

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

        # -----------------------------------------------------
        # IMPORTANTE
        #
        # O endpoint SSE do Gradio recebe:
        #
        # {
        #     "data": [...]
        # }
        #
        # Não usamos {"data": {"...": "..."}}
        # -----------------------------------------------------

        payload = {
            "data": [
                prompt,
                "",
                None,
                strength,
                style_preset,
                aspect_ratio,
                steps,
                guidance_scale,
                seed,
            ]
        }

        # -----------------------------------------------------
        # POST inicial
        # -----------------------------------------------------

        try:

            response = await client.post(
                f"{self.space_url}"
                "/gradio_api/call/generate_images",
                json=payload,
                headers=self._headers(),
            )

        except httpx.HTTPError as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"SSE POST failed: {exc}",
                flush=True,
            )

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_post_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        if response.status_code != 200:

            print(
                "[IMAGE] Hugging Face: "
                f"SSE POST HTTP {response.status_code}: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_{response.status_code}: "
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
                    "sse_invalid_initial_response: "
                    f"{response.text[:1000]}"
                ),
            )

        event_id = data.get(
            "event_id"
        )

        if not event_id:

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_no_event_id: "
                    f"{json.dumps("
                    f"data, "
                    f"ensure_ascii=False"
                    f")[:1000]}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event_id={event_id}",
            flush=True,
        )

        # -----------------------------------------------------
        # GET SSE
        # -----------------------------------------------------

        try:

            async with client.stream(
                "GET",
                f"{self.space_url}"
                "/gradio_api/call/"
                f"generate_images/"
                f"{event_id}",
                headers=self._sse_headers(),
            ) as result_response:

                if (
                    result_response.status_code
                    != 200
                ):

                    body = await (
                        result_response.aread()
                    )

                    text = body.decode(
                        "utf-8",
                        errors="replace",
                    )

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "sse_result_http_"
                            f"{result_response.status_code}: "
                            f"{text[:1500]}"
                        ),
                    )

                current_event = None
                current_data_lines = []

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.strip()

                    # -------------------------------------------------
                    # Linha vazia = fim de evento SSE
                    # -------------------------------------------------

                    if not line:

                        if current_data_lines:

                            raw_data = "\n".join(
                                current_data_lines
                            ).strip()

                            event_name = (
                                current_event
                                or "message"
                            )

                            current_event = None
                            current_data_lines = []

                            if not raw_data:
                                continue

                            # -----------------------------------------
                            # Parse do data
                            # -----------------------------------------

                            try:

                                event_data = (
                                    json.loads(
                                        raw_data
                                    )
                                )

                            except json.JSONDecodeError:

                                event_data = (
                                    raw_data
                                )

                            # -----------------------------------------
                            # Evento de erro
                            # -----------------------------------------

                            if event_name == "error":

                                message = (
                                    self
                                    ._extract_error_message(
                                        event_data
                                    )
                                )

                                if not message:
                                    message = (
                                        raw_data
                                    )

                                print(
                                    "[IMAGE] "
                                    "Hugging Face: "
                                    "generation "
                                    "failed: "
                                    f"{message}",
                                    flush=True,
                                )

                                return ImageResult(
                                    False,
                                    self.name,
                                    error=(
                                        "sse_generation_error: "
                                        f"{message}"
                                    ),
                                )

                            # -----------------------------------------
                            # Alguns servidores enviam erro dentro
                            # de um evento "complete".
                            # -----------------------------------------

                            embedded_error = (
                                self
                                ._extract_error_message(
                                    event_data
                                )
                                if isinstance(
                                    event_data,
                                    dict,
                                )
                                else None
                            )

                            if (
                                embedded_error
                                and event_name
                                in (
                                    "error",
                                    "complete",
                                )
                            ):

                                # Só consideramos erro se houver
                                # uma chave explícita de erro.
                                if isinstance(
                                    event_data,
                                    dict,
                                ) and (
                                    event_data.get(
                                        "error"
                                    )
                                    is not None
                                    or event_data.get(
                                        "detail"
                                    ) is not None
                                ):

                                    return ImageResult(
                                        False,
                                        self.name,
                                        error=(
                                            "sse_generation_error: "
                                            f"{embedded_error}"
                                        ),
                                    )

                            # -----------------------------------------
                            # Procura imagem em qualquer formato
                            # -----------------------------------------

                            image_info = (
                                self
                                ._extract_image_info(
                                    event_data
                                )
                            )

                            if image_info:

                                image_url = (
                                    image_info.get(
                                        "url"
                                    )
                                )

                                image_path = (
                                    image_info.get(
                                        "path"
                                    )
                                )

                                image_bytes = (
                                    await self
                                    ._download_image(
                                        client,
                                        image_url,
                                        image_path,
                                    )
                                )

                                if image_bytes:

                                    print(
                                        "[IMAGE] "
                                        "Hugging Face: "
                                        "image received",
                                        flush=True,
                                    )

                                    return ImageResult(
                                        True,
                                        self.name,
                                        image_url=(
                                            image_url
                                        ),
                                        image_bytes=(
                                            image_bytes
                                        ),
                                    )

                        continue

                    # -------------------------------------------------
                    # event:
                    # -------------------------------------------------

                    if line.startswith(
                        "event:"
                    ):

                        current_event = (
                            line[6:].strip()
                        )

                        continue

                    # -------------------------------------------------
                    # data:
                    # -------------------------------------------------

                    if line.startswith(
                        "data:"
                    ):

                        current_data_lines.append(
                            line[5:].lstrip()
                        )

                        continue

                # -----------------------------------------------------
                # Caso o stream termine sem linha vazia final,
                # processamos o último evento.
                # -----------------------------------------------------

                if current_data_lines:

                    raw_data = "\n".join(
                        current_data_lines
                    ).strip()

                    try:

                        event_data = json.loads(
                            raw_data
                        )

                    except json.JSONDecodeError:

                        event_data = raw_data

                    image_info = (
                        self
                        ._extract_image_info(
                            event_data
                        )
                    )

                    if image_info:

                        image_url = (
                            image_info.get(
                                "url"
                            )
                        )

                        image_path = (
                            image_info.get(
                                "path"
                            )
                        )

                        image_bytes = (
                            await self
                            ._download_image(
                                client,
                                image_url,
                                image_path,
                            )
                        )

                        if image_bytes:

                            return ImageResult(
                                True,
                                self.name,
                                image_url=(
                                    image_url
                                ),
                                image_bytes=(
                                    image_bytes
                                ),
                            )

        except httpx.TimeoutException:

            return ImageResult(
                False,
                self.name,
                error="sse_timeout",
            )

        except httpx.HTTPError as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_stream_http_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return ImageResult(
            False,
            self.name,
            error=(
                "no_image_in_sse_response"
            ),
        )

    # =========================================================
    # GENERATE
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

        if (
            not prompt
            or not prompt.strip()
        ):

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
                # O endpoint direto do Space respondeu 405 nos
                # testes anteriores.
                #
                # Portanto usamos primeiro a API de fila/SSE,
                # que é o mecanismo recomendado pelo Gradio.
                # -------------------------------------------------

                result = await self._generate_sse(
                    client,
                    prompt,
                    request,
                )

                if result.success:
                    return result

                # -------------------------------------------------
                # Mantemos uma tentativa direta apenas para casos
                # em que o Space eventualmente volte a aceitar
                # /run/generate_images.
                #
                # Não fazemos isso para erros internos de geração,
                # pois repetir imediatamente pode consumir GPU/cota.
                # -------------------------------------------------

                error_text = (
                    result.error
                    or ""
                )

                if (
                    "sse_http_404"
                    in error_text
                    or "sse_http_405"
                    in error_text
                    or "sse_post_error"
                    in error_text
                ):

                    print(
                        "[IMAGE] Hugging Face: "
                        "SSE endpoint unavailable, "
                        "trying direct endpoint",
                        flush=True,
                    )

                    direct_result = (
                        await self._generate_direct(
                            client,
                            prompt,
                            request,
                        )
                    )

                    if direct_result.success:
                        return direct_result

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"sse={result.error}; "
                            f"direct="
                            f"{direct_result.error}"
                        ),
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
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        except Exception as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "unexpected_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
