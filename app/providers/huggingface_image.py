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
        headers = {
            "Accept": "application/json",
            "User-Agent": "pamela-ai/1.0",
            "x-gradio-user": "api",
        }

        if self.hf_token:
            token = f"Bearer {self.hf_token}"

            headers["Authorization"] = token
            headers["x-hf-authorization"] = token

        return headers

    def _sse_headers(self) -> dict[str, str]:
        headers = self._headers()

        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"

        return headers

    # =========================================================
    # GENERATION PARAMETERS
    # =========================================================

    @staticmethod
    def _generation_parameters(request=None) -> dict:
        style_preset = "Photoreal / Fotorrealista"
        aspect_ratio = "Portrait 4:5 / Retrato 4:5"

        steps = 28
        guidance_scale = 4.0
        seed = -1
        strength = 0.0

        if request is None:
            return {
                "style_preset": style_preset,
                "aspect_ratio": aspect_ratio,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "strength": strength,
            }

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
        except (TypeError, ValueError):
            width = 1024

        try:
            height = int(height)
        except (TypeError, ValueError):
            height = 1024

        width = max(width, 1)
        height = max(height, 1)

        # -----------------------------------------------------
        # Aspect ratio
        # -----------------------------------------------------

        if width > height:
            ratio = width / height

            if ratio >= 1.6:
                aspect_ratio = (
                    "Wide 16:9 / Panoramico 16:9"
                )
            else:
                aspect_ratio = (
                    "Landscape 5:4 / Horizontal 5:4"
                )

        elif height > width:
            ratio = height / width

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

        # -----------------------------------------------------
        # Style
        # -----------------------------------------------------

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

        return {
            "style_preset": style_preset,
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "strength": strength,
        }

    # =========================================================
    # RESPONSE PARSING
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
        Procura recursivamente informações de imagem.

        Aceita formatos como:

            {
                "image": {
                    "url": "...",
                    "path": "..."
                }
            }

        ou:

            [
                {
                    "image": {
                        "url": "...",
                        "path": "..."
                    }
                }
            ]

        ou estruturas aninhadas retornadas por GalleryData.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(value, dict):

            if cls._is_image_dict(value):
                return {
                    "url": value.get("url"),
                    "path": value.get("path"),
                }

            for key in (
                "image",
                "output",
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
        # List
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

            if text:
                return text

            return None

        if isinstance(value, dict):

            for key in (
                "error",
                "message",
                "detail",
                "msg",
                "exception",
                "code",
            ):
                if key not in value:
                    continue

                candidate = value.get(key)

                if candidate is None:
                    continue

                if isinstance(candidate, str):

                    text = candidate.strip()

                    if text:
                        return text

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
                    ._extract_error_message(
                        item
                    )
                )

                if message:
                    return message

        return None

    @staticmethod
    def _safe_json(
        value,
        limit: int = 1500,
    ) -> str:
        """
        Serializa dados para log/erro sem correr risco
        de gerar SyntaxError ou quebrar a resposta.
        """

        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
            )

        except Exception:
            text = repr(value)

        return text[:limit]

    # =========================================================
    # DOWNLOAD IMAGE
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
                candidates.append(
                    image_url
                )

            elif image_url.startswith("/"):
                candidates.append(
                    f"{self.space_url}{image_url}"
                )

        # -----------------------------------------------------
        # PATH retornado pelo Gradio
        # -----------------------------------------------------

        if image_path:

            if image_path.startswith(
                (
                    "http://",
                    "https://",
                )
            ):
                candidates.append(
                    image_path
                )

            else:

                normalized_path = (
                    image_path.lstrip("/")
                )

                encoded_path = quote(
                    normalized_path,
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

                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file=/{encoded_path}"
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

        # -----------------------------------------------------
        # Tenta cada URL
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

                content_type = (
                    response.headers.get(
                        "content-type",
                        "",
                    )
                    .lower()
                )

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
                # MIME
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
    # PROCESS SSE EVENT
    # =========================================================

    async def _process_sse_event(
        self,
        client: httpx.AsyncClient,
        event_name: str,
        raw_data: str,
    ) -> tuple[str, ImageResult | None]:

        event_name = (
            event_name
            or "message"
        )

        raw_data = raw_data.strip()

        if not raw_data:
            return (
                "continue",
                None,
            )

        # -----------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------

        try:

            event_data = json.loads(
                raw_data
            )

        except json.JSONDecodeError:

            event_data = raw_data

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event={event_name!r} "
            f"data={self._safe_json(event_data, 1000)}",
            flush=True,
        )

        # =====================================================
        # EVENT ERROR
        # =====================================================

        if event_name == "error":

            message = (
                self._extract_error_message(
                    event_data
                )
            )

            # -------------------------------------------------
            # O problema que apareceu no log atual:
            #
            # event='error'
            # data='null'
            #
            # Nesse caso o Gradio não forneceu a exceção.
            # Não podemos tratar None como uma mensagem.
            # -------------------------------------------------

            if not message:

                if raw_data.lower() == "null":

                    message = (
                        "Hugging Face/Gradio returned "
                        "an SSE error event with data=null"
                    )

                else:

                    message = raw_data

            return (
                "error",
                ImageResult(
                    False,
                    self.name,
                    error=(
                        "sse_generation_error: "
                        f"{message}"
                    ),
                ),
            )

        # =====================================================
        # ERROR EMBUTIDO
        # =====================================================

        if isinstance(
            event_data,
            dict,
        ):

            explicit_error = any(
                event_data.get(key) is not None
                for key in (
                    "error",
                    "exception",
                    "detail",
                )
            )

            if explicit_error:

                message = (
                    self._extract_error_message(
                        event_data
                    )
                )

                return (
                    "error",
                    ImageResult(
                        False,
                        self.name,
                        error=(
                            "sse_generation_error: "
                            f"{message or self._safe_json(event_data)}"
                        ),
                    ),
                )

        # =====================================================
        # IMAGE
        # =====================================================

        image_info = (
            self._extract_image_info(
                event_data
            )
        )

        if not image_info:

            return (
                "continue",
                None,
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

        if image_bytes:

            return (
                "success",
                ImageResult(
                    True,
                    self.name,
                    image_url=image_url,
                    image_bytes=image_bytes,
                ),
            )

        return (
            "error",
            ImageResult(
                False,
                self.name,
                image_url=image_url,
                error=(
                    "image_download_failed: "
                    f"url={image_url!r} "
                    f"path={image_path!r}"
                ),
            ),
        )

    # =========================================================
    # DIRECT REST ENDPOINT
    # =========================================================

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        params = (
            self._generation_parameters(
                request
            )
        )

        payload = {
            "t2i_prompt": prompt,
            "i2i_prompt": "",
            "i2i_image": None,
            "strength": params["strength"],
            "style_preset": params["style_preset"],
            "aspect_ratio": params["aspect_ratio"],
            "steps": params["steps"],
            "guidance_scale": params["guidance_scale"],
            "seed": params["seed"],
        }

        print(
            "[IMAGE] Hugging Face: "
            "trying direct /run/generate_images",
            flush=True,
        )

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
            self._extract_image_info(
                data
            )
        )

        if not image_info:

            return ImageResult(
                False,
                self.name,
                error=(
                    "no_image_output: "
                    f"{self._safe_json(data)}"
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
    # SSE / GRADIO QUEUE
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

        params = (
            self._generation_parameters(
                request
            )
        )

        print(
            "[IMAGE] Hugging Face: "
            f"style={params['style_preset']!r} "
            f"aspect={params['aspect_ratio']!r} "
            f"steps={params['steps']} "
            f"guidance={params['guidance_scale']} "
            f"seed={params['seed']}",
            flush=True,
        )

        # -----------------------------------------------------
        # Gradio queue API
        # -----------------------------------------------------

        payload = {
            "data": [
                prompt,
                "",
                None,
                params["strength"],
                params["style_preset"],
                params["aspect_ratio"],
                params["steps"],
                params["guidance_scale"],
                params["seed"],
            ]
        }

        # =====================================================
        # POST INICIAL
        # =====================================================

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
                f"SSE POST HTTP "
                f"{response.status_code}: "
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

        # =====================================================
        # EVENT ID
        # =====================================================

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
                    f"{self._safe_json(initial_data, 1000)}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event_id={event_id}",
            flush=True,
        )

        # =====================================================
        # SSE STREAM
        # =====================================================

        current_event = None
        current_data_lines: list[str] = []

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

                # -------------------------------------------------
                # Ler SSE linha por linha
                # -------------------------------------------------

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.rstrip(
                        "\r"
                    )

                    # -------------------------------------------------
                    # Evento terminou
                    # -------------------------------------------------

                    if line == "":

                        if not current_data_lines:

                            current_event = None

                            continue

                        raw_data = "\n".join(
                            current_data_lines
                        ).strip()

                        event_name = (
                            current_event
                            or "message"
                        )

                        current_event = None
                        current_data_lines = []

                        status, result = (
                            await self
                            ._process_sse_event(
                                client,
                                event_name,
                                raw_data,
                            )
                        )

                        if (
                            status == "success"
                            and result
                        ):
                            return result

                        if (
                            status == "error"
                            and result
                        ):
                            return result

                        continue

                    # -------------------------------------------------
                    # Comentário SSE
                    # -------------------------------------------------

                    if line.startswith(":"):
                        continue

                    # -------------------------------------------------
                    # event:
                    # -------------------------------------------------

                    if line.startswith(
                        "event:"
                    ):

                        current_event = (
                            line[
                                len("event:"):
                            ].strip()
                        )

                        continue

                    # -------------------------------------------------
                    # data:
                    # -------------------------------------------------

                    if line.startswith(
                        "data:"
                    ):

                        current_data_lines.append(
                            line[
                                len("data:"):
                            ].lstrip()
                        )

                        continue

                # -----------------------------------------------------
                # Alguns servidores encerram o stream sem mandar
                # uma linha vazia depois do último evento.
                # -----------------------------------------------------

                if current_data_lines:

                    raw_data = "\n".join(
                        current_data_lines
                    ).strip()

                    event_name = (
                        current_event
                        or "message"
                    )

                    status, result = (
                        await self
                        ._process_sse_event(
                            client,
                            event_name,
                            raw_data,
                        )
                    )

                    if (
                        status in (
                            "success",
                            "error",
                        )
                        and result
                    ):
                        return result

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

        # -----------------------------------------------------
        # Configuração
        # -----------------------------------------------------

        if not await self.available():

            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

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

                # =================================================
                # PRIMEIRA TENTATIVA
                #
                # Gradio Queue / SSE
                # =================================================

                result = (
                    await self._generate_sse(
                        client,
                        prompt,
                        request,
                    )
                )

                if result.success:
                    return result

                error_text = (
                    result.error
                    or ""
                )

                # =================================================
                # FALLBACK
                #
                # O log atual mostrou:
                #
                # SSE event='error' data='null'
                #
                # Nesse cenário o Space recusou/falhou durante
                # a execução. Tentamos a rota REST direta antes
                # de desistir.
                # =================================================

                if (
                    "sse_generation_error"
                    in error_text
                    or "no_image_in_sse_response"
                    in error_text
                ):

                    print(
                        "[IMAGE] Hugging Face: "
                        "SSE failed; "
                        "trying direct endpoint",
                        flush=True,
                    )

                    direct_result = (
                        await self
                        ._generate_direct(
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

                # =================================================
                # SSE INDISPONÍVEL
                # =================================================

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
                        "SSE endpoint unavailable; "
                        "trying direct endpoint",
                        flush=True,
                    )

                    direct_result = (
                        await self
                        ._generate_direct(
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

        # =====================================================
        # TIMEOUT
        # =====================================================

        except httpx.TimeoutException:

            return ImageResult(
                False,
                self.name,
                error="timeout",
            )

        # =====================================================
        # HTTP ERROR
        # =====================================================

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

        # =====================================================
        # QUALQUER OUTRO ERRO
        # =====================================================

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
