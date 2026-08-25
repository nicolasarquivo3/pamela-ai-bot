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

    # =========================================================
    # DISPONIBILIDADE
    # =========================================================

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
    # PARÂMETROS DA IMAGEM
    # =========================================================

    @staticmethod
    def _image_parameters(request=None) -> dict:
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

        # -----------------------------------------------------
        # Dimensões
        # -----------------------------------------------------

        width = getattr(request, "width", 1024)
        height = getattr(request, "height", 1024)

        try:
            width = int(width)
        except (TypeError, ValueError):
            width = 1024

        try:
            height = int(height)
        except (TypeError, ValueError):
            height = 1024

        if width <= 0:
            width = 1024

        if height <= 0:
            height = 1024

        # -----------------------------------------------------
        # Aspect ratio
        # -----------------------------------------------------

        if width > height:
            ratio = width / max(height, 1)

            if ratio >= 1.6:
                aspect_ratio = "Wide 16:9 / Panoramico 16:9"
            else:
                aspect_ratio = (
                    "Landscape 5:4 / Horizontal 5:4"
                )

        elif height > width:
            ratio = height / max(width, 1)

            if ratio >= 1.6:
                aspect_ratio = "Tall 9:16 / Vertical 9:16"
            else:
                aspect_ratio = "Portrait 4:5 / Retrato 4:5"

        else:
            aspect_ratio = "Square 1:1 / Cuadrado 1:1"

        # -----------------------------------------------------
        # Estilo
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
                style_preset = "Editorial / Editorial"

            elif "fantasy" in normalized_style:
                style_preset = "Fantasy / Fantasia"

            else:
                style_preset = (
                    "Photoreal / Fotorrealista"
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
    # EXTRAÇÃO DE IMAGEM
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
        Procura recursivamente uma imagem dentro de uma resposta
        do Gradio.

        Pode encontrar estruturas como:

            {
                "image": {
                    "path": "...",
                    "url": "..."
                }
            }

        ou:

            [
                {
                    "image": {
                        "path": "...",
                        "url": "..."
                    }
                }
            ]

        ou estruturas aninhadas de listas/dicionários.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # Dicionário
        # -----------------------------------------------------

        if isinstance(value, dict):

            # Imagem diretamente no objeto.
            if cls._is_image_dict(value):
                return {
                    "url": value.get("url"),
                    "path": value.get("path"),
                }

            # Campo image.
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

            # Campos comuns do Gradio/API.
            for key in (
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

    # =========================================================
    # EXTRAÇÃO DE ERRO
    # =========================================================

    @staticmethod
    def _extract_error_message(value) -> str | None:
        """
        Extrai uma mensagem de erro sem considerar valores como
        None, null ou strings vazias como erros reais.
        """

        if value is None:
            return None

        if isinstance(value, str):

            text = value.strip()

            if not text:
                return None

            # Alguns Spaces enviam literalmente "None" ou "null".
            if text.lower() in (
                "none",
                "null",
                "undefined",
            ):
                return None

            return text

        if isinstance(value, dict):

            # Primeiro procuramos explicitamente campos de erro.
            for key in (
                "error",
                "exception",
                "message",
                "detail",
                "msg",
            ):
                if key not in value:
                    continue

                candidate = value.get(key)

                if candidate is None:
                    continue

                if isinstance(candidate, str):

                    text = candidate.strip()

                    if not text:
                        continue

                    if text.lower() in (
                        "none",
                        "null",
                        "undefined",
                    ):
                        continue

                    return text

                try:
                    encoded = json.dumps(
                        candidate,
                        ensure_ascii=False,
                    )

                    if encoded.lower() in (
                        "null",
                        '"none"',
                    ):
                        continue

                    return encoded

                except Exception:
                    text = str(candidate).strip()

                    if text:
                        return text

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
    # DETECÇÃO EXPLÍCITA DE ERRO
    # =========================================================

    @staticmethod
    def _has_explicit_error(value) -> bool:
        """
        Retorna True somente quando a estrutura realmente contém
        uma indicação explícita de erro.

        Isso é importante porque alguns servidores Gradio enviam:

            event: error
            data: null

        ou:

            event: error
            data: None

        Isso não deve ser automaticamente tratado como uma
        mensagem de erro válida.
        """

        if not isinstance(value, dict):
            return False

        for key in (
            "error",
            "exception",
        ):
            if key not in value:
                continue

            candidate = value.get(key)

            if candidate is None:
                continue

            if isinstance(candidate, str):
                text = candidate.strip()

                if not text:
                    continue

                if text.lower() in (
                    "none",
                    "null",
                    "undefined",
                ):
                    continue

            return True

        return False

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
                    print(
                        "[IMAGE] Hugging Face: "
                        f"download HTTP "
                        f"{response.status_code}: "
                        f"{candidate}",
                        flush=True,
                    )

                    continue

                content = response.content

                if not content:
                    continue

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
                # MIME de imagem
                # -------------------------------------------------

                if content_type.startswith("image/"):
                    return content

            except (
                httpx.HTTPError,
                OSError,
            ) as exc:

                print(
                    "[IMAGE] Hugging Face: "
                    f"download failed: {exc}",
                    flush=True,
                )

                continue

        return None

    # =========================================================
    # PROCESSAR EVENTO SSE
    # =========================================================

    async def _process_sse_event(
        self,
        client: httpx.AsyncClient,
        event_name: str,
        raw_data: str,
    ) -> ImageResult | None:
        """
        Processa um evento individual do SSE.

        Retorna:
            ImageResult(success=True) quando encontrou imagem.
            ImageResult(success=False) quando encontrou erro real.
            None quando o evento é apenas progresso/status.
        """

        if not raw_data:
            return None

        # -----------------------------------------------------
        # JSON
        # -----------------------------------------------------

        try:
            event_data = json.loads(raw_data)

        except json.JSONDecodeError:
            event_data = raw_data

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event={event_name!r} "
            f"data={raw_data[:2000]!r}",
            flush=True,
        )

        # -----------------------------------------------------
        # Valores nulos não são erro.
        # -----------------------------------------------------

        if event_data is None:
            return None

        if isinstance(event_data, str):

            normalized = event_data.strip().lower()

            if normalized in (
                "",
                "none",
                "null",
                "undefined",
            ):
                return None

        # -----------------------------------------------------
        # Erro explícito dentro do JSON.
        # -----------------------------------------------------

        if (
            isinstance(event_data, dict)
            and self._has_explicit_error(event_data)
        ):

            message = (
                self._extract_error_message(
                    event_data
                )
            )

            if message:

                print(
                    "[IMAGE] Hugging Face: "
                    "generation failed: "
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

        # -----------------------------------------------------
        # Evento explicitamente chamado error.
        #
        # Não tratamos data=null como erro.
        # -----------------------------------------------------

        if event_name == "error":

            message = (
                self._extract_error_message(
                    event_data
                )
            )

            if message:

                print(
                    "[IMAGE] Hugging Face: "
                    "SSE error event: "
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

            # error + null/None:
            # não inventar uma mensagem.
            print(
                "[IMAGE] Hugging Face: "
                "SSE error event contained "
                "no error message",
                flush=True,
            )

            return None

        # -----------------------------------------------------
        # Procurar imagem.
        # -----------------------------------------------------

        image_info = (
            self._extract_image_info(
                event_data
            )
        )

        if not image_info:
            return None

        image_url = image_info.get("url")
        image_path = image_info.get("path")

        image_bytes = await self._download_image(
            client,
            image_url,
            image_path,
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

        print(
            "[IMAGE] Hugging Face: "
            "image received successfully",
            flush=True,
        )

        return ImageResult(
            True,
            self.name,
            image_url=image_url,
            image_bytes=image_bytes,
        )

    # =========================================================
    # ENDPOINT DIRETO
    # =========================================================

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        params = self._image_parameters(request)

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
            "trying direct generation",
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
            self._extract_image_info(data)
        )

        if not image_info:

            try:
                debug_data = json.dumps(
                    data,
                    ensure_ascii=False,
                )
            except Exception:
                debug_data = str(data)

            return ImageResult(
                False,
                self.name,
                error=(
                    "no_image_output: "
                    f"{debug_data[:1500]}"
                ),
            )

        image_url = image_info.get("url")
        image_path = image_info.get("path")

        image_bytes = await self._download_image(
            client,
            image_url,
            image_path,
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

        params = self._image_parameters(request)

        # -----------------------------------------------------
        # Gradio queue:
        #
        # POST:
        # /gradio_api/call/generate_images
        #
        # Body:
        # {
        #     "data": [...]
        # }
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
                f"SSE POST HTTP "
                f"{response.status_code}: "
                f"{response.text[:1500]}",
                flush=True,
            )

            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_{response.status_code}: "
                    f"{response.text[:1500]}"
                ),
            )

        # -----------------------------------------------------
        # Resposta inicial
        # -----------------------------------------------------

        try:

            initial_data = response.json()

        except json.JSONDecodeError:

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_invalid_initial_response: "
                    f"{response.text[:1500]}"
                ),
            )

        event_id = initial_data.get("event_id")

        if not event_id:

            try:
                debug_data = json.dumps(
                    initial_data,
                    ensure_ascii=False,
                )
            except Exception:
                debug_data = str(initial_data)

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_no_event_id: "
                    f"{debug_data[:1500]}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event_id={event_id}",
            flush=True,
        )

        # =====================================================
        # STREAM SSE
        # =====================================================

        sse_url = (
            f"{self.space_url}"
            "/gradio_api/call/"
            f"generate_images/"
            f"{event_id}"
        )

        try:

            async with client.stream(
                "GET",
                sse_url,
                headers=self._sse_headers(),
            ) as result_response:

                if result_response.status_code != 200:

                    body = await (
                        result_response.aread()
                    )

                    text = body.decode(
                        "utf-8",
                        errors="replace",
                    )

                    print(
                        "[IMAGE] Hugging Face: "
                        f"SSE result HTTP "
                        f"{result_response.status_code}: "
                        f"{text[:1500]}",
                        flush=True,
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

                current_event = "message"
                current_data_lines: list[str] = []

                # -------------------------------------------------
                # Ler o SSE linha por linha.
                # -------------------------------------------------

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.rstrip("\r")

                    # -------------------------------------------------
                    # Fim do evento SSE
                    # -------------------------------------------------

                    if line.strip() == "":

                        if not current_data_lines:
                            continue

                        raw_data = "\n".join(
                            current_data_lines
                        ).strip()

                        event_name = (
                            current_event
                            or "message"
                        )

                        current_event = "message"
                        current_data_lines = []

                        result = (
                            await self
                            ._process_sse_event(
                                client,
                                event_name,
                                raw_data,
                            )
                        )

                        if result is not None:

                            # -------------------------------------------------
                            # Sucesso
                            # -------------------------------------------------

                            if result.success:
                                return result

                            # -------------------------------------------------
                            # Erro real
                            # -------------------------------------------------

                            if result.error:
                                return result

                        continue

                    # -------------------------------------------------
                    # event:
                    # -------------------------------------------------

                    if line.startswith("event:"):

                        current_event = (
                            line[len("event:"):].strip()
                        )

                        continue

                    # -------------------------------------------------
                    # data:
                    # -------------------------------------------------

                    if line.startswith("data:"):

                        data_part = (
                            line[len("data:"):]
                            .lstrip()
                        )

                        current_data_lines.append(
                            data_part
                        )

                        continue

                # =====================================================
                # O stream terminou.
                #
                # Alguns servidores não enviam a linha vazia final.
                # =====================================================

                if current_data_lines:

                    raw_data = "\n".join(
                        current_data_lines
                    ).strip()

                    event_name = (
                        current_event
                        or "message"
                    )

                    result = (
                        await self
                        ._process_sse_event(
                            client,
                            event_name,
                            raw_data,
                        )
                    )

                    if result is not None:
                        return result

        except httpx.TimeoutException:

            print(
                "[IMAGE] Hugging Face: "
                "SSE stream timeout",
                flush=True,
            )

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
                # PRIMEIRA TENTATIVA:
                #
                # O mecanismo de fila/SSE do Gradio.
                # -------------------------------------------------

                result = await self._generate_sse(
                    client,
                    prompt,
                    request,
                )

                if result.success:
                    return result

                # -------------------------------------------------
                # Se a geração chegou ao Space e ele respondeu
                # com um erro interno de geração, NÃO repetimos
                # automaticamente a chamada.
                #
                # Isso evita gastar GPU/cota duas vezes.
                # -------------------------------------------------

                error_text = (
                    result.error or ""
                )

                # -------------------------------------------------
                # Fallback somente quando o endpoint SSE não
                # estiver disponível.
                # -------------------------------------------------

                if (
                    "sse_http_404" in error_text
                    or "sse_http_405" in error_text
                    or "sse_post_error" in error_text
                ):

                    print(
                        "[IMAGE] Hugging Face: "
                        "SSE endpoint unavailable; "
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
