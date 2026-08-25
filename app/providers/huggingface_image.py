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
    # CONFIGURAÇÃO
    # =========================================================

    async def available(self) -> bool:
        return bool(self.space_url)

    def _headers(self) -> dict:
        headers = {
            "Accept": "application/json",
        }

        if self.hf_token:
            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        return headers

    # =========================================================
    # PARÂMETROS DO SPACE
    # =========================================================

    @staticmethod
    def _build_generation_parameters(
        request=None,
    ) -> dict:
        """
        Constrói os parâmetros compatíveis com o OpenAPI
        do Space FLUX.2 Klein 4B fornecido pelo usuário.

        Ordem dos parâmetros do Space:

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
        # DIMENSÕES
        # -----------------------------------------------------

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

        if width <= 0:
            width = 1024

        if height <= 0:
            height = 1024

        ratio = width / max(
            height,
            1,
        )

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

        # -----------------------------------------------------
        # ESTILO
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

            else:
                style_preset = (
                    "Photoreal / Fotorrealista"
                )

        # -----------------------------------------------------
        # STEPS
        # -----------------------------------------------------

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
            except (TypeError, ValueError):
                steps = 28

        steps = max(
            1,
            min(
                50,
                steps,
            ),
        )

        # -----------------------------------------------------
        # GUIDANCE SCALE
        # -----------------------------------------------------

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
            except (TypeError, ValueError):
                guidance_scale = 4.0

        guidance_scale = max(
            1.0,
            min(
                10.0,
                guidance_scale,
            ),
        )

        # -----------------------------------------------------
        # SEED
        # -----------------------------------------------------

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
            except (TypeError, ValueError):
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
    # EXTRAÇÃO DE IMAGEM
    # =========================================================

    @classmethod
    def _extract_image_info(
        cls,
        data,
    ):
        """
        Procura recursivamente uma imagem dentro das respostas
        do Gradio.

        Pode receber:

        - GalleryData
        - GalleryImage
        - ImageData
        - listas
        - dicionários aninhados
        - strings contendo URLs
        """

        if data is None:
            return None

        # -----------------------------------------------------
        # LISTA
        # -----------------------------------------------------

        if isinstance(data, list):
            for item in data:
                result = cls._extract_image_info(
                    item
                )

                if result:
                    return result

            return None

        # -----------------------------------------------------
        # STRING
        # -----------------------------------------------------

        if isinstance(data, str):
            value = data.strip()

            if value.startswith(
                (
                    "http://",
                    "https://",
                )
            ):
                return {
                    "url": value,
                    "path": None,
                }

            if value.startswith("/"):
                return {
                    "url": None,
                    "path": value,
                }

            return None

        # -----------------------------------------------------
        # OUTROS TIPOS
        # -----------------------------------------------------

        if not isinstance(data, dict):
            return None

        # -----------------------------------------------------
        # ImageData direto
        # -----------------------------------------------------

        image_url = data.get("url")
        image_path = data.get("path")

        if image_url or image_path:
            return {
                "url": image_url,
                "path": image_path,
            }

        # -----------------------------------------------------
        # GalleryImage
        # -----------------------------------------------------

        image = data.get("image")

        if isinstance(image, dict):
            image_url = image.get("url")
            image_path = image.get("path")

            if image_url or image_path:
                return {
                    "url": image_url,
                    "path": image_path,
                }

        # -----------------------------------------------------
        # Campos conhecidos do Gradio
        # -----------------------------------------------------

        for key in (
            "output",
            "outputs",
            "result",
            "results",
            "gallery",
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
        """
        Faz o download da imagem retornada pelo Gradio.
        """

        candidates: list[str] = []

        # -----------------------------------------------------
        # URL DIRETA
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
                    f"{self.space_url}"
                    f"{image_url}"
                )

        # -----------------------------------------------------
        # PATH DO GRADIO
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
                clean_path = image_path

                if clean_path.startswith("/"):
                    clean_path = clean_path[1:]

                encoded_path = quote(
                    clean_path,
                    safe="",
                )

                # Gradio moderno
                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file="
                    f"{encoded_path}"
                )

                # Compatibilidade com versões antigas
                candidates.append(
                    f"{self.space_url}"
                    f"/file="
                    f"{encoded_path}"
                )

                # Caso o path já seja uma rota pública
                if image_path.startswith("/"):
                    candidates.append(
                        f"{self.space_url}"
                        f"{image_path}"
                    )

        candidates = list(
            dict.fromkeys(candidates)
        )

        if not candidates:
            return None

        # -----------------------------------------------------
        # TENTAR CADA ENDPOINT
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

                # MIME de imagem
                if content_type.startswith(
                    "image/"
                ):
                    return content

                # PNG
                if content.startswith(
                    b"\x89PNG\r\n\x1a\n"
                ):
                    return content

                # JPEG
                if content.startswith(
                    b"\xff\xd8\xff"
                ):
                    return content

                # WEBP
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
    # DECODIFICAR DATA DO SSE
    # =========================================================

    @staticmethod
    def _decode_sse_data(raw_data):
        """
        O Gradio pode enviar o campo data como:

            JSON
            string contendo JSON
            lista
            objeto
            null

        Esta função tenta normalizar esses formatos.
        """

        if raw_data is None:
            return None

        if not isinstance(
            raw_data,
            str,
        ):
            return raw_data

        value = raw_data.strip()

        if not value:
            return None

        # -----------------------------------------------------
        # null
        # -----------------------------------------------------

        if value.lower() == "null":
            return None

        # -----------------------------------------------------
        # JSON
        # -----------------------------------------------------

        try:
            decoded = json.loads(
                value
            )
        except json.JSONDecodeError:
            return value

        # -----------------------------------------------------
        # Algumas versões podem encapsular JSON como string.
        #
        # Exemplo:
        #
        # "\"[{...}]\""
        # -----------------------------------------------------

        if isinstance(
            decoded,
            str,
        ):
            nested = decoded.strip()

            if nested:
                try:
                    return json.loads(
                        nested
                    )
                except json.JSONDecodeError:
                    return decoded

        return decoded

    # =========================================================
    # ERRO DO EVENTO SSE
    # =========================================================

    @staticmethod
    def _format_sse_error(
        event_data,
    ) -> str:
        """
        Converte o conteúdo do evento de erro em uma mensagem
        útil.

        Importante:
        {"error": null} NÃO é considerado erro.
        """

        if event_data is None:
            return (
                "sse_generation_error"
            )

        if isinstance(
            event_data,
            dict,
        ):
            error_value = event_data.get(
                "error"
            )

            if error_value is not None:
                return (
                    "sse_generation_error: "
                    f"{error_value}"
                )

            message = event_data.get(
                "message"
            )

            if message:
                return (
                    "sse_generation_error: "
                    f"{message}"
                )

        try:
            serialized = json.dumps(
                event_data,
                ensure_ascii=False,
            )
        except Exception:
            serialized = str(
                event_data
            )

        return (
            "sse_generation_error: "
            f"{serialized[:1500]}"
        )

    # =========================================================
    # GERAÇÃO VIA GRADIO SSE
    # =========================================================

    async def _generate_sse(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:
        """
        Gera a imagem usando a API SSE do Gradio.

        Endpoint:

            POST /gradio_api/call/generate_images

        seguido de:

            GET /gradio_api/call/generate_images/{event_id}
        """

        parameters = (
            self._build_generation_parameters(
                request
            )
        )

        # =====================================================
        # ORDEM EXATA DEFINIDA NO OPENAPI 6.3.0
        #
        # 1. t2i_prompt
        # 2. i2i_prompt
        # 3. i2i_image
        # 4. strength
        # 5. style_preset
        # 6. aspect_ratio
        # 7. steps
        # 8. guidance_scale
        # 9. seed
        # =====================================================

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

        # -----------------------------------------------------
        # POST INICIAL
        # -----------------------------------------------------

        try:
            response = await client.post(
                call_url,
                json=payload,
                headers={
                    **self._headers(),
                    "Content-Type": (
                        "application/json"
                    ),
                },
            )

        except httpx.HTTPError as exc:
            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_call_http_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

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
        # RESPOSTA INICIAL
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

        # =====================================================
        # GET DO STREAM SSE
        # =====================================================

        result_url = (
            f"{self.space_url}"
            "/gradio_api/call/generate_images/"
            f"{event_id}"
        )

        stream_headers = {
            "Accept": "text/event-stream",
        }

        if self.hf_token:
            stream_headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        # -----------------------------------------------------
        # Ler o stream
        # -----------------------------------------------------

        try:
            async with client.stream(
                "GET",
                result_url,
                headers=stream_headers,
            ) as result_response:

                if result_response.status_code != 200:

                    body = await result_response.aread()

                    try:
                        body_text = body.decode(
                            "utf-8",
                            errors="replace",
                        )
                    except Exception:
                        body_text = str(
                            body
                        )

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"sse_result_http_"
                            f"{result_response.status_code}: "
                            f"{body_text[:1500]}"
                        ),
                    )

                current_event = None
                last_data = None

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    if raw_line is None:
                        continue

                    line = raw_line.strip()

                    if not line:
                        continue

                    # =========================================
                    # EVENT
                    # =========================================

                    if line.startswith(
                        "event:"
                    ):
                        current_event = (
                            line[
                                len("event:"):
                            ].strip()
                        )

                        continue

                    # =========================================
                    # DATA
                    # =========================================

                    if not line.startswith(
                        "data:"
                    ):
                        continue

                    raw_data = (
                        line[
                            len("data:"):
                        ].strip()
                    )

                    event_data = (
                        self._decode_sse_data(
                            raw_data
                        )
                    )

                    last_data = event_data

                    # =========================================
                    # EVENTO DE ERRO
                    # =========================================
                    #
                    # IMPORTANTE:
                    #
                    # {"error": null}
                    #
                    # NÃO é erro.
                    #
                    # Só tratamos como erro quando o próprio
                    # evento é "error" e existe informação
                    # real de erro.
                    # =========================================

                    if current_event == "error":

                        if event_data is None:
                            return ImageResult(
                                False,
                                self.name,
                                error=(
                                    "sse_generation_error"
                                ),
                            )

                        if isinstance(
                            event_data,
                            dict,
                        ):
                            error_value = (
                                event_data.get(
                                    "error"
                                )
                            )

                            if error_value is None:
                                # Algumas implementações
                                # podem emitir error:null.
                                # Não interromper a geração.
                                continue

                        return ImageResult(
                            False,
                            self.name,
                            error=(
                                self._format_sse_error(
                                    event_data
                                )
                            ),
                        )

                    # =========================================
                    # ERRO EMBUTIDO EM EVENTO NORMAL
                    # =========================================

                    if isinstance(
                        event_data,
                        dict,
                    ):

                        error_value = (
                            event_data.get(
                                "error"
                            )
                        )

                        # Somente erro real.
                        if error_value is not None:

                            return ImageResult(
                                False,
                                self.name,
                                error=(
                                    self._format_sse_error(
                                        event_data
                                    )
                                ),
                            )

                    # =========================================
                    # PROCURAR IMAGEM
                    # =========================================

                    image_info = (
                        self._extract_image_info(
                            event_data
                        )
                    )

                    if not image_info:
                        continue

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

                    # =========================================
                    # DOWNLOAD
                    # =========================================

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

        # =====================================================
        # STREAM TERMINOU SEM IMAGEM
        # =====================================================

        debug_data = ""

        if last_data is not None:
            try:
                debug_data = json.dumps(
                    last_data,
                    ensure_ascii=False,
                )[:2000]
            except Exception:
                debug_data = str(
                    last_data
                )[:2000]

        return ImageResult(
            False,
            self.name,
            error=(
                "no_image_in_sse_response: "
                f"last_event={current_event!r}; "
                f"last_data={debug_data}"
            ),
        )

    # =========================================================
    # GERAÇÃO PRINCIPAL
    # =========================================================

    async def generate(
        self,
        request,
        prompt: str,
    ) -> ImageResult:
        """
        Método chamado pelo ImageService.
        """

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

                print(
                    "[IMAGE] Hugging Face: "
                    "starting SSE generation",
                    flush=True,
                )

                result = await self._generate_sse(
                    client,
                    prompt,
                    request,
                )

                if result.success:

                    print(
                        "[IMAGE] Hugging Face: "
                        "generation successful",
                        flush=True,
                    )

                else:

                    print(
                        "[IMAGE] Hugging Face: "
                        f"generation failed: "
                        f"{result.error}",
                        flush=True,
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
