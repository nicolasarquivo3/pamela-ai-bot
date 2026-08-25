import json
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider específico para:

        Xurxowsky/flux2-klein-4b-playground

    Space:
        https://huggingface.co/spaces/Xurxowsky/flux2-klein-4b-playground

    A função Gradio do Space é:

        generate_images(
            mode,
            t2i_prompt,
            i2i_prompt,
            i2i_image,
            strength,
            steps,
            guidance_scale,
            seed,
        )

    IMPORTANTE:
    Este Space NÃO recebe style_preset nem aspect_ratio
    como parâmetros da API.
    """

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
    # AVAILABILITY
    # =========================================================

    async def available(self) -> bool:
        return bool(self.space_url)

    # =========================================================
    # HEADERS
    # =========================================================

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "pamela-ai/1.0",
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
        """
        O Space atualmente aceita somente:

            strength
            steps
            guidance_scale
            seed

        style_preset e aspect_ratio NÃO são enviados porque
        não fazem parte da assinatura de generate_images().
        """

        steps = 4
        guidance_scale = 4.0
        seed = 42
        strength = 0.8

        if request is None:
            return {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "strength": strength,
            }

        # -----------------------------------------------------
        # STEPS
        # -----------------------------------------------------

        requested_steps = getattr(
            request,
            "steps",
            steps,
        )

        try:
            steps = int(requested_steps)
        except (TypeError, ValueError):
            steps = 4

        # O próprio Space limita MAX_STEPS = 50.
        steps = max(1, min(steps, 50))

        # -----------------------------------------------------
        # GUIDANCE
        # -----------------------------------------------------

        requested_guidance = getattr(
            request,
            "guidance_scale",
            guidance_scale,
        )

        try:
            guidance_scale = float(
                requested_guidance
            )
        except (TypeError, ValueError):
            guidance_scale = 4.0

        # -----------------------------------------------------
        # SEED
        # -----------------------------------------------------

        requested_seed = getattr(
            request,
            "seed",
            seed,
        )

        try:
            seed = int(requested_seed)
        except (TypeError, ValueError):
            seed = 42

        # -----------------------------------------------------
        # STRENGTH
        # -----------------------------------------------------

        requested_strength = getattr(
            request,
            "strength",
            strength,
        )

        try:
            strength = float(
                requested_strength
            )
        except (TypeError, ValueError):
            strength = 0.8

        strength = max(
            0.0,
            min(strength, 1.0),
        )

        return {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "strength": strength,
        }

    # =========================================================
    # SAFE JSON
    # =========================================================

    @staticmethod
    def _safe_json(
        value,
        limit: int = 2000,
    ) -> str:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
            )

        except Exception:
            text = repr(value)

        return text[:limit]

    # =========================================================
    # ERROR EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_error_message(
        value,
    ) -> str | None:

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

                if isinstance(
                    candidate,
                    str,
                ):
                    text = candidate.strip()

                    if text:
                        return text

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

    # =========================================================
    # IMAGE DETECTION
    # =========================================================

    @staticmethod
    def _is_image_dict(value) -> bool:

        if not isinstance(
            value,
            dict,
        ):
            return False

        return bool(
            value.get("url")
            or value.get("path")
        )

    @classmethod
    def _extract_image_info(
        cls,
        value,
    ):
        """
        Extrai recursivamente uma imagem das estruturas
        retornadas pelo Gradio.

        Exemplos possíveis:

            {
                "path": "...",
                "url": "..."
            }

        ou:

            {
                "image": {
                    "path": "..."
                }
            }

        ou:

            {
                "data": [
                    {
                        "path": "..."
                    }
                ]
            }

        ou GalleryData/listas aninhadas.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # STRING
        # -----------------------------------------------------

        if isinstance(
            value,
            str,
        ):

            text = value.strip()

            if (
                text.startswith(
                    "http://"
                )
                or text.startswith(
                    "https://"
                )
            ):
                return {
                    "url": text,
                    "path": None,
                }

            return None

        # -----------------------------------------------------
        # DICT
        # -----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            if cls._is_image_dict(value):

                return {
                    "url": value.get(
                        "url"
                    ),
                    "path": value.get(
                        "path"
                    ),
                }

            for key in (
                "image",
                "images",
                "output",
                "outputs",
                "data",
                "result",
                "results",
                "gallery",
            ):

                if key not in value:
                    continue

                nested = (
                    cls._extract_image_info(
                        value.get(key)
                    )
                )

                if nested:
                    return nested

            return None

        # -----------------------------------------------------
        # LIST
        # -----------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            for item in value:

                nested = (
                    cls._extract_image_info(
                        item
                    )
                )

                if nested:
                    return nested

        return None

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
        # URL
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

            elif image_url.startswith(
                "/"
            ):
                candidates.append(
                    f"{self.space_url}"
                    f"{image_url}"
                )

        # -----------------------------------------------------
        # PATH
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

                # Gradio moderno
                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file="
                    f"{encoded_path}"
                )

                # Compatibilidade
                candidates.append(
                    f"{self.space_url}"
                    f"/file="
                    f"{encoded_path}"
                )

                # Algumas versões esperam /
                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file=/"
                    f"{encoded_path}"
                )

                if image_path.startswith(
                    "/"
                ):
                    candidates.append(
                        f"{self.space_url}"
                        f"{image_path}"
                    )

        # -----------------------------------------------------
        # REMOVE DUPLICADOS
        # -----------------------------------------------------

        candidates = list(
            dict.fromkeys(
                candidates
            )
        )

        print(
            "[IMAGE] Hugging Face: "
            f"download candidates="
            f"{candidates}",
            flush=True,
        )

        # -----------------------------------------------------
        # TENTA DOWNLOAD
        # -----------------------------------------------------

        for candidate in candidates:

            try:

                response = await client.get(
                    candidate,
                    headers=self._headers(),
                )

                print(
                    "[IMAGE] Hugging Face: "
                    f"download status="
                    f"{response.status_code} "
                    f"url={candidate}",
                    flush=True,
                )

                if (
                    response.status_code
                    != 200
                ):
                    continue

                content = response.content

                content_type = (
                    response.headers.get(
                        "content-type",
                        "",
                    )
                    .lower()
                )

                # PNG
                if content.startswith(
                    b"\x89PNG\r\n\x1a\n"
                ):
                    return content

                # JPEG
                if content.startswith(
                    b"\xff\xd8"
                ):
                    return content

                # WEBP
                if (
                    len(content) >= 12
                    and content[:4] == b"RIFF"
                    and content[8:12] == b"WEBP"
                ):
                    return content

                # MIME
                if content_type.startswith(
                    "image/"
                ):
                    return content

            except (
                httpx.HTTPError,
                OSError,
            ) as exc:

                print(
                    "[IMAGE] Hugging Face: "
                    f"download failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

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
    ) -> tuple[
        str,
        ImageResult | None,
    ]:

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
        # JSON
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
            f"data="
            f"{self._safe_json(event_data, 1500)}",
            flush=True,
        )

        # =====================================================
        # ERROR
        # =====================================================

        if event_name == "error":

            message = (
                self._extract_error_message(
                    event_data
                )
            )

            if not message:

                if (
                    raw_data.lower()
                    == "null"
                ):
                    message = (
                        "Gradio returned "
                        "an SSE error event "
                        "with data=null. "
                        "The Space rejected or "
                        "failed during execution."
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
        # COMPLETE
        # =====================================================

        if event_name in (
            "complete",
            "done",
        ):

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

        print(
            "[IMAGE] Hugging Face: "
            f"image detected "
            f"url={image_url!r} "
            f"path={image_path!r}",
            flush=True,
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
    # GRADIO SSE
    # =========================================================

    async def _generate_sse(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        print(
            "[IMAGE] Hugging Face: "
            "starting Gradio SSE generation",
            flush=True,
        )

        params = (
            self._generation_parameters(
                request
            )
        )

        print(
            "[IMAGE] Hugging Face: "
            f"steps={params['steps']} "
            f"guidance="
            f"{params['guidance_scale']} "
            f"seed={params['seed']} "
            f"strength="
            f"{params['strength']}",
            flush=True,
        )

        # =====================================================
        # MODO
        # =====================================================
        #
        # O nosso Agent gera atualmente texto -> imagem.
        #
        # Portanto enviamos:
        #
        #   t2i
        #
        # =====================================================

        mode = "t2i"

        # =====================================================
        # PAYLOAD CORRETO
        # =====================================================
        #
        # ATENÇÃO:
        #
        # O Space aceita EXATAMENTE 8 argumentos:
        #
        # 0 mode
        # 1 t2i_prompt
        # 2 i2i_prompt
        # 3 i2i_image
        # 4 strength
        # 5 steps
        # 6 guidance_scale
        # 7 seed
        #
        # NÃO enviar style_preset.
        # NÃO enviar aspect_ratio.
        #
        # =====================================================

        payload = {
            "data": [
                mode,
                prompt,
                "",
                None,
                params["strength"],
                params["steps"],
                params["guidance_scale"],
                params["seed"],
            ]
        }

        print(
            "[IMAGE] Hugging Face: "
            f"POST payload="
            f"{self._safe_json(payload, 2500)}",
            flush=True,
        )

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

        print(
            "[IMAGE] Hugging Face: "
            f"SSE POST status="
            f"{response.status_code} "
            f"body="
            f"{response.text[:1500]}",
            flush=True,
        )

        if (
            response.status_code
            != 200
        ):

            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_"
                    f"{response.status_code}: "
                    f"{response.text[:1500]}"
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
                    f"{response.text[:1500]}"
                ),
            )

        event_id = (
            initial_data.get(
                "event_id"
            )
        )

        if not event_id:

            return ImageResult(
                False,
                self.name,
                error=(
                    "sse_no_event_id: "
                    f"{self._safe_json(initial_data)}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event_id={event_id}",
            flush=True,
        )

        # =====================================================
        # STREAM
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

                print(
                    "[IMAGE] Hugging Face: "
                    f"SSE stream status="
                    f"{result_response.status_code}",
                    flush=True,
                )

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
                # SSE LOOP
                # -------------------------------------------------

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.rstrip(
                        "\r"
                    )

                    # -------------------------------------------------
                    # Evento finalizado
                    # -------------------------------------------------

                    if line == "":

                        if not current_data_lines:

                            current_event = None

                            continue

                        raw_data = (
                            "\n".join(
                                current_data_lines
                            ).strip()
                        )

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
                    # Comentário
                    # -------------------------------------------------

                    if line.startswith(
                        ":"
                    ):
                        continue

                    # -------------------------------------------------
                    # EVENT
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
                    # DATA
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
                # Último evento sem linha vazia
                # -----------------------------------------------------

                if current_data_lines:

                    raw_data = (
                        "\n".join(
                            current_data_lines
                        ).strip()
                    )

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
    # DIRECT ENDPOINT
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

        # -----------------------------------------------------
        # Mesmo conjunto de 8 parâmetros do Space.
        #
        # O endpoint /run/generate_images pode existir
        # dependendo da versão do Gradio.
        # -----------------------------------------------------

        payload = {
            "data": [
                "t2i",
                prompt,
                "",
                None,
                params["strength"],
                params["steps"],
                params["guidance_scale"],
                params["seed"],
            ]
        }

        print(
            "[IMAGE] Hugging Face: "
            "trying direct endpoint",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: "
            f"direct payload="
            f"{self._safe_json(payload, 2500)}",
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

        if (
            response.status_code
            != 200
        ):

            return ImageResult(
                False,
                self.name,
                error=(
                    f"direct_http_"
                    f"{response.status_code}: "
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
                    "direct_invalid_json: "
                    f"{response.text[:1500]}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"direct response="
            f"{self._safe_json(data, 2500)}",
            flush=True,
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
                    "direct_no_image_output: "
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
                    "direct_image_download_failed: "
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
    # GENERATE
    # =========================================================

    async def generate(
        self,
        request,
        prompt: str,
    ) -> ImageResult:

        # -----------------------------------------------------
        # CONFIGURAÇÃO
        # -----------------------------------------------------

        if not await self.available():

            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # -----------------------------------------------------
        # PROMPT
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

                print(
                    "[IMAGE] Hugging Face: "
                    f"SSE failed: "
                    f"{result.error}",
                    flush=True,
                )

                error_text = (
                    result.error
                    or ""
                )

                # =================================================
                # FALLBACK
                # =================================================

                if (
                    "sse_generation_error"
                    in error_text
                    or "no_image_in_sse_response"
                    in error_text
                    or "sse_http_404"
                    in error_text
                    or "sse_http_405"
                    in error_text
                    or "sse_post_error"
                    in error_text
                ):

                    print(
                        "[IMAGE] Hugging Face: "
                        "trying direct endpoint "
                        "as fallback",
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
        # HTTP
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
        # OUTROS ERROS
        # =====================================================

        except Exception as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "
