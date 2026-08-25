import json
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider específico para o Space oficial da Black Forest Labs:

        black-forest-labs/FLUX.2-klein-4B

    O Space expõe a função Gradio:

        /infer

    Assinatura atual:

        infer(
            prompt,
            input_images,
            mode_choice,
            seed,
            randomize_seed,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            prompt_upsampling,
        )

    A chamada é feita pela API HTTP do Gradio e o resultado é
    recebido pelo endpoint SSE:

        /gradio_api/call/infer/{event_id}
    """

    name = "huggingface"

    DEFAULT_SPACE_URL = (
        "https://black-forest-labs-flux2-klein-4b.hf.space"
    )

    INFER_API_NAME = "/infer"

    MODE_DISTILLED = "Distilled (4 steps)"
    MODE_BASE = "Base (50 steps)"

    def __init__(
        self,
        space_url: str | None = None,
        timeout: int = 180,
        hf_token: str | None = None,
    ):
        self.space_url = (
            (space_url or self.DEFAULT_SPACE_URL).rstrip("/")
        )

        self.timeout = int(
            timeout or 180
        )

        self.hf_token = hf_token

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
            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        return headers

    def _sse_headers(self) -> dict[str, str]:
        headers = self._headers()

        headers["Accept"] = (
            "text/event-stream"
        )

        headers["Cache-Control"] = (
            "no-cache"
        )

        return headers

    # =========================================================
    # UTILS
    # =========================================================

    @staticmethod
    def _safe_json(
        value,
        limit: int = 2500,
    ) -> str:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        except Exception:
            text = repr(value)

        return text[:limit]

    @staticmethod
    def _safe_int(
        value,
        default: int,
    ) -> int:
        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_float(
        value,
        default: float,
    ) -> float:
        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _extract_error_message(
        value,
    ) -> str | None:

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):
            text = value.strip()

            if text:
                return text

            return None

        if isinstance(
            value,
            dict,
        ):

            for key in (
                "error",
                "message",
                "detail",
                "msg",
                "exception",
                "title",
            ):

                candidate = value.get(
                    key
                )

                if candidate is None:
                    continue

                if isinstance(
                    candidate,
                    str,
                ):

                    text = (
                        candidate.strip()
                    )

                    if text:
                        return text

                else:

                    try:
                        return json.dumps(
                            candidate,
                            ensure_ascii=False,
                            default=str,
                        )

                    except Exception:
                        return str(
                            candidate
                        )

            return None

        if isinstance(
            value,
            list,
        ):

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
    # GENERATION PARAMETERS
    # =========================================================

    @classmethod
    def _generation_parameters(
        cls,
        request=None,
    ) -> dict:

        width = 1024
        height = 1024

        steps = 4

        guidance_scale = 1.0

        seed = 42

        requested_style = None
        requested_seed = None

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

            requested_style = getattr(
                request,
                "style",
                None,
            )

            requested_seed = getattr(
                request,
                "seed",
                None,
            )

        width = cls._safe_int(
            width,
            1024,
        )

        height = cls._safe_int(
            height,
            1024,
        )

        # -----------------------------------------------------
        # O Space oficial aceita:
        #
        # mínimo: 256
        # máximo: 1024
        #
        # e trabalha com múltiplos de 8.
        # -----------------------------------------------------

        width = max(
            256,
            min(
                1024,
                width,
            ),
        )

        height = max(
            256,
            min(
                1024,
                height,
            ),
        )

        width = max(
            256,
            min(
                1024,
                round(width / 8) * 8,
            ),
        )

        height = max(
            256,
            min(
                1024,
                round(height / 8) * 8,
            ),
        )

        if requested_seed is not None:

            seed = cls._safe_int(
                requested_seed,
                42,
            )

        # -----------------------------------------------------
        # Modo oficial padrão
        # -----------------------------------------------------

        mode_choice = (
            cls.MODE_DISTILLED
        )

        # -----------------------------------------------------
        # Caso o request peça explicitamente
        # o modelo Base.
        # -----------------------------------------------------

        if requested_style:

            normalized = str(
                requested_style
            ).lower()

            if (
                "base"
                in normalized
                or "50"
                in normalized
            ):

                mode_choice = (
                    cls.MODE_BASE
                )

                steps = 50

                guidance_scale = 4.0

        # -----------------------------------------------------
        # Valores oficiais por modo
        # -----------------------------------------------------

        if (
            mode_choice
            == cls.MODE_DISTILLED
        ):

            steps = 4

            guidance_scale = 1.0

        else:

            steps = 50

            guidance_scale = 4.0

        return {
            "width": width,
            "height": height,
            "mode_choice": mode_choice,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "randomize_seed": False,
            "prompt_upsampling": False,
        }

    # =========================================================
    # IMAGE RESULT EXTRACTION
    # =========================================================

    @staticmethod
    def _is_image_dict(
        value,
    ) -> bool:

        if not isinstance(
            value,
            dict,
        ):
            return False

        path = value.get(
            "path"
        )

        url = value.get(
            "url"
        )

        name = value.get(
            "name"
        )

        mime = (
            value.get(
                "mime_type"
            )
            or value.get(
                "mimeType"
            )
        )

        if path or url or name:

            if mime and str(
                mime
            ).startswith(
                "image/"
            ):
                return True

            return True

        return False

    @classmethod
    def _extract_image_info(
        cls,
        value,
    ):
        """
        Extrai FileData do Gradio.

        Exemplos aceitos:

            {
                "path": "...",
                "url": "..."
            }

        ou:

            [
                {
                    "path": "...",
                    "url": "..."
                },
                42
            ]
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            if cls._is_image_dict(
                value
            ):

                return {
                    "url": value.get(
                        "url"
                    ),
                    "path": value.get(
                        "path"
                    ),
                    "name": value.get(
                        "name"
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
                "value",
            ):

                if key not in value:
                    continue

                nested = (
                    cls._extract_image_info(
                        value.get(
                            key
                        )
                    )
                )

                if nested:
                    return nested

            return None

        # -----------------------------------------------------
        # List / tuple
        # -----------------------------------------------------

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
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

        # -----------------------------------------------------
        # String
        # -----------------------------------------------------

        if isinstance(
            value,
            str,
        ):

            text = value.strip()

            if not text:
                return None

            lowered = text.lower()

            if (
                lowered.startswith(
                    "http://"
                )
                or lowered.startswith(
                    "https://"
                )
                or lowered.startswith(
                    "/"
                )
                or lowered.endswith(
                    (
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                    )
                )
            ):

                if lowered.startswith(
                    (
                        "http://",
                        "https://",
                    )
                ):

                    return {
                        "url": text,
                        "path": None,
                        "name": None,
                    }

                return {
                    "url": None,
                    "path": text,
                    "name": None,
                }

        return None

    # =========================================================
    # IMAGE DOWNLOAD
    # =========================================================

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        image_url: str | None,
        image_path: str | None,
        image_name: str | None = None,
    ) -> bytes | None:

        candidates: list[str] = []

        def add_candidate(
            value: str | None,
        ):

            if not value:
                return

            value = str(
                value
            ).strip()

            if not value:
                return

            # -------------------------------------------------
            # URL absoluta
            # -------------------------------------------------

            if value.startswith(
                (
                    "http://",
                    "https://",
                )
            ):

                candidates.append(
                    value
                )

                return

            # -------------------------------------------------
            # Caminho Gradio
            # -------------------------------------------------

            normalized = (
                value.lstrip("/")
            )

            encoded = quote(
                normalized,
                safe="",
            )

            candidates.append(
                f"{self.space_url}"
                f"/gradio_api/file="
                f"{encoded}"
            )

            candidates.append(
                f"{self.space_url}"
                f"/file="
                f"{encoded}"
            )

            # -------------------------------------------------
            # Caminho direto
            # -------------------------------------------------

            if value.startswith(
                "/"
            ):

                candidates.append(
                    f"{self.space_url}"
                    f"{value}"
                )

        add_candidate(
            image_url
        )

        add_candidate(
            image_path
        )

        add_candidate(
            image_name
        )

        candidates = list(
            dict.fromkeys(
                candidates
            )
        )

        # -----------------------------------------------------
        # Tenta cada candidato
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
                    ).lower()
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
                    and content[:4]
                    == b"RIFF"
                    and content[8:12]
                    == b"WEBP"
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
    ) -> tuple[
        str,
        ImageResult | None,
    ]:

        event_name = (
            event_name
            or "message"
        )

        raw_data = (
            raw_data.strip()
        )

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
            f"{self._safe_json(event_data, 1800)}",
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
                        "with data=null"
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
                event_data.get(
                    key
                )
                is not None
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

        image_name = image_info.get(
            "name"
        )

        image_bytes = (
            await self._download_image(
                client,
                image_url,
                image_path,
                image_name,
            )
        )

        if image_bytes:

            return (
                "success",
                ImageResult(
                    True,
                    self.name,
                    image_url=(
                        image_url
                        or image_path
                        or image_name
                    ),
                    image_bytes=image_bytes,
                ),
            )

        return (
            "error",
            ImageResult(
                False,
                self.name,
                image_url=(
                    image_url
                    or image_path
                    or image_name
                ),
                error=(
                    "image_download_failed: "
                    f"url={image_url!r} "
                    f"path={image_path!r} "
                    f"name={image_name!r}"
                ),
            ),
        )

    # =========================================================
    # POST /infer
    # =========================================================

    async def _submit_infer(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        params: dict,
    ) -> ImageResult:

        """
        Envia exatamente a assinatura atual do
        infer() do Space oficial.

        A ordem é:

            1. prompt
            2. input_images
            3. mode_choice
            4. seed
            5. randomize_seed
            6. width
            7. height
            8. num_inference_steps
            9. guidance_scale
            10. prompt_upsampling
        """

        # =====================================================
        # IMPORTANTE
        #
        # Para T2I não enviamos None.
        #
        # O componente Gallery do Gradio recebe uma lista vazia.
        # =====================================================

        input_images = []

        payload = {
            "data": [
                prompt,
                input_images,
                params["mode_choice"],
                params["seed"],
                params["randomize_seed"],
                params["width"],
                params["height"],
                params["steps"],
                params["guidance_scale"],
                params["prompt_upsampling"],
            ]
        }

        print(
            "[IMAGE] Hugging Face: "
            "starting official "
            "FLUX.2 Klein 4B generation",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: "
            f"mode="
            f"{params['mode_choice']!r} "
            f"width="
            f"{params['width']} "
            f"height="
            f"{params['height']} "
            f"steps="
            f"{params['steps']} "
            f"guidance="
            f"{params['guidance_scale']} "
            f"seed="
            f"{params['seed']} "
            "randomize=False "
            "reference=False",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: "
            "POST /gradio_api/call/infer",
            flush=True,
        )

        try:

            response = await client.post(
                (
                    f"{self.space_url}"
                    "/gradio_api/call"
                    f"{self.INFER_API_NAME}"
                ),
                json=payload,
                headers=self._headers(),
            )

        except httpx.TimeoutException:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_post_timeout"
                ),
            )

        except httpx.HTTPError as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_post_http_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        body = response.text[:4000]

        print(
            "[IMAGE] Hugging Face: "
            f"POST status="
            f"{response.status_code} "
            f"body="
            f"{body[:1800]}",
            flush=True,
        )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"gradio_post_http_"
                    f"{response.status_code}: "
                    f"{body}"
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
                    "gradio_invalid_initial_json: "
                    f"{body[:1500]}"
                ),
            )

        event_id = None

        if isinstance(
            initial_data,
            dict,
        ):

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
                    "gradio_no_event_id: "
                    f"{self._safe_json(initial_data, 1800)}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"SSE event_id={event_id}",
            flush=True,
        )

        # =====================================================
        # SSE
        # =====================================================

        return await self._read_sse(
            client,
            event_id,
        )

    # =========================================================
    # SSE STREAM
    # =========================================================

    async def _read_sse(
        self,
        client: httpx.AsyncClient,
        event_id: str,
    ) -> ImageResult:

        current_event: str | None = None

        current_data_lines: list[
            str
        ] = []

        url = (
            f"{self.space_url}"
            f"/gradio_api/call"
            f"{self.INFER_API_NAME}"
            f"/{event_id}"
        )

        try:

            async with client.stream(
                "GET",
                url,
                headers=self._sse_headers(),
            ) as response:

                print(
                    "[IMAGE] Hugging Face: "
                    f"SSE stream "
                    f"status="
                    f"{response.status_code}",
                    flush=True,
                )

                if (
                    response.status_code
                    != 200
                ):

                    body = (
                        await response.aread()
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
                            f"{response.status_code}: "
                            f"{text[:2500]}"
                        ),
                    )

                # =================================================
                # Ler stream
                # =================================================

                async for raw_line in (
                    response.aiter_lines()
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

                # =====================================================
                # Último evento sem linha vazia
                # =====================================================

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
                        status
                        in (
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

        # =====================================================
        # CONFIGURAÇÃO
        # =====================================================

        if not await self.available():

            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # =====================================================
        # PROMPT
        # =====================================================

        if (
            not prompt
            or not prompt.strip()
        ):

            return ImageResult(
                False,
                self.name,
                error="empty_prompt",
            )

        # =====================================================
        # PARÂMETROS
        # =====================================================

        params = (
            self._generation_parameters(
                request
            )
        )

        # =====================================================
        # TIMEOUT
        # =====================================================

        timeout_seconds = max(
            180,
            self.timeout,
        )

        timeout = httpx.Timeout(
            connect=45.0,
            read=timeout_seconds,
            write=60.0,
            pool=60.0,
        )

        # =====================================================
        # CLIENT
        # =====================================================

        try:

            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:

                result = (
                    await self._submit_infer(
                        client,
                        prompt.strip(),
                        params,
                    )
                )

                if result.success:

                    return result

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
        # OUTRO ERRO
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
