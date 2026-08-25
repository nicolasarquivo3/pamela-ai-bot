import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider específico para o Space oficial:

        black-forest-labs/FLUX.2-klein-4B

    Space:
        https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B

    Endpoint:
        /gradio_api/call/infer

    Assinatura atual do Space:

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

    O provider usa a API HTTP/SSE do Gradio diretamente,
    sem depender de gradio_client.
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
    # CONFIGURAÇÃO
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
    # UTILITÁRIOS
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
                default=str,
            )
        except Exception:
            text = repr(value)

        return text[:limit]

    @staticmethod
    def _is_http_url(value) -> bool:
        return isinstance(value, str) and value.startswith(
            (
                "http://",
                "https://",
            )
        )

    @staticmethod
    def _is_image_dict(value) -> bool:
        if not isinstance(value, dict):
            return False

        path = value.get("path")
        url = value.get("url")

        if path or url:
            return True

        return False

    @classmethod
    def _extract_image_info(cls, value):
        """
        Procura recursivamente uma imagem dentro dos retornos
        do Gradio.

        O retorno pode ser algo como:

            [
                {
                    "path": "...",
                    "url": "..."
                },
                42
            ]

        ou:

            {
                "path": "...",
                "url": "..."
            }

        ou:

            {
                "data": [...]
            }

        ou estruturas FileData aninhadas.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # String
        # -----------------------------------------------------

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return None

            lower = text.lower()

            if (
                lower.endswith(".png")
                or lower.endswith(".jpg")
                or lower.endswith(".jpeg")
                or lower.endswith(".webp")
                or lower.endswith(".gif")
                or lower.endswith(".bmp")
                or lower.startswith("http://")
                or lower.startswith("https://")
            ):
                return {
                    "url": (
                        text
                        if cls._is_http_url(text)
                        else None
                    ),
                    "path": (
                        None
                        if cls._is_http_url(text)
                        else text
                    ),
                }

            return None

        # -----------------------------------------------------
        # Dicionário
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
                "value",
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
        # Lista / tupla
        # -----------------------------------------------------

        if isinstance(value, (list, tuple)):

            for item in value:

                nested = cls._extract_image_info(
                    item
                )

                if nested:
                    return nested

        return None

    @staticmethod
    def _extract_error_message(value) -> str | None:
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
                            default=str,
                        )
                    except Exception:
                        return str(candidate)

            return None

        if isinstance(value, (list, tuple)):

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
    # GERAÇÃO DE PARÂMETROS
    # =========================================================

    @staticmethod
    def _generation_parameters(
        request=None,
    ) -> dict:
        """
        Parâmetros compatíveis com o Space oficial.

        Distilled:
            4 steps
            guidance 1.0

        Base:
            30 steps
            guidance 4.0

        O modo padrão é Distilled (4 steps).
        """

        mode_choice = "Distilled (4 steps)"

        seed = 42
        randomize_seed = False

        width = 1024
        height = 1024

        num_inference_steps = 4
        guidance_scale = 1.0

        prompt_upsampling = False

        if request is not None:

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
                    seed = 42

            # -------------------------------------------------
            # Random seed
            # -------------------------------------------------

            requested_randomize = getattr(
                request,
                "randomize_seed",
                None,
            )

            if requested_randomize is not None:
                randomize_seed = bool(
                    requested_randomize
                )

            # -------------------------------------------------
            # Width
            # -------------------------------------------------

            requested_width = getattr(
                request,
                "width",
                None,
            )

            if requested_width is not None:

                try:
                    width = int(
                        requested_width
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    width = 1024

            # -------------------------------------------------
            # Height
            # -------------------------------------------------

            requested_height = getattr(
                request,
                "height",
                None,
            )

            if requested_height is not None:

                try:
                    height = int(
                        requested_height
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    height = 1024

            # -------------------------------------------------
            # Mode
            # -------------------------------------------------

            requested_mode = getattr(
                request,
                "mode_choice",
                None,
            )

            if requested_mode:

                normalized_mode = (
                    str(
                        requested_mode
                    )
                    .strip()
                    .lower()
                )

                if (
                    "30" in normalized_mode
                    or "base" in normalized_mode
                    or "regular" in normalized_mode
                ):
                    mode_choice = (
                        "Regular (30 steps)"
                    )

            # -------------------------------------------------
            # Steps
            # -------------------------------------------------

            requested_steps = getattr(
                request,
                "num_inference_steps",
                None,
            )

            if requested_steps is not None:

                try:
                    num_inference_steps = int(
                        requested_steps
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

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
                    pass

            # -------------------------------------------------
            # Prompt upsampling
            # -------------------------------------------------

            requested_upsampling = getattr(
                request,
                "prompt_upsampling",
                None,
            )

            if requested_upsampling is not None:
                prompt_upsampling = bool(
                    requested_upsampling
                )

        # -----------------------------------------------------
        # Defaults coerentes com o modo
        # -----------------------------------------------------

        if mode_choice == "Distilled (4 steps)":

            if not hasattr(
                request,
                "num_inference_steps",
            ) if request is not None else True:
                num_inference_steps = 4

            if not hasattr(
                request,
                "guidance_scale",
            ) if request is not None else True:
                guidance_scale = 1.0

        else:

            if not hasattr(
                request,
                "num_inference_steps",
            ) if request is not None else True:
                num_inference_steps = 30

            if not hasattr(
                request,
                "guidance_scale",
            ) if request is not None else True:
                guidance_scale = 4.0

        # -----------------------------------------------------
        # Validação do Space
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

        # O Space exige múltiplos de 8.
        width = round(width / 8) * 8
        height = round(height / 8) * 8

        width = max(
            256,
            min(1024, width),
        )

        height = max(
            256,
            min(1024, height),
        )

        num_inference_steps = max(
            1,
            min(
                100,
                num_inference_steps,
            ),
        )

        guidance_scale = max(
            0.0,
            min(
                10.0,
                guidance_scale,
            ),
        )

        return {
            "mode_choice": mode_choice,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt_upsampling": prompt_upsampling,
        }

    # =========================================================
    # REFERÊNCIA DE IMAGEM
    # =========================================================

    @staticmethod
    def _find_reference_path(
        request=None,
    ) -> str | None:
        """
        Tenta encontrar uma imagem de referência no request.

        São aceitos vários nomes para manter compatibilidade
        com diferentes versões do ImageRequest.
        """

        if request is None:
            return None

        candidates = (
            "reference_image_path",
            "input_image_path",
            "image_path",
            "face_reference_image_path",
            "reference_path",
        )

        for attribute in candidates:

            value = getattr(
                request,
                attribute,
                None,
            )

            if not value:
                continue

            if isinstance(value, str):
                path = value.strip()

                if path and os.path.isfile(path):
                    return path

        return None

    # =========================================================
    # UPLOAD PARA O GRADIO
    # =========================================================

    async def _upload_reference_image(
        self,
        client: httpx.AsyncClient,
        image_path: str,
    ):
        """
        Faz upload da imagem para o servidor Gradio.

        O Gallery do Space é:

            gr.Gallery(type="pil")

        Portanto o servidor precisa receber um FileData válido.
        """

        if not image_path:
            return None

        path = Path(
            image_path
        )

        if not path.is_file():
            print(
                "[IMAGE] Hugging Face: "
                f"reference image not found: {path}",
                flush=True,
            )
            return None

        mime_type = (
            mimetypes.guess_type(
                path.name
            )[0]
            or "image/jpeg"
        )

        try:

            with path.open(
                "rb"
            ) as file:

                files = {
                    "files": (
                        path.name,
                        file,
                        mime_type,
                    )
                }

                response = await client.post(
                    f"{self.space_url}"
                    "/gradio_api/upload",
                    files=files,
                    headers={
                        key: value
                        for key, value
                        in self._headers().items()
                        if key
                        not in (
                            "Accept",
                            "Content-Type",
                        )
                    },
                )

        except (
            httpx.HTTPError,
            OSError,
        ) as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"reference upload failed: {exc}",
                flush=True,
            )

            return None

        if response.status_code not in (
            200,
            201,
        ):

            print(
                "[IMAGE] Hugging Face: "
                "reference upload HTTP "
                f"{response.status_code}: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        try:

            data = response.json()

        except json.JSONDecodeError:

            print(
                "[IMAGE] Hugging Face: "
                "invalid upload response: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        if isinstance(data, list) and data:

            uploaded_path = data[0]

        elif isinstance(data, dict):

            uploaded_path = (
                data.get("path")
                or data.get("url")
            )

        else:

            uploaded_path = None

        if not uploaded_path:
            return None

        # -----------------------------------------------------
        # FileData usado pelo Gradio
        # -----------------------------------------------------

        file_data = {
            "path": uploaded_path,
            "url": (
                uploaded_path
                if self._is_http_url(
                    uploaded_path
                )
                else None
            ),
            "orig_name": path.name,
            "size": path.stat().st_size,
            "mime_type": mime_type,
            "meta": {
                "_type": "gradio.FileData",
            },
        }

        return file_data

    # =========================================================
    # DOWNLOAD DA IMAGEM RESULTANTE
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

            if self._is_http_url(
                image_url
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
        # PATH do Gradio
        # -----------------------------------------------------

        if image_path:

            if self._is_http_url(
                image_path
            ):
                candidates.append(
                    image_path
                )

            else:

                normalized_path = (
                    str(image_path)
                    .lstrip("/")
                )

                encoded_path = quote(
                    normalized_path,
                    safe="",
                )

                candidates.extend(
                    [
                        (
                            f"{self.space_url}"
                            f"/gradio_api/file="
                            f"{encoded_path}"
                        ),
                        (
                            f"{self.space_url}"
                            f"/file="
                            f"{encoded_path}"
                        ),
                        (
                            f"{self.space_url}"
                            f"/gradio_api/file=/"
                            f"{encoded_path}"
                        ),
                    ]
                )

        candidates = list(
            dict.fromkeys(
                candidates
            )
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
            ):
                continue

        return None

    # =========================================================
    # PROCESSAMENTO DO EVENTO SSE
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

        raw_data = (
            raw_data
            .strip()
        )

        if not raw_data:
            return (
                "continue",
                None,
            )

        # -----------------------------------------------------
        # Heartbeat
        # -----------------------------------------------------

        if raw_data.lower() in (
            "null",
            "null\n",
        ) and event_name in (
            "heartbeat",
            "generating",
        ):
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
            f"data={self._safe_json(event_data, 1200)}",
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

                if raw_data.lower() == "null":
                    message = (
                        "Gradio returned an SSE "
                        "error event with data=null"
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
        # COMPLETION / OUTPUT
        # =====================================================

        image_info = (
            self._extract_image_info(
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
                await self._download_image(
                    client,
                    image_url,
                    image_path,
                )
            )

            if image_bytes:

                print(
                    "[IMAGE] Hugging Face: "
                    "image downloaded successfully",
                    flush=True,
                )

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

        # =====================================================
        # CONTINUE
        # =====================================================

        return (
            "continue",
            None,
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
        # Provider disponível?
        # -----------------------------------------------------

        if not await self.available():

            return ImageResult(
                False,
                self.name,
                error="not_configured",
            )

        # -----------------------------------------------------
        # Prompt válido?
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

        params = (
            self._generation_parameters(
                request
            )
        )

        print(
            "[IMAGE] Hugging Face: "
            "starting official "
            "FLUX.2 Klein 4B generation",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: "
            f"mode={params['mode_choice']!r} "
            f"width={params['width']} "
            f"height={params['height']} "
            f"steps={params['num_inference_steps']} "
            f"guidance={params['guidance_scale']} "
            f"seed={params['seed']} "
            f"randomize={params['randomize_seed']} "
            f"upsampling={params['prompt_upsampling']}",
            flush=True,
        )

        # -----------------------------------------------------
        # Timeout
        # -----------------------------------------------------

        timeout = httpx.Timeout(
            connect=30.0,
            read=self.timeout,
            write=self.timeout,
            pool=30.0,
        )

        try:

            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:

                # =================================================
                # IMAGEM DE REFERÊNCIA
                # =================================================

                input_images = None

                reference_path = (
                    self._find_reference_path(
                        request
                    )
                )

                if reference_path:

                    reference_file = (
                        await self
                        ._upload_reference_image(
                            client,
                            reference_path,
                        )
                    )

                    if reference_file:

                        # Gallery:
                        #
                        # list[
                        #     (
                        #         image,
                        #         caption
                        #     )
                        # ]
                        #
                        # A API recebe FileData.
                        input_images = [
                            [
                                reference_file,
                                None,
                            ]
                        ]

                        print(
                            "[IMAGE] Hugging Face: "
                            "reference image uploaded",
                            flush=True,
                        )

                    else:

                        print(
                            "[IMAGE] Hugging Face: "
                            "reference upload failed; "
                            "continuing without reference",
                            flush=True,
                        )

                # =================================================
                # PAYLOAD EXATO DO SPACE
                # =================================================
                #
                # Ordem confirmada no app.py:
                #
                # 1 prompt
                # 2 input_images
                # 3 mode_choice
                # 4 seed
                # 5 randomize_seed
                # 6 width
                # 7 height
                # 8 num_inference_steps
                # 9 guidance_scale
                # 10 prompt_upsampling
                #
                # =================================================

                payload = {
                    "data": [
                        prompt,
                        input_images,
                        params["mode_choice"],
                        params["seed"],
                        params["randomize_seed"],
                        params["width"],
                        params["height"],
                        params[
                            "num_inference_steps"
                        ],
                        params[
                            "guidance_scale"
                        ],
                        params[
                            "prompt_upsampling"
                        ],
                    ]
                }

                print(
                    "[IMAGE] Hugging Face: "
                    "POST /gradio_api/call/infer "
                    f"width={params['width']} "
                    f"height={params['height']} "
                    f"steps={params['num_inference_steps']} "
                    f"guidance={params['guidance_scale']} "
                    f"seed={params['seed']} "
                    f"randomize={params['randomize_seed']} "
                    f"reference="
                    f"{bool(input_images)}",
                    flush=True,
                )

                # Não imprimimos o payload inteiro porque ele
                # pode conter a imagem de referência.

                # =================================================
                # POST INICIAL
                # =================================================

                try:

                    response = await client.post(
                        f"{self.space_url}"
                        "/gradio_api/call/infer",
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

                print(
                    "[IMAGE] Hugging Face: "
                    f"POST status="
                    f"{response.status_code} "
                    f"body="
                    f"{response.text[:2000]}",
                    flush=True,
                )

                if response.status_code != 200:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "gradio_post_http_"
                            f"{response.status_code}: "
                            f"{response.text[:2000]}"
                        ),
                    )

                # =================================================
                # EVENT ID
                # =================================================

                try:

                    initial_data = (
                        response.json()
                    )

                except json.JSONDecodeError:

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "gradio_invalid_initial_response: "
                            f"{response.text[:1000]}"
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
                            "gradio_missing_event_id: "
                            f"{self._safe_json(initial_data)}"
                        ),
                    )

                print(
                    "[IMAGE] Hugging Face: "
                    f"SSE event_id={event_id}",
                    flush=True,
                )

                # =================================================
                # SSE
                # =================================================

                current_event = None

                current_data_lines: list[
                    str
                ] = []

                async with client.stream(
                    "GET",
                    f"{self.space_url}"
                    "/gradio_api/call/infer/"
                    f"{event_id}",
                    headers=self._sse_headers(),
                ) as stream:

                    print(
                        "[IMAGE] Hugging Face: "
                        f"SSE stream status="
                        f"{stream.status_code}",
                        flush=True,
                    )

                    if stream.status_code != 200:

                        body = await (
                            stream.aread()
                        )

                        text = body.decode(
                            "utf-8",
                            errors="replace",
                        )

                        return ImageResult(
                            False,
                            self.name,
                            error=(
                                "sse_http_"
                                f"{stream.status_code}: "
                                f"{text[:2000]}"
                            ),
                        )

                    async for raw_line in (
                        stream.aiter_lines()
                    ):

                        line = raw_line.rstrip(
                            "\r"
                        )

                        # -------------------------------------------------
                        # Fim do evento
                        # -------------------------------------------------

                        if line == "":

                            if not current_data_lines:

                                current_event = None
                                continue

                            raw_data = (
                                "\n".join(
                                    current_data_lines
                                )
                                .strip()
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
                                status
                                == "success"
                                and result
                            ):
                                return result

                            if (
                                status
                                == "error"
                                and result
                            ):
                                return result

                            continue

                        # -------------------------------------------------
                        # Comentários / heartbeat
                        # -------------------------------------------------

                        if line.startswith(":"):
                            continue

                        # -------------------------------------------------
                        # Event
                        # -------------------------------------------------

                        if line.startswith(
                            "event:"
                        ):

                            current_event = (
                                line[
                                    len(
                                        "event:"
                                    ):
                                ].strip()
                            )

                            continue

                        # -------------------------------------------------
                        # Data
                        # -------------------------------------------------

                        if line.startswith(
                            "data:"
                        ):

                            current_data_lines.append(
                                line[
                                    len(
                                        "data:"
                                    ):
                                ].lstrip()
                            )

                            continue

                    # =====================================================
                    # ÚLTIMO EVENTO SEM LINHA VAZIA
                    # =====================================================

                    if current_data_lines:

                        raw_data = (
                            "\n".join(
                                current_data_lines
                            )
                            .strip()
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

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        "no_image_in_sse_response"
                    ),
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
                error=(
                    "http_client_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        except Exception as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"unexpected error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            return ImageResult(
                False,
                self.name,
                error=(
                    "unexpected_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
