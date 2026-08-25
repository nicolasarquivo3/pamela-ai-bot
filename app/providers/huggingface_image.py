import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider de geração de imagens usando o Space oficial:

        black-forest-labs/FLUX.2-klein-4B

    Endpoint Gradio:

        /gradio_api/call/infer

    Assinatura atual do Space:

        infer(
            prompt,
            input_images=None,
            mode_choice="Distilled (4 steps)",
            seed=42,
            randomize_seed=False,
            width=1024,
            height=1024,
            num_inference_steps=4,
            guidance_scale=4.0,
            prompt_upsampling=False,
        )

    O provider também suporta:
        - geração somente por prompt;
        - imagem(ns) de referência;
        - upload de imagens para o Space;
        - autenticação via HF_TOKEN;
        - SSE do Gradio;
        - download do resultado;
        - respostas FileData;
        - respostas aninhadas;
        - fallback para diferentes formatos de arquivo.
    """

    name = "huggingface"

    DEFAULT_SPACE_URL = (
        "https://black-forest-labs-flux-2-klein-4b.hf.space"
    )

    INFER_API_NAME = "infer"

    DISTILLED_MODE = "Distilled (4 steps)"
    BASE_MODE = "Base (50 steps)"

    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 256

    DISTILLED_STEPS = 4
    DISTILLED_GUIDANCE = 1.0

    BASE_STEPS = 50
    BASE_GUIDANCE = 4.0

    def __init__(
        self,
        space_url: str | None = None,
        timeout: int = 180,
        hf_token: str | None = None,
    ):
        configured_url = (
            space_url
            or os.getenv("HF_FLUX_SPACE_URL")
            or self.DEFAULT_SPACE_URL
        )

        self.space_url = configured_url.rstrip("/")

        self.timeout = timeout

        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )

    # =========================================================
    # AVAILABILITY
    # =========================================================

    async def available(self) -> bool:
        return bool(self.space_url)

    # =========================================================
    # HEADERS
    # =========================================================

    def _headers(
        self,
        accept: str = "application/json",
    ) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "pamela-ai/1.0",
        }

        if self.hf_token:
            headers["Authorization"] = (
                f"Bearer {self.hf_token}"
            )

        return headers

    def _sse_headers(self) -> dict[str, str]:
        headers = self._headers(
            "text/event-stream"
        )

        headers["Cache-Control"] = "no-cache"

        return headers

    # =========================================================
    # SAFE JSON
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

    # =========================================================
    # ERROR EXTRACTION
    # =========================================================

    @classmethod
    def _extract_error_message(
        cls,
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
                "traceback",
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
                    return cls._safe_json(
                        candidate
                    )

            return None

        if isinstance(value, list):
            for item in value:
                message = (
                    cls._extract_error_message(
                        item
                    )
                )

                if message:
                    return message

        return None

    # =========================================================
    # IMAGE DICT
    # =========================================================

    @staticmethod
    def _is_image_dict(
        value,
    ) -> bool:
        if not isinstance(value, dict):
            return False

        if value.get("path"):
            return True

        if value.get("url"):
            return True

        if value.get("image"):
            return True

        return False

    # =========================================================
    # IMAGE EXTRACTION
    # =========================================================

    @classmethod
    def _extract_image_info(
        cls,
        value,
    ):
        """
        Procura recursivamente um FileData/arquivo de imagem
        dentro da resposta do Gradio.

        Exemplos aceitos:

            {
                "path": "/tmp/gradio/image.png"
            }

        ou:

            {
                "url": "https://..."
            }

        ou:

            {
                "image": {
                    "path": "..."
                }
            }

        ou listas/estruturas aninhadas.
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

            lowered = text.lower()

            image_extensions = (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".bmp",
            )

            if (
                lowered.startswith(
                    (
                        "http://",
                        "https://",
                    )
                )
                or lowered.endswith(
                    image_extensions
                )
            ):
                return {
                    "url": (
                        text
                        if lowered.startswith(
                            (
                                "http://",
                                "https://",
                            )
                        )
                        else None
                    ),
                    "path": (
                        text
                        if not lowered.startswith(
                            (
                                "http://",
                                "https://",
                            )
                        )
                        else None
                    ),
                }

            return None

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(value, dict):

            if cls._is_image_dict(value):

                nested_image = value.get(
                    "image"
                )

                if isinstance(
                    nested_image,
                    dict,
                ):
                    nested = cls._extract_image_info(
                        nested_image
                    )

                    if nested:
                        return nested

                return {
                    "url": value.get("url"),
                    "path": value.get("path"),
                }

            for key in (
                "image",
                "images",
                "output",
                "outputs",
                "result",
                "results",
                "data",
                "gallery",
                "value",
                "file",
                "files",
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
        # List / Tuple
        # -----------------------------------------------------

        if isinstance(
            value,
            (list, tuple),
        ):
            for item in value:

                nested = cls._extract_image_info(
                    item
                )

                if nested:
                    return nested

        return None

    # =========================================================
    # MIME / IMAGE VALIDATION
    # =========================================================

    @staticmethod
    def _looks_like_image(
        content: bytes,
        content_type: str,
    ) -> bool:
        if not content:
            return False

        content_type = (
            content_type
            or ""
        ).lower()

        # PNG
        if content.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            return True

        # JPEG
        if content.startswith(
            b"\xff\xd8"
        ):
            return True

        # WEBP
        if (
            len(content) >= 12
            and content[:4] == b"RIFF"
            and content[8:12] == b"WEBP"
        ):
            return True

        # GIF
        if content.startswith(
            (
                b"GIF87a",
                b"GIF89a",
            )
        ):
            return True

        # MIME
        if content_type.startswith(
            "image/"
        ):
            return True

        return False

    # =========================================================
    # DIMENSIONS
    # =========================================================

    @staticmethod
    def _normalize_dimension(
        value,
        default: int = 1024,
    ) -> int:
        try:
            value = int(value)
        except (
            TypeError,
            ValueError,
        ):
            value = default

        value = max(
            256,
            min(
                1024,
                value,
            ),
        )

        # FLUX.2 Space usa múltiplos de 8.
        value = round(
            value / 8
        ) * 8

        value = max(
            256,
            min(
                1024,
                value,
            ),
        )

        return value

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

        width = cls._normalize_dimension(
            width
        )

        height = cls._normalize_dimension(
            height
        )

        # -----------------------------------------------------
        # Modelo
        # -----------------------------------------------------

        mode = cls.DISTILLED_MODE

        if request is not None:

            requested_mode = getattr(
                request,
                "hf_mode",
                None,
            )

            if not requested_mode:
                requested_mode = getattr(
                    request,
                    "mode",
                    None,
                )

            if requested_mode:

                normalized = str(
                    requested_mode
                ).lower()

                if (
                    "base" in normalized
                    or "50" in normalized
                ):
                    mode = cls.BASE_MODE

        # -----------------------------------------------------
        # Seed
        # -----------------------------------------------------

        seed = 42

        if request is not None:

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

        seed = max(
            0,
            seed,
        )

        # -----------------------------------------------------
        # Randomize
        # -----------------------------------------------------

        randomize_seed = False

        if request is not None:

            requested_randomize = getattr(
                request,
                "randomize_seed",
                None,
            )

            if requested_randomize is not None:
                randomize_seed = bool(
                    requested_randomize
                )

        # -----------------------------------------------------
        # Steps / CFG
        # -----------------------------------------------------

        if mode == cls.BASE_MODE:

            steps = cls.BASE_STEPS
            guidance = cls.BASE_GUIDANCE

        else:

            steps = cls.DISTILLED_STEPS
            guidance = cls.DISTILLED_GUIDANCE

        # -----------------------------------------------------
        # Overrides
        # -----------------------------------------------------

        if request is not None:

            requested_steps = getattr(
                request,
                "steps",
                None,
            )

            if requested_steps is not None:

                try:
                    requested_steps = int(
                        requested_steps
                    )

                    if (
                        requested_steps >= 1
                        and requested_steps <= 100
                    ):
                        steps = requested_steps

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            requested_guidance = getattr(
                request,
                "guidance_scale",
                None,
            )

            if requested_guidance is not None:

                try:
                    requested_guidance = float(
                        requested_guidance
                    )

                    guidance = max(
                        0.0,
                        min(
                            10.0,
                            requested_guidance,
                        ),
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # -----------------------------------------------------
        # Prompt upsampling
        # -----------------------------------------------------

        prompt_upsampling = False

        if request is not None:

            requested_upsampling = getattr(
                request,
                "prompt_upsampling",
                None,
            )

            if requested_upsampling is not None:
                prompt_upsampling = bool(
                    requested_upsampling
                )

        return {
            "mode_choice": mode,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt_upsampling": prompt_upsampling,
        }

    # =========================================================
    # REQUEST IMAGE PATH
    # =========================================================

    @classmethod
    def _extract_reference_path(
        cls,
        request,
    ) -> list[str]:
        """
        Tenta encontrar uma imagem de referência no request.

        Não assume uma estrutura específica da aplicação.

        Aceita:
            request.reference_image_path
            request.face_reference_image_path
            request.input_image_path
            request.image_path

        Também aceita listas.
        """

        if request is None:
            return []

        candidates = []

        for attribute in (
            "reference_image_path",
            "face_reference_image_path",
            "input_image_path",
            "image_path",
            "reference_image",
            "input_image",
        ):

            value = getattr(
                request,
                attribute,
                None,
            )

            if value:
                candidates.append(
                    value
                )

        result = []

        for candidate in candidates:

            if isinstance(
                candidate,
                (list, tuple),
            ):

                for item in candidate:

                    if item:
                        result.append(
                            str(item)
                        )

            else:

                result.append(
                    str(candidate)
                )

        # Remove duplicados
        return list(
            dict.fromkeys(result)
        )

    # =========================================================
    # UPLOAD FILE TO GRADIO
    # =========================================================

    async def _upload_file(
        self,
        client: httpx.AsyncClient,
        file_path: str,
    ) -> dict | None:
        """
        Faz upload da imagem local para o Space.

        O Gradio retorna algo como:

            [
                "/tmp/gradio/....png"
            ]

        Depois transformamos em FileData.
        """

        path = Path(file_path)

        if not path.exists():
            print(
                "[IMAGE] Hugging Face: "
                f"reference file not found: {file_path}",
                flush=True,
            )

            return None

        try:

            content = path.read_bytes()

        except OSError as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"could not read reference file: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            return None

        if not content:
            return None

        content_type = (
            "image/jpeg"
            if path.suffix.lower()
            in (
                ".jpg",
                ".jpeg",
            )
            else "image/png"
        )

        try:

            response = await client.post(
                f"{self.space_url}"
                "/gradio_api/upload",
                files={
                    "files": (
                        path.name,
                        content,
                        content_type,
                    )
                },
                headers=self._headers(),
            )

        except httpx.HTTPError as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"upload failed: {exc}",
                flush=True,
            )

            return None

        if response.status_code != 200:

            print(
                "[IMAGE] Hugging Face: "
                "upload HTTP "
                f"{response.status_code}: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        try:

            uploaded = response.json()

        except json.JSONDecodeError:

            print(
                "[IMAGE] Hugging Face: "
                "upload returned invalid JSON: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        uploaded_path = None

        if isinstance(
            uploaded,
            list,
        ) and uploaded:

            uploaded_path = uploaded[0]

        elif isinstance(
            uploaded,
            dict,
        ):

            uploaded_path = (
                uploaded.get("path")
                or uploaded.get("url")
            )

        if not uploaded_path:

            print(
                "[IMAGE] Hugging Face: "
                "upload returned no path: "
                f"{self._safe_json(uploaded)}",
                flush=True,
            )

            return None

        print(
            "[IMAGE] Hugging Face: "
            f"reference uploaded: {uploaded_path}",
            flush=True,
        )

        return {
            "path": str(
                uploaded_path
            ),
            "orig_name": path.name,
            "meta": {
                "_type": "gradio.FileData"
            },
        }

    # =========================================================
    # PREPARE INPUT IMAGES
    # =========================================================

    async def _prepare_input_images(
        self,
        client: httpx.AsyncClient,
        request=None,
    ) -> list[dict] | None:

        reference_paths = (
            self._extract_reference_path(
                request
            )
        )

        if not reference_paths:
            return None

        uploaded_images = []

        for path in reference_paths:

            uploaded = await self._upload_file(
                client,
                path,
            )

            if uploaded:
                uploaded_images.append(
                    uploaded
                )

        if not uploaded_images:
            return None

        return uploaded_images

    # =========================================================
    # DOWNLOAD RESULT
    # =========================================================

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        image_url: str | None,
        image_path: str | None,
    ) -> bytes | None:

        candidates: list[str] = []

        # -----------------------------------------------------
        # Direct URL
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

        # -----------------------------------------------------
        # Path
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

                normalized = (
                    image_path.lstrip("/")
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

                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file=/{encoded}"
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
                    headers=self._headers(
                        "image/*"
                    ),
                )

            except httpx.HTTPError:
                continue

            if response.status_code != 200:
                continue

            content = response.content

            content_type = (
                response.headers.get(
                    "content-type",
                    "",
                )
            )

            if self._looks_like_image(
                content,
                content_type,
            ):
                return content

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

        raw_data = (
            raw_data
            or ""
        ).strip()

        if not raw_data:
            return (
                "continue",
                None,
            )

        # -----------------------------------------------------
        # Parse
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
            f"data={self._safe_json(event_data, 1500)}",
            flush=True,
        )

        # =====================================================
        # ERROR EVENT
        # =====================================================

        if event_name.lower() == "error":

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
        # COMPLETE EVENT
        # =====================================================

        if event_name.lower() in (
            "complete",
            "completed",
        ):

            image_info = (
                self._extract_image_info(
                    event_data
                )
            )

            if not image_info:

                return (
                    "error",
                    ImageResult(
                        False,
                        self.name,
                        error=(
                            "sse_complete_without_image: "
                            f"{self._safe_json(event_data)}"
                        ),
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

        # =====================================================
        # IMAGE IN ANY EVENT
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
            "continue",
            None,
        )

    # =========================================================
    # SSE GENERATION
    # =========================================================

    async def _generate_sse(
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

        print(
            "[IMAGE] Hugging Face: "
            "starting official FLUX.2 Klein 4B generation",
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

        # =====================================================
        # INPUT IMAGES
        # =====================================================

        input_images = (
            await self._prepare_input_images(
                client,
                request,
            )
        )

        reference_enabled = bool(
            input_images
        )

        print(
            "[IMAGE] Hugging Face: "
            f"reference={reference_enabled}",
            flush=True,
        )

        # =====================================================
        # GRADIO PAYLOAD
        # =====================================================

        payload = {
            "data": [
                prompt,
                input_images,
                params["mode_choice"],
                params["seed"],
                params["randomize_seed"],
                params["width"],
                params["height"],
                params["num_inference_steps"],
                params["guidance_scale"],
                params["prompt_upsampling"],
            ]
        }

        print(
            "[IMAGE] Hugging Face: "
            "POST /gradio_api/call/infer "
            f"payload={self._safe_json(payload, 2500)}",
            flush=True,
        )

        # =====================================================
        # POST
        # =====================================================

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
                error="gradio_post_timeout",
            )

        except httpx.HTTPError as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_post_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        body = response.text

        print(
            "[IMAGE] Hugging Face: "
            f"POST status={response.status_code} "
            f"body={body[:3000]}",
            flush=True,
        )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"gradio_post_http_"
                    f"{response.status_code}: "
                    f"{body[:3000]}"
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
                    f"{body[:2000]}"
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
                or initial_data.get(
                    "eventId"
                )
            )

        if not event_id:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_no_event_id: "
                    f"{self._safe_json(initial_data)}"
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
                f"infer/"
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

                    body_bytes = await (
                        stream.aread()
                    )

                    text = body_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "gradio_sse_http_"
                            f"{stream.status_code}: "
                            f"{text[:2500]}"
                        ),
                    )

                async for raw_line in (
                    stream.aiter_lines()
                ):

                    line = (
                        raw_line
                        or ""
                    ).rstrip(
                        "\r"
                    )

                    # -------------------------------------------------
                    # Empty line = end of SSE event
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
                    # SSE comment
                    # -------------------------------------------------

                    if line.startswith(":"):
                        continue

                    # -------------------------------------------------
                    # Event name
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
                    # Data
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
                # Stream ended without blank line.
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
                error="gradio_sse_timeout",
            )

        except httpx.HTTPError as exc:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_sse_http_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return ImageResult(
            False,
            self.name,
            error=(
                "no_image_in_gradio_sse_response"
            ),
        )

    # =========================================================
    # INFO ENDPOINT
    # =========================================================

    async def _get_space_info(
        self,
        client: httpx.AsyncClient,
    ) -> dict | None:

        try:

            response = await client.get(
                f"{self.space_url}"
                "/gradio_api/info",
                headers=self._headers(),
            )

        except httpx.HTTPError as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"could not read Gradio info: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            return None

        if response.status_code != 200:

            print(
                "[IMAGE] Hugging Face: "
                f"info HTTP {response.status_code}: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        try:
            return response.json()

        except json.JSONDecodeError:
            return None

    # =========================================================
    # DIRECT GENERATION
    # =========================================================

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:
        """
        Mantido como fallback.

        O Space oficial utiliza a API de chamada assíncrona
        /gradio_api/call/infer, portanto SSE é a rota principal.

        Este método não tenta inventar outro endpoint de geração.
        Em caso de falha do SSE, fazemos uma nova tentativa da
        chamada oficial, evitando endpoints antigos incompatíveis.
        """

        print(
            "[IMAGE] Hugging Face: "
            "retrying official Gradio endpoint",
            flush=True,
        )

        return await self._generate_sse(
            client,
            prompt,
            request,
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
        # CONFIGURATION
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
        # TOKEN
        # =====================================================

        if self.hf_token:

            print(
                "[IMAGE] Hugging Face: "
                "HF_TOKEN detected; "
                "authenticated ZeroGPU request",
                flush=True,
            )

        else:

            print(
                "[IMAGE] Hugging Face: "
                "WARNING: HF_TOKEN not detected; "
                "request will be anonymous",
                flush=True,
            )

        # =====================================================
        # TIMEOUT
        # =====================================================

        timeout = httpx.Timeout(
            connect=30.0,
            read=float(
                self.timeout
            ),
            write=float(
                self.timeout
            ),
            pool=30.0,
        )

        try:

            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:

                # =================================================
                # MAIN REQUEST
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

                # =================================================
                # RETRY
                # =================================================

                print(
                    "[IMAGE] Hugging Face: "
                    f"generation failed: "
                    f"{result.error}",
                    flush=True,
                )

                print(
                    "[IMAGE] Hugging Face: "
                    "performing one retry using "
                    "the same official endpoint",
                    flush=True,
                )

                retry_result = (
                    await self._generate_direct(
                        client,
                        prompt,
                        request,
                    )
                )

                if retry_result.success:
                    return retry_result

                return ImageResult(
                    False,
                    self.name,
                    error=(
                        f"first={result.error}; "
                        f"retry={retry_result.error}"
                    ),
                )

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
        # UNEXPECTED
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
