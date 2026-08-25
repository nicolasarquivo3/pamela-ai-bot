import json
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

    Interface utilizada pelo Space:

        /infer

    Parâmetros da função infer():

        prompt
        input_images
        seed
        randomize_seed
        width
        height
        num_inference_steps
        guidance_scale
        prompt_upsampling

    O Space utiliza Gradio e a chamada é feita através da
    API de fila/SSE do Gradio.
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

    def _headers(
        self,
        accept: str = "application/json",
    ) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "pamela-ai/1.0",
        }

        if self.hf_token:
            token = f"Bearer {self.hf_token}"

            headers["Authorization"] = token
            headers["x-hf-authorization"] = token

        return headers

    def _sse_headers(self) -> dict[str, str]:
        headers = self._headers(
            "text/event-stream"
        )

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
            )
        except Exception:
            text = repr(value)

        return text[:limit]

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

    # =========================================================
    # IMAGE EXTRACTION
    # =========================================================

    @classmethod
    def _extract_image_info(
        cls,
        value,
    ):
        """
        Procura recursivamente uma imagem dentro das respostas
        retornadas pelo Gradio.

        Aceita:

            {"path": "...", "url": "..."}

        ou:

            {"image": {...}}

        ou:

            [{"path": "..."}]

        além das estruturas FileData/Gallery.
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

            if (
                text.startswith("http://")
                or text.startswith("https://")
                or text.startswith("/")
                or text.startswith("file=")
            ):
                return {
                    "url": (
                        text
                        if text.startswith(
                            (
                                "http://",
                                "https://",
                            )
                        )
                        else None
                    ),
                    "path": text,
                }

            return None

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(value, dict):

            # FileData
            path = value.get("path")
            url = value.get("url")

            if path or url:

                return {
                    "url": url,
                    "path": path,
                }

            # Campos conhecidos
            for key in (
                "image",
                "output",
                "data",
                "result",
                "results",
                "gallery",
                "images",
                "file",
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
        # Lista / Tuple
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
    # DIMENSIONS
    # =========================================================

    @staticmethod
    def _get_dimensions(
        request=None,
    ) -> tuple[int, int]:

        width = 1024
        height = 1024

        if request is not None:

            try:
                width = int(
                    getattr(
                        request,
                        "width",
                        1024,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                width = 1024

            try:
                height = int(
                    getattr(
                        request,
                        "height",
                        1024,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                height = 1024

        # -----------------------------------------------------
        # O Space oficial limita as dimensões a 256-1024
        # e trabalha em múltiplos de 8.
        # -----------------------------------------------------

        width = max(
            256,
            min(1024, width),
        )

        height = max(
            256,
            min(1024, height),
        )

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

        return width, height

    # =========================================================
    # GENERATION SETTINGS
    # =========================================================

    @staticmethod
    def _get_generation_settings(
        request=None,
    ) -> dict:

        # -----------------------------------------------------
        # FLUX.2 Klein é um modelo distilled.
        #
        # 4 steps é uma configuração adequada para o Space
        # e mantém a geração rápida.
        # -----------------------------------------------------

        steps = 4
        guidance_scale = 1.0
        seed = 42
        randomize_seed = False
        prompt_upsampling = False

        if request is not None:

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
                    steps = 4

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
                    guidance_scale = 1.0

            guidance_scale = max(
                0.0,
                min(20.0, guidance_scale),
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
                    seed = 42

            seed = max(
                0,
                min(
                    2147483647,
                    seed,
                ),
            )

            # -------------------------------------------------
            # Randomize seed
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

        return {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "prompt_upsampling": prompt_upsampling,
        }

    # =========================================================
    # INPUT IMAGE DISCOVERY
    # =========================================================

    @staticmethod
    def _find_input_image_path(
        request=None,
    ) -> str | None:
        """
        Tenta encontrar uma imagem de referência caso o objeto
        ImageRequest forneça um caminho.

        Isso não interfere no fluxo normal de geração.

        Possíveis nomes aceitos:

            input_image_path
            reference_image_path
            image_path
            source_image_path
        """

        if request is None:
            return None

        for attribute in (
            "input_image_path",
            "reference_image_path",
            "image_path",
            "source_image_path",
        ):

            value = getattr(
                request,
                attribute,
                None,
            )

            if not value:
                continue

            path = str(value)

            if os.path.isfile(path):
                return path

        return None

    # =========================================================
    # UPLOAD IMAGE TO GRADIO
    # =========================================================

    async def _upload_image(
        self,
        client: httpx.AsyncClient,
        image_path: str,
    ):
        """
        Faz upload de uma imagem local para o Space Gradio.

        O resultado é utilizado no parâmetro input_images da
        função /infer.
        """

        if not image_path:
            return None

        path = Path(image_path)

        if not path.is_file():
            return None

        try:

            with path.open(
                "rb"
            ) as file:

                response = await client.post(
                    f"{self.space_url}"
                    "/gradio_api/upload",
                    files={
                        "files": (
                            path.name,
                            file,
                            "application/octet-stream",
                        )
                    },
                    headers=self._headers(),
                )

        except (
            httpx.HTTPError,
            OSError,
        ) as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"image upload failed: {exc}",
                flush=True,
            )

            return None

        if response.status_code not in (
            200,
            201,
        ):

            print(
                "[IMAGE] Hugging Face: "
                f"image upload HTTP "
                f"{response.status_code}: "
                f"{response.text[:1000]}",
                flush=True,
            )

            return None

        try:
            data = response.json()
        except json.JSONDecodeError:
            return None

        # Gradio normalmente retorna uma lista de caminhos.
        if isinstance(
            data,
            list,
        ) and data:

            uploaded_path = data[0]

            if isinstance(
                uploaded_path,
                str,
            ):
                return {
                    "path": uploaded_path
                }

            if isinstance(
                uploaded_path,
                dict,
            ):
                return uploaded_path

        if isinstance(
            data,
            dict,
        ):

            return data

        return None

    # =========================================================
    # DOWNLOAD GENERATED IMAGE
    # =========================================================

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        image_url: str | None,
        image_path: str | None,
    ) -> bytes | None:

        candidates: list[str] = []

        # -----------------------------------------------------
        # URL absoluta
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

                # Se já for um caminho absoluto do Space
                if image_path.startswith("/"):
                    candidates.append(
                        f"{self.space_url}"
                        f"{image_path}"
                    )

        candidates = list(
            dict.fromkeys(candidates)
        )

        for candidate in candidates:

            try:

                response = await client.get(
                    candidate,
                    headers=self._headers(
                        "image/*"
                    ),
                )

            except (
                httpx.HTTPError,
                OSError,
            ):
                continue

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
            raw_data
            or ""
        ).strip()

        if not raw_data:
            return (
                "continue",
                None,
            )

        # -----------------------------------------------------
        # Gradio pode enviar JSON
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
            f"{self._safe_json(event_data, 1200)}",
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
        # COMPLETE / IMAGE
        # =====================================================

        image_info = (
            self._extract_image_info(
                event_data
            )
        )

        if image_info:

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

                print(
                    "[IMAGE] Hugging Face: "
                    "image successfully downloaded",
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
        # COMPLETE SEM IMAGEM
        # =====================================================

        if event_name == "complete":

            return (
                "continue",
                None,
            )

        return (
            "continue",
            None,
        )

    # =========================================================
    # GENERATE VIA OFFICIAL GRADIO SPACE
    # =========================================================

    async def _generate_sse(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        print(
            "[IMAGE] Hugging Face: "
            "starting official FLUX.2 Klein 4B "
            "Gradio generation",
            flush=True,
        )

        # =====================================================
        # SETTINGS
        # =====================================================

        width, height = (
            self._get_dimensions(
                request
            )
        )

        generation = (
            self._get_generation_settings(
                request
            )
        )

        steps = generation[
            "steps"
        ]

        guidance_scale = generation[
            "guidance_scale"
        ]

        seed = generation[
            "seed"
        ]

        randomize_seed = generation[
            "randomize_seed"
        ]

        prompt_upsampling = generation[
            "prompt_upsampling"
        ]

        # =====================================================
        # INPUT IMAGE
        # =====================================================

        input_images = []

        input_image_path = (
            self._find_input_image_path(
                request
            )
        )

        if input_image_path:

            uploaded = (
                await self._upload_image(
                    client,
                    input_image_path,
                )
            )

            if uploaded:

                # Gallery aceita FileData.
                input_images = [
                    uploaded
                ]

                print(
                    "[IMAGE] Hugging Face: "
                    "reference image uploaded: "
                    f"{input_image_path}",
                    flush=True,
                )

            else:

                print(
                    "[IMAGE] Hugging Face: "
                    "reference image upload "
                    "failed; continuing without "
                    "input image",
                    flush=True,
                )

        # =====================================================
        # PAYLOAD OFICIAL
        # =====================================================
        #
        # Corresponde exatamente à função:
        #
        # infer(
        #     prompt,
        #     input_images,
        #     seed,
        #     randomize_seed,
        #     width,
        #     height,
        #     num_inference_steps,
        #     guidance_scale,
        #     prompt_upsampling
        # )
        #
        # =====================================================

        payload = {
            "data": [
                prompt,
                input_images,
                seed,
                randomize_seed,
                width,
                height,
                steps,
                guidance_scale,
                prompt_upsampling,
            ]
        }

        print(
            "[IMAGE] Hugging Face: "
            f"POST /gradio_api/call/infer "
            f"width={width} "
            f"height={height} "
            f"steps={steps} "
            f"guidance={guidance_scale} "
            f"seed={seed} "
            f"randomize={randomize_seed} "
            f"upsampling={prompt_upsampling}",
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

        except httpx.HTTPError as exc:

            print(
                "[IMAGE] Hugging Face: "
                f"POST failed: {exc}",
                flush=True,
            )

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_post_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        print(
            "[IMAGE] Hugging Face: "
            f"POST status={response.status_code} "
            f"body={response.text[:1000]}",
            flush=True,
        )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    "gradio_post_http_"
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
                    "gradio_invalid_initial_json: "
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
                            "gradio_sse_http_"
                            f"{stream.status_code}: "
                            f"{text[:1500]}"
                        ),
                    )

                # -------------------------------------------------
                # Ler SSE
                # -------------------------------------------------

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
                # Último evento sem linha vazia
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
                "gradio_sse_finished_without_image"
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

        # -----------------------------------------------------
        # Timeout maior para GPU/ZeroGPU.
        # -----------------------------------------------------

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

                result = (
                    await self._generate_sse(
                        client,
                        prompt,
                        request,
                    )
                )

                if result.success:

                    return result

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
