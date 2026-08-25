import json
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider para o Space:

        Xurxowsky/flux2-klein-4b-playground

    Esse Space utiliza Gradio e expõe a função:

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

    O acesso é feito através da Queue API do Gradio:

        POST /gradio_api/call/generate_images
        GET  /gradio_api/call/generate_images/{event_id}
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
            "Content-Type": "application/json",
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
            )
        except Exception:
            text = repr(value)

        return text[:limit]

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
            ):
                candidate = value.get(key)

                if candidate is None:
                    continue

                if isinstance(candidate, str):
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
                    ._extract_error_message(item)
                )

                if message:
                    return message

        return None

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
            or value.get("name")
        )

    @classmethod
    def _extract_image_info(cls, value):
        """
        Procura recursivamente uma imagem dentro da resposta
        do Gradio.

        Aceita estruturas como:

            {
                "path": "...",
                "url": "..."
            }

        ou:

            {
                "image": {
                    "path": "...",
                    "url": "..."
                }
            }

        ou:

            [
                {
                    "path": "...",
                    "url": "..."
                }
            ]

        ou GalleryData/listas aninhadas.
        """

        if value is None:
            return None

        # -----------------------------------------------------
        # STRING
        # -----------------------------------------------------

        if isinstance(value, str):
            text = value.strip()

            if (
                text.startswith("http://")
                or text.startswith("https://")
            ):
                return {
                    "url": text,
                    "path": None,
                }

            if (
                text.lower().endswith(".png")
                or text.lower().endswith(".jpg")
                or text.lower().endswith(".jpeg")
                or text.lower().endswith(".webp")
            ):
                return {
                    "url": None,
                    "path": text,
                }

            return None

        # -----------------------------------------------------
        # DICT
        # -----------------------------------------------------

        if isinstance(value, dict):

            if cls._is_image_dict(value):
                return {
                    "url": value.get("url"),
                    "path": (
                        value.get("path")
                        or value.get("name")
                    ),
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
        # LISTA / TUPLA
        # -----------------------------------------------------

        if isinstance(value, (list, tuple)):

            for item in value:
                nested = cls._extract_image_info(
                    item
                )

                if nested:
                    return nested

        return None

    # =========================================================
    # PARÂMETROS
    # =========================================================

    @staticmethod
    def _generation_parameters(request=None) -> dict:
        """
        Parâmetros compatíveis com o Space específico.

        O Space não recebe style_preset nem aspect_ratio.

        Valores usados pelo Space:

            mode
            t2i_prompt
            i2i_prompt
            i2i_image
            strength
            steps
            guidance_scale
            seed
        """

        steps = 4
        guidance_scale = 4.0
        seed = 42
        strength = 0.8

        if request is not None:

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

            requested_strength = getattr(
                request,
                "strength",
                None,
            )

            if requested_strength is not None:
                try:
                    strength = float(
                        requested_strength
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    strength = 0.8

        # O Space define máximo de 50 passos.
        steps = max(
            1,
            min(
                steps,
                50,
            ),
        )

        guidance_scale = max(
            1.0,
            min(
                guidance_scale,
                10.0,
            ),
        )

        strength = max(
            0.0,
            min(
                strength,
                1.0,
            ),
        )

        return {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "strength": strength,
        }

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
                candidates.append(
                    image_url
                )

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
                candidates.append(
                    image_path
                )

            else:

                clean_path = (
                    image_path.lstrip("/")
                )

                encoded_path = quote(
                    clean_path,
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
                        f"{self.space_url}"
                        f"{image_path}"
                    )

        # -----------------------------------------------------
        # Remove duplicados
        # -----------------------------------------------------

        candidates = list(
            dict.fromkeys(candidates)
        )

        if not candidates:
            return None

        print(
            "[IMAGE] Hugging Face: "
            f"trying {len(candidates)} image URL(s)",
            flush=True,
        )

        # -----------------------------------------------------
        # Tentativas
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
                        f"{response.status_code} "
                        f"for {candidate}",
                        flush=True,
                    )
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
                    f"image download failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

        return None

    # =========================================================
    # PROCESSAR EVENTO SSE
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
            f"data={self._safe_json(event_data, 1500)}",
            flush=True,
        )

        # =====================================================
        # ERRO
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
        # COMPLETED / COMPLETE
        # =====================================================

        if event_name in (
            "complete",
            "completed",
        ):
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

                    return (
                        "success",
                        ImageResult(
                            True,
                            self.name,
                            image_url=image_url,
                            image_bytes=image_bytes,
                        ),
                    )

            # Um evento complete sem imagem
            # não deve ser tratado como sucesso.
            return (
                "continue",
                None,
            )

        # =====================================================
        # QUALQUER EVENTO COM IMAGEM
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
            f"guidance={params['guidance_scale']} "
            f"seed={params['seed']} "
            f"strength={params['strength']}",
            flush=True,
        )

        # =====================================================
        # PAYLOAD EXATO DO SPACE
        # =====================================================
        #
        # generate_images(
        #     mode,
        #     t2i_prompt,
        #     i2i_prompt,
        #     i2i_image,
        #     strength,
        #     steps,
        #     guidance_scale,
        #     seed,
        # )
        #
        # Para nosso caso usamos text-to-image:
        #
        # mode = "t2i"
        #
        # =====================================================

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
            f"POST payload={self._safe_json(payload, 2000)}",
            flush=True,
        )

        # =====================================================
        # POST PARA CRIAR JOB
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
            f"SSE POST status={response.status_code} "
            f"body={response.text[:1000]}",
            flush=True,
        )

        if response.status_code != 200:

            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_{response.status_code}: "
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
                    f"{self._safe_json(initial_data)}"
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
                # Ler SSE
                # -------------------------------------------------

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.rstrip(
                        "\r"
                    )

                    # -------------------------------------------------
                    # Comentários SSE
                    # -------------------------------------------------

                    if line.startswith(":"):
                        continue

                    # -------------------------------------------------
                    # Fim de evento
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
                            and result is not None
                        ):
                            return result

                        if (
                            status == "error"
                            and result is not None
                        ):
                            return result

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

                # -------------------------------------------------
                # Último evento sem linha vazia
                # -------------------------------------------------

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
                        and result is not None
                    ):
                        return result

        except httpx.TimeoutException:

            print(
                "[IMAGE] Hugging Face: "
                "SSE timeout",
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

        # -----------------------------------------------------
        # Cliente HTTP
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
