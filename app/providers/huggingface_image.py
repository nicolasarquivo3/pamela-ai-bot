from __future__ import annotations

import json
import random
from urllib.parse import quote

import httpx

from app.images.models import ImageResult
from app.providers.image import ImageProvider


class HuggingFaceImageProvider(ImageProvider):
    """
    Provider específico para o Space:

        Xurxowsky/flux2-klein-4b-playground

    URL esperada:

        https://xurxowsky-flux2-klein-4b-playground.hf.space

    A função pública do Space é:

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
    O Space NÃO possui os parâmetros style_preset ou aspect_ratio.

    O endpoint utilizado é o sistema de Queue/SSE do Gradio:

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
        """
        Prepara SOMENTE os parâmetros aceitos pelo Space.

        O Space aceita:

            mode
            t2i_prompt
            i2i_prompt
            i2i_image
            strength
            steps
            guidance_scale
            seed

        Não enviamos:

            style_preset
            aspect_ratio

        porque esses campos NÃO existem na função
        generate_images() desse Space.
        """

        # -----------------------------------------------------
        # Valores padrão do Space
        # -----------------------------------------------------

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
        # Steps
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

        # O Space aceita de 1 a 50.
        steps = max(1, min(steps, 50))

        # -----------------------------------------------------
        # Guidance
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

        # O componente do Space usa 1.0 até 10.0.
        guidance_scale = max(
            1.0,
            min(guidance_scale, 10.0),
        )

        # -----------------------------------------------------
        # Seed
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

        # Nosso projeto usa -1 para "aleatório".
        # O Space espera uma seed inteira.
        if seed < 0:
            seed = random.randint(
                0,
                2_147_483_647,
            )

        # -----------------------------------------------------
        # Strength
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

            if not text:
                return None

            # Às vezes o Gradio manda JSON como string.
            if (
                text.startswith("{")
                or text.startswith("[")
            ):
                try:
                    decoded = json.loads(text)

                    nested = cls._extract_error_message(
                        decoded
                    )

                    if nested:
                        return nested
                except Exception:
                    pass

            return text

        if isinstance(value, dict):
            for key in (
                "error",
                "exception",
                "message",
                "detail",
                "msg",
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
                            default=str,
                        )
                    except Exception:
                        return str(candidate)

            return None

        if isinstance(value, list):
            for item in value:
                message = cls._extract_error_message(
                    item
                )

                if message:
                    return message

        return None

    # =========================================================
    # IMAGE DETECTION
    # =========================================================

    @staticmethod
    def _is_image_dict(value) -> bool:
        if not isinstance(value, dict):
            return False

        return bool(
            value.get("url")
            or value.get("path")
            or value.get("image")
        )

    # =========================================================
    # IMAGE EXTRACTION
    # =========================================================

    @classmethod
    def _extract_image_info(
        cls,
        value,
    ):
        """
        Extrai recursivamente uma imagem retornada pelo Gradio.

        O Space retorna:

            return [result], status

        e o Gallery pode representar a imagem como:

            [
                {
                    "path": "...",
                    "url": "..."
                }
            ]

        Também suportamos estruturas aninhadas do Gradio.
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
            ):
                return {
                    "url": text,
                    "path": None,
                }

            if (
                text.startswith("/")
                or text.startswith("file=")
            ):
                return {
                    "url": None,
                    "path": text,
                }

            return None

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(value, dict):

            # Gradio file object.
            if (
                value.get("url")
                or value.get("path")
            ):
                return {
                    "url": value.get("url"),
                    "path": value.get("path"),
                }

            # Algumas respostas podem colocar a imagem
            # dentro de "image".
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
        # List / Tuple
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
    # IMAGE DOWNLOAD
    # =========================================================

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        image_url: str | None,
        image_path: str | None,
    ) -> bytes | None:

        candidates: list[str] = []

        # -----------------------------------------------------
        # URL retornada diretamente pelo Gradio
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

                # Gradio moderno.
                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file="
                    f"{encoded_path}"
                )

                # Algumas versões retornam o caminho
                # sem o prefixo esperado.
                candidates.append(
                    f"{self.space_url}"
                    f"/gradio_api/file=/{encoded_path}"
                )

                # Endpoint legado.
                candidates.append(
                    f"{self.space_url}"
                    f"/file={encoded_path}"
                )

                # Última tentativa: caminho direto.
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
            f"download candidates={candidates}",
            flush=True,
        )

        # -----------------------------------------------------
        # Download
        # -----------------------------------------------------

        for candidate in candidates:

            try:

                response = await client.get(
                    candidate,
                    headers=self._headers(),
                )

                content_type = (
                    response.headers.get(
                        "content-type",
                        "",
                    )
                    .lower()
                )

                print(
                    "[IMAGE] Hugging Face: "
                    f"download status="
                    f"{response.status_code} "
                    f"content-type="
                    f"{content_type!r} "
                    f"url={candidate}",
                    flush=True,
                )

                if response.status_code != 200:
                    continue

                content = response.content

                if not content:
                    continue

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
                # GIF
                # -------------------------------------------------

                if content.startswith(
                    (
                        b"GIF87a",
                        b"GIF89a",
                    )
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
            ) as exc:

                print(
                    "[IMAGE] Hugging Face: "
                    f"download failed "
                    f"url={candidate!r} "
                    f"error={exc}",
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
            f"data={self._safe_json(event_data, 1500)}",
            flush=True,
        )

        # =====================================================
        # ERROR
        # =====================================================

        if event_name.lower() == "error":

            message = (
                self._extract_error_message(
                    event_data
                )
            )

            if not message:
                message = (
                    "Hugging Face/Gradio returned "
                    "an SSE error event"
                )

                if raw_data:
                    message += (
                        f": {raw_data[:1000]}"
                    )

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
        # EVENTOS DE PROGRESSO / GENERAÇÃO
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

                return (
                    "success",
                    ImageResult(
                        True,
                        self.name,
                        image_url=image_url,
                        image_bytes=image_bytes,
                    ),
                )

        # -----------------------------------------------------
        # Alguns eventos de erro podem não ter o nome "error".
        # -----------------------------------------------------

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

        return (
            "continue",
            None,
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
        # PAYLOAD CORRETO DO SPACE
        # =====================================================
        #
        # A assinatura do Space é:
        #
        # 0 = mode
        # 1 = t2i_prompt
        # 2 = i2i_prompt
        # 3 = i2i_image
        # 4 = strength
        # 5 = steps
        # 6 = guidance_scale
        # 7 = seed
        #
        # Portanto são EXATAMENTE 8 valores.
        #
        # Para T2I:
        #
        # [
        #     "t2i",
        #     prompt,
        #     "",
        #     None,
        #     strength,
        #     steps,
        #     guidance_scale,
        #     seed,
        # ]
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

        except httpx.TimeoutException:

            return ImageResult(
                False,
                self.name,
                error="sse_post_timeout",
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
            f"body={response.text[:1500]}",
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
                    f"{response.text[:1500]}"
                ),
            )

        event_id = (
            initial_data.get(
                "event_id"
            )
            if isinstance(
                initial_data,
                dict,
            )
            else None
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
        # SSE STREAM
        # =====================================================

        current_event: str | None = None
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
                # Ler SSE linha por linha
                # -------------------------------------------------

                async for raw_line in (
                    result_response.aiter_lines()
                ):

                    line = raw_line.rstrip(
                        "\r"
                    )

                    # -------------------------------------------------
                    # Fim do evento SSE
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
                # Último evento sem linha vazia.
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
            error="no_image_in_sse_response",
        )

    # =========================================================
    # LEGACY DIRECT ENDPOINT
    # =========================================================

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:
        """
        Fallback para instalações antigas do Gradio.

        O caminho principal continua sendo SSE.

        IMPORTANTE:
        Mesmo neste fallback usamos a assinatura correta
        de 8 argumentos do Space.
        """

        params = (
            self._generation_parameters(
                request
            )
        )

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
            "trying legacy direct endpoint",
            flush=True,
        )

        # -----------------------------------------------------
        # Primeiro endpoint legado
        # -----------------------------------------------------

        endpoints = [
            (
                f"{self.space_url}"
                "/run/generate_images"
            ),
            (
                f"{self.space_url}"
                "/api/predict"
            ),
        ]

        last_error = (
            "direct_endpoint_failed"
        )

        for endpoint in endpoints:

            try:

                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=self._headers(),
                )

            except httpx.HTTPError as exc:

                last_error = (
                    f"{endpoint}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                continue

            print(
                "[IMAGE] Hugging Face: "
                f"direct status="
                f"{response.status_code} "
                f"endpoint={endpoint}",
                flush=True,
            )

            if response.status_code != 200:

                last_error = (
                    f"{endpoint}: "
                    f"http_{response.status_code}: "
                    f"{response.text[:1000]}"
                )

                continue

            try:

                data = response.json()

            except json.JSONDecodeError:

                last_error = (
                    f"{endpoint}: "
                    "invalid_json_response: "
                    f"{response.text[:1000]}"
                )

                continue

            image_info = (
                self._extract_image_info(
                    data
                )
            )

            if not image_info:

                last_error = (
                    f"{endpoint}: "
                    "no_image_output: "
                    f"{self._safe_json(data)}"
                )

                continue

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

                return ImageResult(
                    True,
                    self.name,
                    image_url=image_url,
                    image_bytes=image_bytes,
                )

            last_error = (
                f"{endpoint}: "
                "image_download_failed: "
                f"url={image_url!r} "
                f"path={image_path!r}"
            )

        return ImageResult(
            False,
            self.name,
            error=last_error,
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
        # HTTP timeout
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
                # PRINCIPAL
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
                # Só usamos o endpoint antigo quando o SSE
                # não está disponível.
                # =================================================

                should_try_direct = (
                    "sse_http_404"
                    in error_text
                    or "sse_http_405"
                    in error_text
                    or "sse_post_error"
                    in error_text
                    or "sse_result_http_404"
                    in error_text
                    or "sse_result_http_405"
                    in error_text
                )

                if should_try_direct:

                    print(
                        "[IMAGE] Hugging Face: "
                        "SSE endpoint unavailable; "
                        "trying legacy direct endpoint",
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

                # -------------------------------------------------
                # Se o próprio Space retornou erro durante geração,
                # não escondemos o erro real.
                # -------------------------------------------------

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
