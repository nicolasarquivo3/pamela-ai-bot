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

    def _headers(self) -> dict:
        headers = {
            "Accept": "application/json",
        }

        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        return headers

    @staticmethod
    def _extract_output(data):
        """
        Extrai a primeira imagem da resposta do endpoint
        /run/generate_images.

        O OpenAPI do Space informa que:
            output   = GalleryData
            output_1 = string
        """

        if not isinstance(data, dict):
            return None

        output = data.get("output")

        if output is None:
            return None

        # GalleryData
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                image = item.get("image")

                if not isinstance(image, dict):
                    continue

                image_url = image.get("url")
                image_path = image.get("path")

                if image_url:
                    return {
                        "url": image_url,
                        "path": image_path,
                    }

                if image_path:
                    return {
                        "url": None,
                        "path": image_path,
                    }

        # Alguns wrappers podem devolver diretamente um objeto de imagem.
        if isinstance(output, dict):
            image_url = output.get("url")
            image_path = output.get("path")

            if image_url or image_path:
                return {
                    "url": image_url,
                    "path": image_path,
                }

            nested_image = output.get("image")

            if isinstance(nested_image, dict):
                image_url = nested_image.get("url")
                image_path = nested_image.get("path")

                if image_url or image_path:
                    return {
                        "url": image_url,
                        "path": image_path,
                    }

        return None

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        image_url: str | None,
        image_path: str | None,
    ) -> bytes | None:

        candidates = []

        # -----------------------------------------------------
        # URL fornecida diretamente pelo Gradio
        # -----------------------------------------------------

        if image_url:
            if image_url.startswith("http://"):
                candidates.append(image_url)

            elif image_url.startswith("https://"):
                candidates.append(image_url)

            elif image_url.startswith("/"):
                candidates.append(
                    f"{self.space_url}{image_url}"
                )

        # -----------------------------------------------------
        # Path retornado pelo Gradio
        # -----------------------------------------------------

        if image_path:

            if image_path.startswith("http://"):
                candidates.append(image_path)

            elif image_path.startswith("https://"):
                candidates.append(image_path)

            else:
                encoded_path = quote(
                    image_path,
                    safe="",
                )

                candidates.append(
                    f"{self.space_url}/gradio_api/file={encoded_path}"
                )

                candidates.append(
                    f"{self.space_url}/file={encoded_path}"
                )

                if image_path.startswith("/"):
                    candidates.append(
                        f"{self.space_url}{image_path}"
                    )

        # Remove duplicados mantendo a ordem.
        candidates = list(dict.fromkeys(candidates))

        for candidate in candidates:

            try:
                response = await client.get(
                    candidate,
                    headers=self._headers(),
                )

                if response.status_code == 200:
                    content_type = response.headers.get(
                        "content-type",
                        "",
                    ).lower()

                    # Aceitamos imagens mesmo que o servidor
                    # não tenha informado corretamente o MIME type.
                    if (
                        content_type.startswith("image/")
                        or response.content[:8] == b"\x89PNG\r\n\x1a\n"
                        or response.content[:2] == b"\xff\xd8"
                        or response.content[:4] == b"RIFF"
                    ):
                        return response.content

            except Exception:
                continue

        return None

    async def _generate_direct(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        # -----------------------------------------------------
        # Valores padrão compatíveis com o OpenAPI 6.3.0
        # -----------------------------------------------------

        style_preset = "Photoreal / Fotorrealista"
        aspect_ratio = "Portrait 4:5 / Retrato 4:5"
        steps = 28
        guidance_scale = 4.0
        seed = -1
        strength = 0.0

        # -----------------------------------------------------
        # Ajuste de parâmetros a partir do ImageRequest
        # -----------------------------------------------------

        if request is not None:

            width = getattr(request, "width", 1024)
            height = getattr(request, "height", 1024)

            if width > height:
                if width / max(height, 1) >= 1.6:
                    aspect_ratio = "Wide 16:9 / Panoramico 16:9"
                else:
                    aspect_ratio = "Landscape 5:4 / Horizontal 5:4"

            elif height > width:
                if height / max(width, 1) >= 1.6:
                    aspect_ratio = "Tall 9:16 / Vertical 9:16"
                else:
                    aspect_ratio = "Portrait 4:5 / Retrato 4:5"

            else:
                aspect_ratio = "Square 1:1 / Cuadrado 1:1"

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

        # -----------------------------------------------------
        # IMPORTANTE:
        #
        # O OpenAPI mostra que /run/generate_images recebe
        # um objeto JSON, NÃO {"data": [...]}
        # -----------------------------------------------------

        payload = {
            "t2i_prompt": prompt,
            "i2i_prompt": "",
            "i2i_image": None,
            "strength": strength,
            "style_preset": style_preset,
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        response = await client.post(
            f"{self.space_url}/run/generate_images",
            json=payload,
            headers=self._headers(),
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

        image_info = self._extract_output(data)

        if not image_info:
            return ImageResult(
                False,
                self.name,
                error=(
                    "no_image_output: "
                    f"{json.dumps(data, ensure_ascii=False)[:1500]}"
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

    async def _generate_sse(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        request=None,
    ) -> ImageResult:

        """
        Fallback para instalações do Gradio que não retornem
        diretamente pelo /run/generate_images.

        Aqui também usamos o formato correto de argumentos
        do Space.
        """

        payload = {
            "data": [
                prompt,
                "",
                None,
                0.0,
                "Photoreal / Fotorrealista",
                "Portrait 4:5 / Retrato 4:5",
                28,
                4.0,
                -1,
            ]
        }

        response = await client.post(
            f"{self.space_url}/gradio_api/call/generate_images",
            json=payload,
            headers=self._headers(),
        )

        if response.status_code != 200:
            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_http_{response.status_code}: "
                    f"{response.text[:1000]}"
                ),
            )

        try:
            data = response.json()
        except json.JSONDecodeError:
            return ImageResult(
                False,
                self.name,
                error="sse_invalid_initial_response",
            )

        event_id = data.get("event_id")

        if not event_id:
            return ImageResult(
                False,
                self.name,
                error="sse_no_event_id",
            )

        result_response = await client.get(
            f"{self.space_url}/gradio_api/call/"
            f"generate_images/{event_id}",
            headers=self._headers(),
        )

        if result_response.status_code != 200:
            return ImageResult(
                False,
                self.name,
                error=(
                    f"sse_result_http_"
                    f"{result_response.status_code}: "
                    f"{result_response.text[:1000]}"
                ),
            )

        # -----------------------------------------------------
        # O SSE pode devolver:
        #
        # event: generating
        # data: ...
        #
        # event: complete
        # data: ...
        #
        # ou event: error
        # -----------------------------------------------------

        for line in result_response.text.splitlines():

            line = line.strip()

            if not line.startswith("data:"):
                continue

            raw = line[5:].strip()

            if not raw:
                continue

            try:
                event_data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # ---------------------------------------------
            # Caso o evento seja erro
            # ---------------------------------------------

            if isinstance(event_data, dict):
                if event_data.get("error"):
                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            "sse_generation_error: "
                            f"{event_data.get('error')}"
                        ),
                    )

            # ---------------------------------------------
            # Caso seja lista de resultados
            # ---------------------------------------------

            if isinstance(event_data, list):

                for item in event_data:

                    if not isinstance(item, dict):
                        continue

                    image = item.get("image")

                    if not isinstance(image, dict):
                        continue

                    image_url = image.get("url")
                    image_path = image.get("path")

                    image_bytes = await self._download_image(
                        client,
                        image_url,
                        image_path,
                    )

                    if image_bytes:
                        return ImageResult(
                            True,
                            self.name,
                            image_url=image_url,
                            image_bytes=image_bytes,
                        )

            # ---------------------------------------------
            # Algumas versões podem devolver diretamente
            # o objeto de imagem.
            # ---------------------------------------------

            if isinstance(event_data, dict):

                image = event_data.get("image")

                if isinstance(image, dict):

                    image_url = image.get("url")
                    image_path = image.get("path")

                    image_bytes = await self._download_image(
                        client,
                        image_url,
                        image_path,
                    )

                    if image_bytes:
                        return ImageResult(
                            True,
                            self.name,
                            image_url=image_url,
                            image_bytes=image_bytes,
                        )

        return ImageResult(
            False,
            self.name,
            error="no_image_in_sse_response",
        )

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
                # Usa exatamente o endpoint documentado pelo
                # OpenAPI fornecido pelo usuário.
                # -------------------------------------------------

                result = await self._generate_direct(
                    client,
                    prompt,
                    request,
                )

                if result.success:
                    return result

                # -------------------------------------------------
                # Não fazemos fallback silencioso para qualquer
                # erro. O fallback SSE só é útil quando o
                # endpoint direto não consegue produzir uma
                # imagem por incompatibilidade da implementação.
                # -------------------------------------------------

                direct_error = result.error or ""

                if (
                    "http_404" in direct_error
                    or "405" in direct_error
                    or "invalid_json_response" in direct_error
                ):
                    sse_result = await self._generate_sse(
                        client,
                        prompt,
                        request,
                    )

                    if sse_result.success:
                        return sse_result

                    return ImageResult(
                        False,
                        self.name,
                        error=(
                            f"direct={direct_error}; "
                            f"sse={sse_result.error}"
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
                error=f"http_client_error: {exc}",
            )

        except Exception as exc:
            return ImageResult(
                False,
                self.name,
                error=(
                    f"unexpected_error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
