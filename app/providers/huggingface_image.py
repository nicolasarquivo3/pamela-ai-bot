import asyncio
import base64
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

    Space:
        https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B

    API:
        /infer

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

    A implementação tenta utilizar gradio_client quando disponível,
    pois isso reproduz a chamada que o próprio ecossistema Gradio usa.

    Caso gradio_client não esteja disponível no ambiente do Render,
    existe fallback HTTP para a API /gradio_api/call/infer.
    """

    name = "huggingface"

    DEFAULT_SPACE_URL = (
        "https://huggingface.co/spaces/"
        "black-forest-labs/FLUX.2-klein-4B"
    )

    DEFAULT_API_NAME = "/infer"

    DISTILLED_MODE = "Distilled (4 steps)"
    BASE_MODE = "Base (50 steps)"

    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 256

    DEFAULT_STEPS_DISTILLED = 4
    DEFAULT_STEPS_BASE = 50

    DEFAULT_CFG_DISTILLED = 1.0
    DEFAULT_CFG_BASE = 4.0

    DEFAULT_SEED = 42

    def __init__(
        self,
        space_url: str,
        timeout: int = 180,
        hf_token: str | None = None,
    ):
        self.space_url = (
            space_url or self.DEFAULT_SPACE_URL
        ).rstrip("/")

        self.timeout = int(timeout or 180)

        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    async def available(self) -> bool:
        return bool(self.space_url)

    # =========================================================
    # HEADERS
    # =========================================================

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "pamela-ai/1.0",
            "Accept": "application/json",
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

        headers["Cache-Control"] = "no-cache"

        return headers

    # =========================================================
    # SAFE SERIALIZATION
    # =========================================================

    @staticmethod
    def _safe_json(
        value,
        limit: int = 4000,
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
    def _safe_text(
        value,
        limit: int = 4000,
    ) -> str:
        try:
            text = str(value)
        except Exception:
            text = repr(value)

        return text[:limit]

    # =========================================================
    # GENERATION PARAMETERS
    # =========================================================

    @classmethod
    def _generation_parameters(
        cls,
        request=None,
    ) -> dict:
        """
        Monta parâmetros compatíveis com o Space oficial.

        O Space atual possui dois modos:

            Distilled (4 steps)
                steps = 4
                CFG   = 1.0

            Base (50 steps)
                steps = 50
                CFG   = 4.0
        """

        mode = cls.DISTILLED_MODE

        width = 1024
        height = 1024

        seed = cls.DEFAULT_SEED
        randomize_seed = False

        prompt_upsampling = False

        if request is not None:

            requested_mode = getattr(
                request,
                "mode_choice",
                None,
            )

            if requested_mode:
                requested_mode = str(
                    requested_mode
                ).strip()

                if requested_mode in (
                    cls.DISTILLED_MODE,
                    cls.BASE_MODE,
                ):
                    mode = requested_mode

            requested_width = getattr(
                request,
                "width",
                1024,
            )

            requested_height = getattr(
                request,
                "height",
                1024,
            )

            try:
                width = int(
                    requested_width
                )
            except (
                TypeError,
                ValueError,
            ):
                width = 1024

            try:
                height = int(
                    requested_height
                )
            except (
                TypeError,
                ValueError,
            ):
                height = 1024

            requested_seed = getattr(
                request,
                "seed",
                cls.DEFAULT_SEED,
            )

            try:
                seed = int(
                    requested_seed
                )
            except (
                TypeError,
                ValueError,
            ):
                seed = cls.DEFAULT_SEED

            requested_randomize = getattr(
                request,
                "randomize_seed",
                False,
            )

            randomize_seed = bool(
                requested_randomize
            )

            requested_upsampling = getattr(
                request,
                "prompt_upsampling",
                False,
            )

            prompt_upsampling = bool(
                requested_upsampling
            )

        # -----------------------------------------------------
        # Limites reais do Space
        # -----------------------------------------------------

        width = max(
            cls.MIN_IMAGE_SIZE,
            min(
                cls.MAX_IMAGE_SIZE,
                width,
            ),
        )

        height = max(
            cls.MIN_IMAGE_SIZE,
            min(
                cls.MAX_IMAGE_SIZE,
                height,
            ),
        )

        # -----------------------------------------------------
        # FLUX exige dimensões múltiplas de 8
        # -----------------------------------------------------

        width = max(
            cls.MIN_IMAGE_SIZE,
            min(
                cls.MAX_IMAGE_SIZE,
                round(width / 8) * 8,
            ),
        )

        height = max(
            cls.MIN_IMAGE_SIZE,
            min(
                cls.MAX_IMAGE_SIZE,
                round(height / 8) * 8,
            ),
        )

        # -----------------------------------------------------
        # Parâmetros determinados pelo modo
        # -----------------------------------------------------

        if mode == cls.BASE_MODE:

            steps = cls.DEFAULT_STEPS_BASE
            guidance_scale = (
                cls.DEFAULT_CFG_BASE
            )

        else:

            mode = cls.DISTILLED_MODE

            steps = (
                cls.DEFAULT_STEPS_DISTILLED
            )

            guidance_scale = (
                cls.DEFAULT_CFG_DISTILLED
            )

        return {
            "mode_choice": mode,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "prompt_upsampling": (
                prompt_upsampling
            ),
        }

    # =========================================================
    # IMAGE INPUT HELPERS
    # =========================================================

    @staticmethod
    def _get_request_reference_path(
        request,
    ) -> str | None:
        """
        Tenta localizar uma imagem de referência caso o objeto
        request possua algum destes atributos.

        Isso é opcional.

        O fluxo normal do bot continua funcionando sem referência.
        """

        if request is None:
            return None

        possible_names = (
            "reference_image_path",
            "input_image_path",
            "image_path",
            "face_reference_image_path",
        )

        for name in possible_names:

            value = getattr(
                request,
                name,
                None,
            )

            if not value:
                continue

            value = str(value).strip()

            if value:
                return value

        return None

    @staticmethod
    def _find_reference_path(
        request,
    ) -> str | None:
        path = (
            HuggingFaceImageProvider
            ._get_request_reference_path(
                request
            )
        )

        if not path:
            return None

        candidates = [
            path,
        ]

        if not os.path.isabs(path):
            candidates.extend(
                [
                    os.path.join(
                        "/app",
                        path,
                    ),
                    os.path.join(
                        os.getcwd(),
                        path,
                    ),
                ]
            )

        for candidate in candidates:

            try:
                if os.path.isfile(candidate):
                    return candidate
            except OSError:
                continue

        return None

    @staticmethod
    def _image_to_data_uri(
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        encoded = base64.b64encode(
            image_bytes
        ).decode("ascii")

        return (
            f"data:{mime_type};base64,"
            f"{encoded}"
        )

    async def _read_reference_image(
        self,
        path: str | None,
    ) -> bytes | None:

        if not path:
            return None

        try:

            return await asyncio.to_thread(
                Path(path).read_bytes
            )

        except (
            OSError,
            IOError,
        ):

            return None

    # =========================================================
    # GRADIO CLIENT
    # =========================================================

    @staticmethod
    def _gradio_client_available():
        try:
            from gradio_client import Client

            return Client

        except ImportError:

            return None

    @staticmethod
    def _extract_gradio_result(
        result,
    ):
        """
        O Space retorna:

            (image, seed)

        O primeiro item normalmente é um arquivo
        representado por path ou FileData.
        """

        if isinstance(
            result,
            tuple,
        ):
            if len(result) == 0:
                return None, None

            image_result = result[0]

            used_seed = (
                result[1]
                if len(result) > 1
                else None
            )

            return (
                image_result,
                used_seed,
            )

        if isinstance(
            result,
            list,
        ):

            if not result:
                return None, None

            image_result = result[0]

            used_seed = (
                result[1]
                if len(result) > 1
                else None
            )

            return (
                image_result,
                used_seed,
            )

        return result, None

    @staticmethod
    def _extract_path_from_gradio_file(
        value,
    ) -> str | None:

        if value is None:
            return None

        if isinstance(
            value,
            os.PathLike,
        ):
            return os.fspath(value)

        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            for key in (
                "path",
                "url",
                "image",
                "file",
            ):

                candidate = value.get(
                    key
                )

                if candidate:

                    if isinstance(
                        candidate,
                        dict,
                    ):
                        nested = (
                            HuggingFaceImageProvider
                            ._extract_path_from_gradio_file(
                                candidate
                            )
                        )

                        if nested:
                            return nested

                    else:
                        return str(
                            candidate
                        )

        return None

    async def _read_file_url(
        self,
        client: httpx.AsyncClient,
        value,
    ) -> tuple[bytes | None, str | None]:

        path = (
            self._extract_path_from_gradio_file(
                value
            )
        )

        if not path:
            return None, None

        # -----------------------------------------------------
        # Arquivo local retornado pelo Gradio client
        # -----------------------------------------------------

        if os.path.isfile(path):

            try:

                data = await asyncio.to_thread(
                    Path(path).read_bytes
                )

                return data, None

            except (
                OSError,
                IOError,
            ):

                pass

        # -----------------------------------------------------
        # URL direta
        # -----------------------------------------------------

        if path.startswith(
            (
                "http://",
                "https://",
            )
        ):

            try:

                response = await client.get(
                    path,
                    headers=self._headers(),
                )

                if response.status_code == 200:

                    return (
                        response.content,
                        path,
                    )

            except httpx.HTTPError:
                pass

            return None, path

        # -----------------------------------------------------
        # Path relativo do Gradio
        # -----------------------------------------------------

        normalized = path.lstrip("/")

        candidates = [
            (
                f"{self.space_url}"
                f"/gradio_api/file={quote(normalized, safe='')}"
            ),
            (
                f"{self.space_url}"
                f"/file={quote(normalized, safe='')}"
            ),
            (
                f"{self.space_url}/{normalized}"
            ),
        ]

        for candidate in candidates:

            try:

                response = await client.get(
                    candidate,
                    headers=self._headers(),
                )

                if response.status_code != 200:
                    continue

                content = response
