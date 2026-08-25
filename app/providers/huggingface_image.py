from __future__ import annotations

import json
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import requests


class HuggingFaceImageProvider:
    """
    Provider para o Space oficial da Black Forest Labs:

        black-forest-labs/FLUX.2-klein-4B

    Space:

        https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B

    Esta implementação NÃO assume que o endpoint público seja
    /infer.

    O Space oficial utiliza uma função Python chamada infer(),
    mas o Gradio pode expô-la como endpoint nomeado ou anônimo,
    dependendo da versão/configuração do Space.

    Por isso:

        1. Consultamos /gradio_api/info?all_endpoints=true
        2. Localizamos o endpoint pelos parâmetros
        3. Guardamos o nome real do endpoint
        4. Enviamos os parâmetros na ordem correta
        5. Recebemos o event_id
        6. Fazemos polling SSE
        7. Extraímos o FileData da imagem
        8. Baixamos a imagem para bytes

    Também expõe:

        provider.available

    para compatibilidade com o ImageProviderRouter do projeto.
    """

    name = "huggingface"

    DEFAULT_SPACE_URL = (
        "https://black-forest-labs-flux-2-klein-4b.hf.space"
    )

    DEFAULT_TIMEOUT = 180

    MAX_SEED = 2_147_483_647

    DEFAULT_MODE = "Distilled (4 steps)"
    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE = 1.0

    def __init__(
        self,
        space_url: str | None = None,
        timeout: int | float = DEFAULT_TIMEOUT,
        hf_token: str | None = None,
    ):
        self.space_url = (
            space_url
            or os.getenv("HF_FLUX_SPACE_URL")
            or self.DEFAULT_SPACE_URL
        ).rstrip("/")

        self.timeout = float(timeout)

        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "PamelaAI/1.0 "
                    "HuggingFaceImageProvider"
                )
            }
        )

        if self.hf_token:
            self.session.headers.update(
                {
                    "Authorization": (
                        f"Bearer {self.hf_token}"
                    )
                }
            )

        self._api_info: dict[str, Any] | None = None

        # Nome REAL do endpoint descoberto no Gradio.
        #
        # Exemplos possíveis:
        #
        #   infer
        #   0
        #   1
        #   algum_nome
        #
        self._endpoint: str | None = None

        # Parâmetros exatamente na ordem informada pelo Gradio.
        self._endpoint_parameters: list[
            dict[str, Any]
        ] = []

        # True depois que o endpoint foi descoberto.
        self._available: bool | None = None

    # =========================================================
    # COMPATIBILIDADE COM IMAGE PROVIDER ROUTER
    # =========================================================

    @property
    def available(self) -> bool:
        """
        Compatibilidade com o ImageProviderRouter.

        Importante:
        não fazemos uma chamada de rede toda vez que o router
        consulta esta propriedade.

        Se ainda não verificamos o Space, fazemos uma checagem
        leve de /gradio_api/info.
        """

        if self._available is not None:
            return self._available

        try:
            response = self.session.get(
                self._url("/gradio_api/info"),
                timeout=30,
            )

            self._available = (
                response.status_code == 200
            )

        except Exception:
            self._available = False

        return bool(self._available)

    # =========================================================
    # URL / HTTP
    # =========================================================

    def _url(
        self,
        path: str,
    ) -> str:
        if not path.startswith("/"):
            path = "/" + path

        return f"{self.space_url}{path}"

    def _request_timeout(
        self,
    ) -> tuple[float, float]:
        """
        Timeout separado para conexão e leitura.

        O Space pode estar dormindo no ZeroGPU e precisar
        de tempo para acordar.
        """

        connect_timeout = min(
            30.0,
            self.timeout,
        )

        read_timeout = max(
            60.0,
            self.timeout,
        )

        return (
            connect_timeout,
            read_timeout,
        )

    # =========================================================
    # API INFO
    # =========================================================

    def _get_api_info(
        self,
    ) -> dict[str, Any]:
        """
        Consulta a configuração pública do Gradio.

        Endpoint:

            GET /gradio_api/info?all_endpoints=true
        """

        url = self._url(
            "/gradio_api/info?all_endpoints=true"
        )

        print(
            "[IMAGE] Hugging Face: consultando "
            f"{url}",
            flush=True,
        )

        response = self.session.get(
            url,
            timeout=self._request_timeout(),
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face API info failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:3000]}"
            )

        try:
            data = response.json()

        except Exception as exc:
            raise RuntimeError(
                "Hugging Face returned invalid JSON "
                "from /gradio_api/info."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "Hugging Face returned an invalid "
                "API info object."
            )

        self._available = True

        return data

    # =========================================================
    # PARAMETER HELPERS
    # =========================================================

    @staticmethod
    def _parameter_names(
        endpoint: dict[str, Any],
    ) -> set[str]:
        """
        Extrai nomes de parâmetros de uma definição Gradio.
        """

        names: set[str] = set()

        parameters = endpoint.get(
            "parameters",
            [],
        )

        if not isinstance(
            parameters,
            list,
        ):
            return names

        for parameter in parameters:
            if not isinstance(
                parameter,
                dict,
            ):
                continue

            name = parameter.get(
                "parameter_name"
            )

            if name:
                names.add(
                    str(name)
                )

            name = parameter.get(
                "name"
            )

            if name:
                names.add(
                    str(name)
                )

        return names

    @staticmethod
    def _parameter_list(
        endpoint: dict[str, Any],
    ) -> list[dict[str, Any]]:
        parameters = endpoint.get(
            "parameters",
            [],
        )

        if not isinstance(
            parameters,
            list,
        ):
            return []

        return [
            parameter
            for parameter in parameters
            if isinstance(
                parameter,
                dict,
            )
        ]

    @staticmethod
    def _parameter_name(
        parameter: dict[str, Any],
    ) -> str | None:
        value = parameter.get(
            "parameter_name"
        )

        if value:
            return str(value)

        value = parameter.get(
            "name"
        )

        if value:
            return str(value)

        return None

    # =========================================================
    # ENDPOINT SCORING
    # =========================================================

    @classmethod
    def _score_endpoint(
        cls,
        endpoint: dict[str, Any],
    ) -> int:
        """
        Identifica o endpoint do FLUX.2 Klein.

        A assinatura atual oficial contém:

            prompt
            input_images
            mode_choice
            seed
            randomize_seed
            width
            height
            num_inference_steps
            guidance_scale
            prompt_upsampling
        """

        names = {
            name.lower()
            for name in cls._parameter_names(
                endpoint
            )
        }

        score = 0

        expected = {
            "prompt": 50,
            "input_images": 30,
            "mode_choice": 30,
            "seed": 15,
            "randomize_seed": 15,
            "width": 15,
            "height": 15,
            "num_inference_steps": 20,
            "guidance_scale": 20,
            "prompt_upsampling": 20,
        }

        for name, points in expected.items():
            if name in names:
                score += points

        # Prompt é obrigatório para identificar nosso endpoint.
        if "prompt" not in names:
            return 0

        # Esse conjunto é especialmente característico
        # do Space oficial atual.
        if (
            "width" in names
            and "height" in names
        ):
            score += 20

        if (
            "num_inference_steps" in names
            and "guidance_scale" in names
        ):
            score += 20

        return score

    # =========================================================
    # ENDPOINT DISCOVERY
    # =========================================================

    def _discover_endpoint(
        self,
    ) -> str:
        """
        Descobre o endpoint público REAL do Gradio.

        NÃO assume que seja /infer.

        O Gradio pode devolver:

            named_endpoints

        ou:

            unnamed_endpoints

        e o endpoint anônimo pode ser algo como:

            0

        Nesse caso chamaremos:

            /gradio_api/call/0

        em vez de:

            /gradio_api/call/infer
        """

        info = self._get_api_info()

        self._api_info = info

        candidates: list[
            tuple[
                int,
                str,
                dict[str, Any],
                str,
            ]
        ] = []

        # -----------------------------------------------------
        # NAMED ENDPOINTS
        # -----------------------------------------------------

        named = info.get(
            "named_endpoints",
            {},
        )

        if isinstance(
            named,
            dict,
        ):
            for endpoint_name, endpoint_data in named.items():

                if not isinstance(
                    endpoint_data,
                    dict,
                ):
                    continue

                score = self._score_endpoint(
                    endpoint_data
                )

                if score <= 0:
                    continue

                clean_name = str(
                    endpoint_name
                ).strip("/")

                candidates.append(
                    (
                        score,
                        clean_name,
                        endpoint_data,
                        "named",
                    )
                )

        # -----------------------------------------------------
        # UNNAMED ENDPOINTS
        # -----------------------------------------------------

        unnamed = info.get(
            "unnamed_endpoints",
            {},
        )

        if isinstance(
            unnamed,
            dict,
        ):
            for endpoint_name, endpoint_data in unnamed.items():

                if not isinstance(
                    endpoint_data,
                    dict,
                ):
                    continue

                score = self._score_endpoint(
                    endpoint_data
                )

                if score <= 0:
                    continue

                clean_name = str(
                    endpoint_name
                ).strip("/")

                candidates.append(
                    (
                        score,
                        clean_name,
                        endpoint_data,
                        "unnamed",
                    )
                )

        # -----------------------------------------------------
        # DEPENDENCIES
        # -----------------------------------------------------

        if not candidates:

            dependencies = info.get(
                "dependencies",
                [],
            )

            if isinstance(
                dependencies,
                list,
            ):
                for index, dependency in enumerate(
                    dependencies
                ):

                    if not isinstance(
                        dependency,
                        dict,
                    ):
                        continue

                    score = self._score_endpoint(
                        dependency
                    )

                    if score <= 0:
                        continue

                    api_name = dependency.get(
                        "api_name"
                    )

                    if api_name:
                        endpoint_name = str(
                            api_name
                        ).strip("/")

                        source = "dependency-api-name"

                    else:
                        endpoint_name = str(
                            index
                        )

                        source = "dependency-index"

                    candidates.append(
                        (
                            score,
                            endpoint_name,
                            dependency,
                            source,
                        )
                    )

        # -----------------------------------------------------
        # NENHUM ENDPOINT
        # -----------------------------------------------------

        if not candidates:
            summary = self._summarize_api_info(
                info
            )

            raise RuntimeError(
                "Não foi possível localizar o endpoint "
                "do FLUX.2 Klein 4B.\n\n"
                "Endpoints encontrados:\n"
                f"{summary}"
            )

        # Maior score primeiro.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            score,
            endpoint_name,
            endpoint_data,
            source,
        ) = candidates[0]

        self._endpoint = (
            endpoint_name
        )

        self._endpoint_parameters = (
            self._parameter_list(
                endpoint_data
            )
        )

        print(
            "[IMAGE] Hugging Face: endpoint "
            "descoberto: "
            f"/{endpoint_name} "
            f"(score={score}, source={source})",
            flush=True,
        )

        names = [
            self._parameter_name(
                parameter
            )
            for parameter in self._endpoint_parameters
        ]

        names = [
            name
            for name in names
            if name
        ]

        print(
            "[IMAGE] Hugging Face: parâmetros "
            "na ordem do Gradio: "
            + ", ".join(names),
            flush=True,
        )

        return endpoint_name

    @staticmethod
    def _summarize_api_info(
        info: dict[str, Any],
    ) -> str:
        lines: list[str] = []

        named = info.get(
            "named_endpoints",
            {},
        )

        if isinstance(
            named,
            dict,
        ):
            for name, endpoint in named.items():

                if isinstance(
                    endpoint,
                    dict,
                ):
                    params = sorted(
                        HuggingFaceImageProvider
                        ._parameter_names(
                            endpoint
                        )
                    )
                else:
                    params = []

                lines.append(
                    f"named /{name}: "
                    f"{params}"
                )

        unnamed = info.get(
            "unnamed_endpoints",
            {},
        )

        if isinstance(
            unnamed,
            dict,
        ):
            for name, endpoint in unnamed.items():

                if isinstance(
                    endpoint,
                    dict,
                ):
                    params = sorted(
                        HuggingFaceImageProvider
                        ._parameter_names(
                            endpoint
                        )
                    )
                else:
                    params = []

                lines.append(
                    f"unnamed /{name}: "
                    f"{params}"
                )

        dependencies = info.get(
            "dependencies",
            [],
        )

        if isinstance(
            dependencies,
            list,
        ):
            for index, dependency in enumerate(
                dependencies
            ):
                if not isinstance(
                    dependency,
                    dict,
                ):
                    continue

                params = sorted(
                    HuggingFaceImageProvider
                    ._parameter_names(
                        dependency
                    )
                )

                api_name = dependency.get(
                    "api_name"
                )

                lines.append(
                    "dependency "
                    f"{index}"
                    + (
                        f" api_name=/{api_name}"
                        if api_name
                        else ""
                    )
                    + f": {params}"
                )

        if not lines:
            return "(nenhum endpoint encontrado)"

        return "\n".join(
            lines
        )

    # =========================================================
    # FILE UPLOAD
    # =========================================================

    def _upload_file(
        self,
        path: str | Path,
    ) -> str:
        """
        Upload de imagem para o Gradio.

        Endpoint:

            POST /gradio_api/upload
        """

        file_path = Path(
            path
        )

        if not file_path.exists():
            raise FileNotFoundError(
                "Imagem de referência não encontrada: "
                f"{file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                "O caminho de referência não é "
                f"um arquivo: {file_path}"
            )

        mime_type, _ = mimetypes.guess_type(
            file_path.name
        )

        if not mime_type:
            mime_type = (
                "application/octet-stream"
            )

        print(
            "[IMAGE] Hugging Face: upload "
            f"da referência: {file_path}",
            flush=True,
        )

        with file_path.open(
            "rb"
        ) as file_handle:

            response = self.session.post(
                self._url(
                    "/gradio_api/upload"
                ),
                files={
                    "files": (
                        file_path.name,
                        file_handle,
                        mime_type,
                    )
                },
                timeout=self._request_timeout(),
            )

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face file upload failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:3000]}"
            )

        try:
            uploaded = response.json()

        except Exception as exc:
            raise RuntimeError(
                "Hugging Face upload returned "
                "invalid JSON."
            ) from exc

        if not isinstance(
            uploaded,
            list,
        ) or not uploaded:
            raise RuntimeError(
                "Hugging Face upload returned "
                "no file path."
            )

        uploaded_path = uploaded[0]

        if isinstance(
            uploaded_path,
            dict,
        ):
            uploaded_path = (
                uploaded_path.get("path")
                or uploaded_path.get("url")
            )

        if not uploaded_path:
            raise RuntimeError(
                "Hugging Face upload returned "
                "an empty file path."
            )

        return str(
            uploaded_path
        )

    # =========================================================
    # INPUT IMAGE
    # =========================================================

    @staticmethod
    def _file_data(
        path: str,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        """
        FileData compatível com Gradio.
        """

        return {
            "path": path,
            "url": None,
            "size": None,
            "orig_name": (
                original_name
                or Path(path).name
            ),
            "mime_type": None,
            "is_stream": False,
            "meta": {
                "_type": "gradio.FileData"
            },
        }

    def _prepare_input_images(
        self,
        reference_images: Any,
    ) -> list[
        list[Any]
    ]:
        """
        O componente input_images do Space oficial é:

            gr.Gallery(type="pil")

        Portanto, na API o Gallery recebe uma lista
        de itens, normalmente representados como:

            [
                [FileData, caption]
            ]

        Retornamos exatamente essa estrutura.
        """

        if not reference_images:
            return []

        if isinstance(
            reference_images,
            (str, Path),
        ):
            reference_images = [
                reference_images
            ]

        if not isinstance(
            reference_images,
            (list, tuple),
        ):
            return []

        result: list[
            list[Any]
        ] = []

        for item in reference_images:

            if not item:
                continue

            # -------------------------------------------------
            # Path
            # -------------------------------------------------

            if isinstance(
                item,
                Path,
            ):
                uploaded = self._upload_file(
                    item
                )

                result.append(
                    [
                        self._file_data(
                            uploaded,
                            item.name,
                        ),
                        None,
                    ]
                )

                continue

            # -------------------------------------------------
            # String
            # -------------------------------------------------

            if isinstance(
                item,
                str,
            ):
                if item.startswith(
                    "http://"
                ) or item.startswith(
                    "https://"
                ):
                    result.append(
                        [
                            self._file_data(
                                item,
                                Path(
                                    item.split("?")[0]
                                ).name,
                            ),
                            None,
                        ]
                    )

                    continue

                local_path = Path(
                    item
                )

                if local_path.exists():
                    uploaded = (
                        self._upload_file(
                            local_path
                        )
                    )

                    result.append(
                        [
                            self._file_data(
                                uploaded,
                                local_path.name,
                            ),
                            None,
                        ]
                    )

                    continue

            # -------------------------------------------------
            # Estrutura já serializada
            # -------------------------------------------------

            if isinstance(
                item,
                dict,
            ):
                if item.get(
                    "path"
                ):
                    result.append(
                        [
                            item,
