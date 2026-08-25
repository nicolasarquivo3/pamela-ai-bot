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
    Provider de imagens para o Space oficial:

        black-forest-labs/FLUX.2-klein-4B

    O provider NÃO assume que o endpoint seja /infer.

    Ele consulta dinamicamente:

        /gradio_api/info?all_endpoints=true

    e encontra o endpoint compatível pelos nomes dos parâmetros.

    Isso evita o erro:

        FnIndexInferError:
        Could not infer function index for API name: infer

    O Space oficial atualmente utiliza a função infer() com os
    parâmetros do FLUX.2 Klein 4B, mas o Gradio pode expô-la de
    maneira diferente conforme a versão/configuração do Space.
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
        self._endpoint: str | None = None
        self._endpoint_parameters: list[dict[str, Any]] = []

    # =========================================================
    # URL / HTTP HELPERS
    # =========================================================

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path

        return f"{self.space_url}{path}"

    def _request_timeout(self) -> tuple[float, float]:
        """
        Timeout separado para conexão e leitura.

        O Space pode precisar de algum tempo para acordar a
        ZeroGPU / carregar o modelo.
        """

        connect_timeout = min(30.0, self.timeout)

        read_timeout = max(
            60.0,
            self.timeout,
        )

        return connect_timeout, read_timeout

    # =========================================================
    # API DISCOVERY
    # =========================================================

    def _get_api_info(self) -> dict[str, Any]:
        """
        Consulta a API atual do Gradio.

        Endpoint:

            GET /gradio_api/info?all_endpoints=true

        Não usamos /infer diretamente porque o nome público
        do endpoint pode mudar.
        """

        response = self.session.get(
            self._url(
                "/gradio_api/info?all_endpoints=true"
            ),
            timeout=self._request_timeout(),
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face API info failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face returned invalid JSON from "
                "/gradio_api/info"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Hugging Face returned an invalid API info object."
            )

        return data

    @staticmethod
    def _parameter_names(
        endpoint: dict[str, Any],
    ) -> set[str]:
        """
        Extrai os nomes dos parâmetros de um endpoint.

        Gradio pode representar os parâmetros de maneiras
        ligeiramente diferentes entre versões.
        """

        names: set[str] = set()

        parameters = endpoint.get(
            "parameters",
            [],
        )

        if isinstance(parameters, list):
            for parameter in parameters:
                if not isinstance(parameter, dict):
                    continue

                name = parameter.get("parameter_name")

                if name:
                    names.add(str(name))

                name = parameter.get("name")

                if name:
                    names.add(str(name))

        return names

    @staticmethod
    def _score_endpoint(
        endpoint: dict[str, Any],
    ) -> int:
        """
        Calcula o quanto um endpoint parece ser o infer()
        do FLUX.2 Klein 4B.

        Quanto maior, melhor.
        """

        names = {
            name.lower()
            for name in HuggingFaceImageProvider._parameter_names(
                endpoint
            )
        }

        score = 0

        required = {
            "prompt",
            "input_images",
            "mode_choice",
            "seed",
            "randomize_seed",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "prompt_upsampling",
        }

        for name in required:
            if name in names:
                score += 10

        # Alguns endpoints podem não expor input_images
        # de maneira tradicional quando não há imagem.
        if "prompt" in names:
            score += 30

        if "mode_choice" in names:
            score += 20

        if "guidance_scale" in names:
            score += 10

        return score

    def _discover_endpoint(self) -> str:
        """
        Descobre automaticamente o endpoint do FLUX.2 Klein.

        Retorna uma string adequada para:

            /gradio_api/call/v2/{endpoint}
        """

        info = self._get_api_info()

        self._api_info = info

        candidates: list[
            tuple[int, str, dict[str, Any]]
        ] = []

        # -----------------------------------------------------
        # ENDPOINTS NOMEADOS
        # -----------------------------------------------------

        named = info.get(
            "named_endpoints",
            {},
        )

        if isinstance(named, dict):
            for endpoint_name, endpoint_data in named.items():
                if not isinstance(endpoint_data, dict):
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
                    )
                )

        # -----------------------------------------------------
        # ENDPOINTS NÃO NOMEADOS
        # -----------------------------------------------------

        unnamed = info.get(
            "unnamed_endpoints",
            {},
        )

        if isinstance(unnamed, dict):
            for endpoint_id, endpoint_data in unnamed.items():
                if not isinstance(endpoint_data, dict):
                    continue

                score = self._score_endpoint(
                    endpoint_data
                )

                if score <= 0:
                    continue

                candidates.append(
                    (
                        score,
                        str(endpoint_id),
                        endpoint_data,
                    )
                )

        if not candidates:
            # Última tentativa usando dependencies, porque
            # algumas versões do Gradio expõem essa informação
            # de forma diferente.
            dependencies = info.get(
                "dependencies",
                [],
            )

            if isinstance(dependencies, list):
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

                    endpoint_name = dependency.get(
                        "api_name"
                    )

                    if endpoint_name:
                        endpoint_name = str(
                            endpoint_name
                        ).strip("/")

                    else:
                        endpoint_name = str(index)

                    candidates.append(
                        (
                            score,
                            endpoint_name,
                            dependency,
                        )
                    )

        if not candidates:
            available = self._summarize_api_info(info)

            raise RuntimeError(
                "Não foi possível localizar o endpoint "
                "do FLUX.2 Klein 4B no Space oficial.\n"
                f"Endpoints encontrados:\n{available}"
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        score, endpoint_name, endpoint_data = (
            candidates[0]
        )

        self._endpoint = endpoint_name

        parameters = endpoint_data.get(
            "parameters",
            [],
        )

        if isinstance(parameters, list):
            self._endpoint_parameters = parameters

        else:
            self._endpoint_parameters = []

        print(
            "[IMAGE] Hugging Face: endpoint descoberto: "
            f"/{endpoint_name} "
            f"(score={score})",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: parâmetros: "
            + ", ".join(
                sorted(
                    self._parameter_names(
                        endpoint_data
                    )
                )
            ),
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

        if isinstance(named, dict):
            for name, endpoint in named.items():
                parameters = (
                    HuggingFaceImageProvider
                    ._parameter_names(endpoint)
                    if isinstance(
                        endpoint,
                        dict,
                    )
                    else set()
                )

                lines.append(
                    f"/{name}: "
                    f"{sorted(parameters)}"
                )

        unnamed = info.get(
            "unnamed_endpoints",
            {},
        )

        if isinstance(unnamed, dict):
            for name, endpoint in unnamed.items():
                parameters = (
                    HuggingFaceImageProvider
                    ._parameter_names(endpoint)
                    if isinstance(
                        endpoint,
                        dict,
                    )
                    else set()
                )

                lines.append(
                    f"/{name}: "
                    f"{sorted(parameters)}"
                )

        if not lines:
            return "(nenhum endpoint encontrado)"

        return "\n".join(lines)

    # =========================================================
    # FILE UPLOAD
    # =========================================================

    def _upload_file(
        self,
        path: str | Path,
    ) -> str:
        """
        Faz upload de uma imagem para o armazenamento temporário
        do Space.

        Gradio 6 utiliza:

            POST /gradio_api/upload

        com multipart/form-data.
        """

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Imagem de referência não encontrada: "
                f"{file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"O caminho de referência não é um arquivo: "
                f"{file_path}"
            )

        mime_type, _ = mimetypes.guess_type(
            file_path.name
        )

        if not mime_type:
            mime_type = "application/octet-stream"

        print(
            "[IMAGE] Hugging Face: enviando imagem "
            f"de referência: {file_path}",
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
                f"{response.text[:2000]}"
            )

        try:
            uploaded = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face upload returned invalid JSON."
            ) from exc

        if not isinstance(
            uploaded,
            list,
        ) or not uploaded:
            raise RuntimeError(
                "Hugging Face upload returned no file path."
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
                "Hugging Face upload returned an empty "
                "file path."
            )

        return str(uploaded_path)

    # =========================================================
    # INPUT IMAGE SERIALIZATION
    # =========================================================

    @staticmethod
    def _file_data(
        path: str,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Converte um caminho retornado pelo Gradio upload
        em FileData.
        """

        return {
            "path": path,
            "meta": {
                "_type": "gradio.FileData"
            },
            "orig_name": (
                original_name
                or Path(path).name
            ),
        }

    def _prepare_input_images(
        self,
        reference_images: Any,
    ) -> list[dict[str, Any]]:
        """
        Converte reference_images para a estrutura esperada
        pelo Gallery do Space oficial.

        Aceita:
            - string
            - Path
            - lista/tupla de strings
            - lista/tupla de Paths
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
            dict[str, Any]
        ] = []

        for item in reference_images:
            if not item:
                continue

            # -------------------------------------------------
            # Caminho local
            # -------------------------------------------------

            if isinstance(
                item,
                Path,
            ):
                uploaded = self._upload_file(
                    item
                )

                result.append(
                    self._file_data(
                        uploaded,
                        item.name,
                    )
                )

                continue

            if isinstance(
                item,
                str,
            ):
                # URL pública
                if item.startswith(
                    "http://"
                ) or item.startswith(
                    "https://"
                ):
                    result.append(
                        {
                            "path": item,
                            "meta": {
                                "_type": (
                                    "gradio.FileData"
                                )
                            },
                            "orig_name": Path(
                                item.split("?")[0]
                            ).name,
                        }
                    )

                    continue

                local_path = Path(item)

                if local_path.exists():
                    uploaded = self._upload_file(
                        local_path
                    )

                    result.append(
                        self._file_data(
                            uploaded,
                            local_path.name,
                        )
                    )

                    continue

            # -------------------------------------------------
            # Estrutura já serializada
            # -------------------------------------------------

            if isinstance(
                item,
                dict,
            ):
                if item.get("path"):
                    result.append(item)

        return result

    # =========================================================
    # ENDPOINT PAYLOAD
    # =========================================================

    def _build_payload(
        self,
        prompt: str,
        request: Any | None = None,
    ) -> dict[str, Any]:
        """
        Monta os parâmetros do endpoint oficial.

        A estrutura principal é baseada diretamente no infer()
        atual do Space oficial:
        
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

        width = self._get_request_value(
            request,
            "width",
            1024,
        )

        height = self._get_request_value(
            request,
            "height",
            1024,
        )

        seed = self._get_request_value(
            request,
            "seed",
            42,
        )

        mode_choice = self._get_request_value(
            request,
            "mode_choice",
            self.DEFAULT_MODE,
        )

        steps = self._get_request_value(
            request,
            "num_inference_steps",
            self.DEFAULT_STEPS,
        )

        guidance = self._get_request_value(
            request,
            "guidance_scale",
            self.DEFAULT_GUIDANCE,
        )

        randomize_seed = self._get_request_value(
            request,
            "randomize_seed",
            False,
        )

        prompt_upsampling = self._get_request_value(
            request,
            "prompt_upsampling",
            False,
        )

        reference_images = self._get_request_value(
            request,
            "reference_images",
            None,
        )

        input_images = (
            self._prepare_input_images(
                reference_images
            )
        )

        if randomize_seed:
            seed = random.randint(
                0,
                self.MAX_SEED,
            )

        # -----------------------------------------------------
        # Sanitização
        # -----------------------------------------------------

        try:
            width = int(width)
        except Exception:
            width = 1024

        try:
            height = int(height)
        except Exception:
            height = 1024

        try:
            steps = int(steps)
        except Exception:
            steps = self.DEFAULT_STEPS

        try:
            guidance = float(guidance)
        except Exception:
            guidance = self.DEFAULT_GUIDANCE

        try:
            seed = int(seed)
        except Exception:
            seed = 42

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

        # FLUX.2 Klein aceita dimensões múltiplas de 8.
        width = (
            round(width / 8) * 8
        )

        height = (
            round(height / 8) * 8
        )

        steps = max(
            1,
            min(
                100,
                steps,
            ),
        )

        guidance = max(
            0.0,
            min(
                10.0,
                guidance,
            ),
        )

        seed = max(
            0,
            min(
                self.MAX_SEED,
                seed,
            ),
        )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        final_prompt = str(
            prompt or ""
        ).strip()

        # -----------------------------------------------------
        # Payload oficial
        # -----------------------------------------------------

        payload = {
            "prompt": final_prompt,
            "input_images": input_images,
            "mode_choice": mode_choice,
            "seed": seed,
            "randomize_seed": bool(
                randomize_seed
            ),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt_upsampling": bool(
                prompt_upsampling
            ),
        }

        # -----------------------------------------------------
        # Se descobrirmos que o endpoint atual não possui
        # determinado campo, removemos esse campo.
        # -----------------------------------------------------

        if self._endpoint_parameters:
            available = {
                name
                for name in (
                    self._parameter_names(
                        {
                            "parameters": (
                                self._endpoint_parameters
                            )
                        }
                    )
                )
            }

            if available:
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key in available
                }

        return payload

    @staticmethod
    def _get_request_value(
        request: Any | None,
        name: str,
        default: Any,
    ) -> Any:
        """
        Lê um atributo de ImageRequest sem assumir se é
        dataclass, Pydantic ou objeto comum.
        """

        if request is None:
            return default

        if isinstance(
            request,
            dict,
        ):
            value = request.get(name)

            if value is None:
                return default

            return value

        try:
            value = getattr(
                request,
                name,
            )
        except Exception:
            return default

        if value is None:
            return default

        return value

    # =========================================================
    # API CALL
    # =========================================================

    def _call_generation(
        self,
        payload: dict[str, Any],
    ) -> str:
        """
        Executa a geração através do endpoint v2 do Gradio.

        POST:

            /gradio_api/call/v2/{endpoint}

        Depois:

            GET /gradio_api/call/{endpoint}/{event_id}

        O uso de /call/v2 evita depender da antiga estrutura
        de payload {"data": [...]}.
        """

        if not self._endpoint:
            self._discover_endpoint()

        assert self._endpoint is not None

        endpoint = self._endpoint.strip("/")

        print(
            "[IMAGE] Hugging Face: POST "
            f"/gradio_api/call/v2/{endpoint}",
            flush=True,
        )

        print(
            "[IMAGE] Hugging Face: payload="
            + self._safe_json(
                payload
            ),
            flush=True,
        )

        response = self.session.post(
            self._url(
                f"/gradio_api/call/v2/{endpoint}"
            ),
            json=payload,
            timeout=self._request_timeout(),
        )

        if response.status_code != 200:
            body = response.text[:5000]

            raise RuntimeError(
                "Hugging Face generation POST failed: "
                f"HTTP {response.status_code}: "
                f"{body}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face generation returned "
                "invalid JSON."
            ) from exc

        event_id = (
            data.get("event_id")
            if isinstance(
                data,
                dict,
            )
            else None
        )

        if not event_id:
            # Algumas versões retornam o event_id em
            # estruturas diferentes.
            if isinstance(
                data,
                str,
            ):
                event_id = data.strip()

        if not event_id:
            raise RuntimeError(
                "Hugging Face did not return an event_id.\n"
                f"Response: {str(data)[:2000]}"
            )

        print(
            "[IMAGE] Hugging Face: event_id="
            f"{event_id}",
            flush=True,
        )

        return self._poll_event(
            endpoint,
            str(event_id),
        )

    # =========================================================
    # SSE
    # =========================================================

    def _poll_event(
        self,
        endpoint: str,
        event_id: str,
    ) -> str:
        """
        Lê o SSE retornado pelo Gradio até receber:

            event: complete

        ou um evento de erro.
        """

        url = self._url(
            f"/gradio_api/call/"
            f"{endpoint}/{event_id}"
        )

        print(
            "[IMAGE] Hugging Face: polling SSE "
            f"{url}",
            flush=True,
        )

        started = time.monotonic()

        try:
            with self.session.get(
                url,
                stream=True,
                timeout=self._request_timeout(),
            ) as response:

                if response.status_code != 200:
                    body = response.text[:5000]

                    raise RuntimeError(
                        "Hugging Face SSE failed: "
                        f"HTTP {response.status_code}: "
                        f"{body}"
                    )

                event_name: str | None = None
                event_data: str | None = None

                for raw_line in response.iter_lines(
                    decode_unicode=True
                ):
                    if (
                        time.monotonic()
                        - started
                        > self.timeout
                    ):
                        raise TimeoutError(
                            "Timeout aguardando geração "
                            "do Hugging Face."
                        )

                    if raw_line is None:
                        continue

                    line = str(
                        raw_line
                    )

                    # Evento completo.
                    if line.startswith(
                        "event:"
                    ):
                        event_name = (
                            line[6:]
                            .strip()
                        )
                        continue

                    # Dados do evento.
                    if line.startswith(
                        "data:"
                    ):
                        event_data = (
                            line[5:]
                            .strip()
                        )
                        continue

                    # Linha vazia encerra o evento.
                    if not line.strip():
                        if event_name:
                            result = (
                                self._handle_sse_event(
                                    event_name,
                                    event_data,
                                )
                            )

                            if result is not None:
                                return result

                        event_name = None
                        event_data = None

                # Alguns servidores podem encerrar o stream
                # imediatamente após enviar o evento.
                if event_name:
                    result = (
                        self._handle_sse_event(
                            event_name,
                            event_data,
                        )
                    )

                    if result is not None:
                        return result

        except requests.RequestException as exc:
            raise RuntimeError(
                "Erro de rede durante SSE do Hugging Face: "
                f"{exc}"
            ) from exc

        raise RuntimeError(
            "Hugging Face encerrou o SSE sem retornar "
            "uma imagem."
        )

    def _handle_sse_event(
        self,
        event_name: str,
        event_data: str | None,
    ) -> str | None:
        """
        Processa um evento SSE.

        O evento final normalmente é:

            event: complete
            data: [...]

        """

        print(
            "[IMAGE] Hugging Face: SSE event="
            f"{event_name} "
            f"data={str(event_data)[:1000]}",
            flush=True,
        )

        if event_name.lower() in {
            "error",
            "cancel",
        }:
            raise RuntimeError(
                "Hugging Face SSE returned "
                f"{event_name}: "
                f"{event_data}"
            )

        if event_name.lower() not in {
            "complete",
            "done",
        }:
            return None

        if not event_data:
            raise RuntimeError(
                "Hugging Face returned a complete "
                "SSE event without data."
            )

        try:
            data = json.loads(
                event_data
            )
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face returned invalid "
                "SSE completion data."
            ) from exc

        return self._extract_image_reference(
            data
        )

    # =========================================================
    # RESULT PARSING
    # =========================================================

    def _extract_image_reference(
        self,
        data: Any,
    ) -> str:
        """
        Extrai o arquivo de imagem do resultado do Gradio.

        O Space oficial retorna a imagem como primeiro output
        da função infer().
        """

        # -----------------------------------------------------
        # Primeiro output
        # -----------------------------------------------------

        value = data

        if isinstance(
            data,
            (list, tuple),
        ):
            if not data:
                raise RuntimeError(
                    "Hugging Face returned an empty "
                    "completion result."
                )

            value = data[0]

        # -----------------------------------------------------
        # Gallery / lista aninhada
        # -----------------------------------------------------

        if isinstance(
            value,
            (list, tuple),
        ):
            if not value:
                raise RuntimeError(
                    "Hugging Face returned an empty "
                    "image gallery."
                )

            value = value[0]

        # -----------------------------------------------------
        # FileData
        # -----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):
            for key in (
                "path",
                "url",
                "image",
                "file",
                "video",
            ):
                candidate = value.get(key)

                if candidate:
                    return self._normalize_remote_file(
                        str(candidate)
                    )

            # Estrutura aninhada.
            for key in (
                "data",
                "value",
            ):
                nested = value.get(key)

                if nested:
                    try:
                        return self._extract_image_reference(
                            nested
                        )
                    except Exception:
                        pass

        # -----------------------------------------------------
        # String
        # -----------------------------------------------------

        if isinstance(
            value,
            str,
        ):
            return self._normalize_remote_file(
                value
            )

        raise RuntimeError(
            "Não foi possível localizar a imagem "
            "na resposta do Hugging Face.\n"
            f"Resposta: {str(data)[:5000]}"
        )

    def _normalize_remote_file(
        self,
        value: str,
    ) -> str:
        """
        Normaliza:
            - URL HTTP
            - caminho retornado pelo Gradio
            - caminho relativo
        """

        value = value.strip()

        if not value:
            raise RuntimeError(
                "Hugging Face returned an empty image path."
            )

        if value.startswith(
            "http://"
        ) or value.startswith(
            "https://"
        ):
            return value

        # Caminhos do cache do Gradio precisam ser buscados
        # pelo endpoint /gradio_api/file=...
        encoded = requests.utils.quote(
            value,
            safe="",
        )

        return self._url(
            f"/gradio_api/file={encoded}"
        )

    # =========================================================
    # IMAGE DOWNLOAD
    # =========================================================

    def _download_image(
        self,
        image_reference: str,
    ) -> bytes:
        """
        Baixa a imagem gerada para bytes.

        O ImageService do projeto trabalha com image_bytes,
        então não deixamos o caminho temporário do Space
        atravessar o restante da arquitetura.
        """

        print(
            "[IMAGE] Hugging Face: downloading result",
            flush=True,
        )

        response = self.session.get(
            image_reference,
            timeout=self._request_timeout(),
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face image download failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )

        content = response.content

        if not content:
            raise RuntimeError(
                "Hugging Face returned an empty image."
            )

        return content

    # =========================================================
    # PUBLIC GENERATE
    # =========================================================

    def generate(
        self,
        request: Any,
        prompt: str,
    ):
        """
        Interface esperada pelo ImageProviderRouter:

            generate(request, prompt) -> ImageResult

        O método retorna um objeto simples compatível com
        os campos usados pelo restante do projeto.
        """

        job_id = str(
            uuid.uuid4()
        )

        try:
            print(
                "[IMAGE] Hugging Face: starting official "
                "FLUX.2 Klein 4B generation",
                flush=True,
            )

            # -------------------------------------------------
            # Descobrir endpoint
            # -------------------------------------------------

            if not self._endpoint:
                self._discover_endpoint()

            # -------------------------------------------------
            # Montar payload
            # -------------------------------------------------

            payload = self._build_payload(
                prompt,
                request,
            )

            print(
                "[IMAGE] Hugging Face: mode="
                f"{payload.get('mode_choice', self.DEFAULT_MODE)} "
                f"width={payload.get('width', 1024)} "
                f"height={payload.get('height', 1024)} "
                f"steps={payload.get('num_inference_steps', 4)} "
                f"guidance={payload.get('guidance_scale', 1.0)} "
                f"seed={payload.get('seed', 42)} "
                f"reference="
                f"{bool(payload.get('input_images'))}",
                flush=True,
            )

            # -------------------------------------------------
            # Geração
            # -------------------------------------------------

            image_reference = (
                self._call_generation(
                    payload
                )
            )

            # -------------------------------------------------
            # Download
            # -------------------------------------------------

            image_bytes = (
                self._download_image(
                    image_reference
                )
            )

            print(
                "[IMAGE] Hugging Face: generation "
                "completed successfully "
                f"bytes={len(image_bytes)}",
                flush=True,
            )

            return self._make_result(
                success=True,
                provider=self.name,
                job_id=job_id,
                image_url=image_reference,
                image_bytes=image_bytes,
                error=None,
                face_swapped=False,
            )

        except Exception as exc:
            error = (
                "huggingface:"
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "[IMAGE ERROR] Generation failed. "
                f"error={error!r}",
                flush=True,
            )

            return self._make_result(
                success=False,
                provider=self.name,
                job_id=job_id,
                image_url=None,
                image_bytes=None,
                error=error,
                face_swapped=False,
            )

    # =========================================================
    # RESULT OBJECT
    # =========================================================

    @staticmethod
    def _make_result(
        *,
        success: bool,
        provider: str,
        job_id: str,
        image_url: str | None,
        image_bytes: bytes | None,
        error: str | None,
        face_swapped: bool,
    ):
        """
        Cria o ImageResult do projeto sem obrigar este provider
        a conhecer detalhes internos de implementação.

        Primeiro tentamos localizar o ImageResult existente
        no projeto.

        Caso a localização tenha mudado, usamos um pequeno
        objeto compatível por atributos.
        """

        # -----------------------------------------------------
        # Tentativa 1: localizações comuns do projeto
        # -----------------------------------------------------

        candidates = (
            "app.images.models",
            "app.images.schemas",
            "app.images.provider",
            "app.images.base",
        )

        result_class = None

        for module_name in candidates:
            try:
                module = __import__(
                    module_name,
                    fromlist=["ImageResult"],
                )

                candidate = getattr(
                    module,
                    "ImageResult",
                    None,
                )

                if candidate is not None:
                    result_class = candidate
                    break

            except Exception:
                continue

        if result_class is not None:
            try:
                return result_class(
                    success=success,
                    provider=provider,
                    job_id=job_id,
                    image_url=image_url,
                    image_bytes=image_bytes,
                    error=error,
                    face_swapped=face_swapped,
                )
            except TypeError:
                # Algumas versões podem não ter todos os
                # campos no construtor.
                try:
                    return result_class(
                        success=success,
                        provider=provider,
                        job_id=job_id,
                        image_url=image_url,
                        image_bytes=image_bytes,
                        error=error,
                    )
                except Exception:
                    pass

        # -----------------------------------------------------
        # Fallback compatível
        # -----------------------------------------------------

        class ProviderResult:
            def __init__(self):
                self.success = success
                self.provider = provider
                self.job_id = job_id
                self.image_url = image_url
                self.image_bytes = image_bytes
                self.error = error
                self.face_swapped = face_swapped

            def __repr__(self) -> str:
                return (
                    "ProviderResult("
                    f"success={self.success!r}, "
                    f"provider={self.provider!r}, "
                    f"job_id={self.job_id!r}, "
                    f"image_url={self.image_url!r}, "
                    f"image_bytes="
                    f"{len(self.image_bytes or b'')} bytes, "
                    f"error={self.error!r}, "
                    f"face_swapped="
                    f"{self.face_swapped!r}"
                    ")"
                )

        return ProviderResult()

    # =========================================================
    # LOGGING
    # =========================================================

    @staticmethod
    def _safe_json(
        value: Any,
    ) -> str:
        """
        JSON seguro para logs.

        Evita imprimir blobs enormes de imagem/base64.
        """

        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            text = str(value)

        # Não deixar logs gigantes no Render.
        if len(text) > 6000:
            text = (
                text[:6000]
                + "...[truncated]"
            )

        return text

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    def health_check(self) -> bool:
        """
        Verifica se o Space está acessível.

        Não dispara geração.
        """

        try:
            response = self.session.get(
                self._url(
                    "/gradio_api/info"
                ),
                timeout=30,
            )

            return response.status_code == 200

        except Exception:
            return False


__all__ = [
    "HuggingFaceImageProvider",
]
