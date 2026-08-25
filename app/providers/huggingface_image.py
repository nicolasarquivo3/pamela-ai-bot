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
    """Provider para o Space oficial FLUX.2 Klein 4B.

    O endpoint Gradio é descoberto dinamicamente para evitar
    depender de /infer, que causou FnIndexInferError.
    """

    name = "huggingface"
    DEFAULT_SPACE_URL = "https://black-forest-labs-flux-2-klein-4b.hf.space"
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
            {"User-Agent": "PamelaAI/1.0 HuggingFaceImageProvider"}
        )
        if self.hf_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.hf_token}"}
            )

        self._api_info: dict[str, Any] | None = None
        self._endpoint: str | None = None
        self._endpoint_parameters: list[dict[str, Any]] = []
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Compatibilidade com ImageProviderRouter."""
        if self._available is not None:
            return self._available
        try:
            response = self.session.get(
                self._url("/gradio_api/info"),
                timeout=(15.0, 30.0),
            )
            self._available = response.status_code == 200
        except Exception:
            self._available = False
        return bool(self._available)

    def _url(self, path: str) -> str:
        return f"{self.space_url}/{path.lstrip('/')}"

    def _request_timeout(self) -> tuple[float, float]:
        return min(30.0, self.timeout), max(60.0, self.timeout)

    def _get_api_info(self) -> dict[str, Any]:
        url = self._url("/gradio_api/info?all_endpoints=true")
        print(f"[IMAGE] Hugging Face: GET {url}", flush=True)
        try:
            response = self.session.get(
                url, timeout=self._request_timeout()
            )
        except requests.RequestException as exc:
            self._available = False
            raise RuntimeError(
                f"Erro de rede consultando Hugging Face: {exc}"
            ) from exc

        if response.status_code != 200:
            self._available = False
            raise RuntimeError(
                "Hugging Face API info failed: "
                f"HTTP {response.status_code}: {response.text[:3000]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Hugging Face retornou JSON inválido em /gradio_api/info."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError("Hugging Face retornou API info inválida.")

        self._available = True
        return data

    @staticmethod
    def _parameter_names(endpoint: dict[str, Any]) -> set[str]:
        result: set[str] = set()
        parameters = endpoint.get("parameters", [])
        if not isinstance(parameters, list):
            return result
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue
            for key in ("parameter_name", "name"):
                value = parameter.get(key)
                if value:
                    result.add(str(value))
        return result

    @staticmethod
    def _parameter_list(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
        parameters = endpoint.get("parameters", [])
        if not isinstance(parameters, list):
            return []
        return [p for p in parameters if isinstance(p, dict)]

    @staticmethod
    def _parameter_name(parameter: dict[str, Any]) -> str | None:
        value = parameter.get("parameter_name") or parameter.get("name")
        return str(value) if value else None

    @classmethod
    def _score_endpoint(cls, endpoint: dict[str, Any]) -> int:
        names = {n.lower() for n in cls._parameter_names(endpoint)}
        if "prompt" not in names:
            return 0

        weights = {
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
        score = sum(v for k, v in weights.items() if k in names)
        if {"width", "height"} <= names:
            score += 20
        if {"num_inference_steps", "guidance_scale"} <= names:
            score += 20
        return score

    def _discover_endpoint(self) -> str:
        info = self._get_api_info()
        self._api_info = info
        candidates: list[tuple[int, str, dict[str, Any], str]] = []

        for source_key, source_name in (
            ("named_endpoints", "named"),
            ("unnamed_endpoints", "unnamed"),
        ):
            source = info.get(source_key, {})
            if not isinstance(source, dict):
                continue
            for endpoint_name, endpoint_data in source.items():
                if not isinstance(endpoint_data, dict):
                    continue
                score = self._score_endpoint(endpoint_data)
                if score > 0:
                    candidates.append(
                        (score, str(endpoint_name).strip("/"),
                         endpoint_data, source_name)
                    )

        if not candidates:
            dependencies = info.get("dependencies", [])
            if isinstance(dependencies, list):
                for index, dependency in enumerate(dependencies):
                    if not isinstance(dependency, dict):
                        continue
                    score = self._score_endpoint(dependency)
                    if score <= 0:
                        continue
                    api_name = dependency.get("api_name")
                    endpoint_name = (
                        str(api_name).strip("/")
                        if api_name
                        else str(index)
                    )
                    source = (
                        "dependency-api-name"
                        if api_name
                        else "dependency-index"
                    )
                    candidates.append(
                        (score, endpoint_name, dependency, source)
                    )

        if not candidates:
            raise RuntimeError(
                "Não foi possível localizar o endpoint do FLUX.2 Klein 4B.\n"
                + self._summarize_api_info(info)
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, endpoint_name, endpoint_data, source = candidates[0]
        self._endpoint = endpoint_name
        self._endpoint_parameters = self._parameter_list(endpoint_data)

        names = [
            self._parameter_name(p)
            for p in self._endpoint_parameters
        ]
        names = [n for n in names if n]

        print(
            "[IMAGE] Hugging Face: endpoint descoberto: "
            f"/{endpoint_name} (score={score}, source={source})",
            flush=True,
        )
        print(
            "[IMAGE] Hugging Face: parâmetros: " + ", ".join(names),
            flush=True,
        )
        return endpoint_name

    @staticmethod
    def _summarize_api_info(info: dict[str, Any]) -> str:
        lines: list[str] = []
        for source_key, label in (
            ("named_endpoints", "named"),
            ("unnamed_endpoints", "unnamed"),
        ):
            source = info.get(source_key, {})
            if isinstance(source, dict):
                for name, endpoint in source.items():
                    params = (
                        sorted(
                            HuggingFaceImageProvider._parameter_names(endpoint)
                        )
                        if isinstance(endpoint, dict)
                        else []
                    )
                    lines.append(f"{label} /{name}: {params}")

        dependencies = info.get("dependencies", [])
        if isinstance(dependencies, list):
            for i, dependency in enumerate(dependencies):
                if not isinstance(dependency, dict):
                    continue
                params = sorted(
                    HuggingFaceImageProvider._parameter_names(dependency)
                )
                api_name = dependency.get("api_name")
                suffix = f" api_name=/{api_name}" if api_name else ""
                lines.append(f"dependency {i}{suffix}: {params}")

        return "\n".join(lines) if lines else "(nenhum endpoint encontrado)"

    def _upload_file(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Imagem de referência não encontrada: {file_path}"
            )
        if not file_path.is_file():
            raise ValueError(f"Não é arquivo: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path.name)
        mime_type = mime_type or "application/octet-stream"

        try:
            with file_path.open("rb") as handle:
                response = self.session.post(
                    self._url("/gradio_api/upload"),
                    files={"files": (file_path.name, handle, mime_type)},
                    timeout=self._request_timeout(),
                )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Erro de rede no upload para Hugging Face: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face upload failed: "
                f"HTTP {response.status_code}: {response.text[:3000]}"
            )

        try:
            uploaded = response.json()
        except ValueError as exc:
            raise RuntimeError("Upload retornou JSON inválido.") from exc

        if not isinstance(uploaded, list) or not uploaded:
            raise RuntimeError("Upload não retornou caminho de arquivo.")

        value = uploaded[0]
        if isinstance(value, dict):
            value = value.get("path") or value.get("url")
        if not value:
            raise RuntimeError("Upload retornou caminho vazio.")
        return str(value)

    @staticmethod
    def _file_data(path: str, original_name: str | None = None) -> dict[str, Any]:
        return {
            "path": path,
            "url": None,
            "size": None,
            "orig_name": original_name or Path(path).name,
            "mime_type": None,
            "is_stream": False,
            "meta": {"_type": "gradio.FileData"},
        }

    def _prepare_input_images(self, reference_images: Any) -> list[list[Any]]:
        if not reference_images:
            return []

        if isinstance(reference_images, (str, Path)):
            reference_images = [reference_images]

        if not isinstance(reference_images, (list, tuple)):
            return []

        result: list[list[Any]] = []

        for item in reference_images:
            if not item:
                continue

            if isinstance(item, Path):
                uploaded = self._upload_file(item)
                result.append([self._file_data(uploaded, item.name), None])
                continue

            if isinstance(item, str):
                if item.startswith(("http://", "https://")):
                    result.append([
                        self._file_data(
                            item,
                            Path(item.split("?")[0]).name,
                        ),
                        None,
                    ])
                    continue

                local_path = Path(item)
                if local_path.exists():
                    uploaded = self._upload_file(local_path)
                    result.append([
                        self._file_data(uploaded, local_path.name),
                        None,
                    ])
                    continue

            if isinstance(item, dict) and item.get("path"):
                result.append([item, None])

        return result

    @staticmethod
    def _get_request_value(
        request: Any | None,
        name: str,
        default: Any,
    ) -> Any:
        if request is None:
            return default
        if isinstance(request, dict):
            value = request.get(name)
            return default if value is None else value
        try:
            value = getattr(request, name)
        except Exception:
            return default
        return default if value is None else value

    def _build_payload(
        self,
        prompt: str,
        request: Any | None = None,
    ) -> dict[str, Any]:
        width = self._get_request_value(request, "width", 1024)
        height = self._get_request_value(request, "height", 1024)
        seed = self._get_request_value(request, "seed", 42)
        mode_choice = self._get_request_value(
            request, "mode_choice", self.DEFAULT_MODE
        )
        steps = self._get_request_value(
            request, "num_inference_steps", self.DEFAULT_STEPS
        )
        guidance = self._get_request_value(
            request, "guidance_scale", self.DEFAULT_GUIDANCE
        )
        randomize_seed = bool(
            self._get_request_value(request, "randomize_seed", False)
        )
        prompt_upsampling = bool(
            self._get_request_value(request, "prompt_upsampling", False)
        )
        reference_images = self._get_request_value(
            request, "reference_images", None
        )

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

        width = max(256, min(1024, width))
        height = max(256, min(1024, height))
        width = round(width / 8) * 8
        height = round(height / 8) * 8
        steps = max(1, min(100, steps))
        guidance = max(0.0, min(10.0, guidance))
        seed = max(0, min(self.MAX_SEED, seed))

        if randomize_seed:
            seed = random.randint(0, self.MAX_SEED)

        payload: dict[str, Any] = {
            "prompt": str(prompt or "").strip(),
            "input_images": self._prepare_input_images(reference_images),
            "mode_choice": mode_choice,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt_upsampling": prompt_upsampling,
        }

        if self._endpoint_parameters:
            available = {
                name
                for name in (
                    self._parameter_name(p)
                    for p in self._endpoint_parameters
                )
                if name
            }
            if available:
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key in available
                }

        return payload

    def _payload_as_gradio_data(self, payload: dict[str, Any]) -> list[Any]:
        """Converte o dict para a ordem exata publicada pelo Gradio."""
        data: list[Any] = []
        for parameter in self._endpoint_parameters:
            name = self._parameter_name(parameter)
            data.append(payload.get(name) if name else None)
        return data

    def _call_generation(self, payload: dict[str, Any]) -> str:
        if not self._endpoint:
            self._discover_endpoint()
        if not self._endpoint:
            raise RuntimeError("Endpoint não descoberto.")

        endpoint = self._endpoint.strip("/")
        url = self._url(f"/gradio_api/call/{endpoint}")

        if self._endpoint_parameters:
            body: dict[str, Any] = {
                "data": self._payload_as_gradio_data(payload)
            }
        else:
            body = payload

        print(
            f"[IMAGE] Hugging Face: POST /gradio_api/call/{endpoint}",
            flush=True,
        )
        print(
            "[IMAGE] Hugging Face: payload="
            + self._safe_json(body),
            flush=True,
        )

        try:
            response = self.session.post(
                url,
                json=body,
                timeout=self._request_timeout(),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Erro de rede no POST de geração: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face generation POST failed: "
                f"HTTP {response.status_code}: {response.text[:5000]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Geração retornou JSON inválido."
            ) from exc

        event_id = None
        if isinstance(data, dict):
            event_id = data.get("event_id")
        elif isinstance(data, str):
            event_id = data.strip()

        if not event_id:
            raise RuntimeError(
                "Hugging Face não retornou event_id. "
                f"Response: {str(data)[:3000]}"
            )

        print(f"[IMAGE] Hugging Face: event_id={event_id}", flush=True)
        return self._poll_event(endpoint, str(event_id))

    def _poll_event(self, endpoint: str, event_id: str) -> str:
        url = self._url(f"/gradio_api/call/{endpoint}/{event_id}")
        print(f"[IMAGE] Hugging Face: polling SSE {url}", flush=True)

        started = time.monotonic()

        try:
            with self.session.get(
                url,
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=self._request_timeout(),
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        "Hugging Face SSE failed: "
                        f"HTTP {response.status_code}: {response.text[:5000]}"
                    )

                event_name: str | None = None
                data_lines: list[str] = []

                for raw_line in response.iter_lines(
                    decode_unicode=True
                ):
                    if time.monotonic() - started > self.timeout:
                        raise TimeoutError(
                            "Timeout aguardando geração do Hugging Face."
                        )

                    if raw_line is None:
                        continue

                    line = str(raw_line)

                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue

                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue

                    if not line.strip():
                        if event_name:
                            result = self._handle_sse_event(
                                event_name,
                                "\n".join(data_lines),
                            )
                            if result is not None:
                                return result
                        event_name = None
                        data_lines = []

                if event_name:
                    result = self._handle_sse_event(
                        event_name,
                        "\n".join(data_lines),
                    )
                    if result is not None:
                        return result

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Erro de rede durante SSE: {exc}"
            ) from exc

        raise RuntimeError(
            "Hugging Face encerrou o SSE sem retornar imagem."
        )

    def _handle_sse_event(
        self,
        event_name: str,
        event_data: str | None,
    ) -> str | None:
        print(
            f"[IMAGE] Hugging Face: SSE event={event_name} "
            f"data={str(event_data)[:1500]}",
            flush=True,
        )

        normalized = event_name.lower().strip()

        if normalized in {"error", "cancel"}:
            raise RuntimeError(
                f"Hugging Face SSE returned {event_name}: {event_data}"
            )

        if normalized not in {"complete", "done"}:
            return None

        if not event_data:
            raise RuntimeError(
                "Evento complete sem dados."
            )

        try:
            data = json.loads(event_data)
        except ValueError as exc:
            raise RuntimeError(
                "Dados do evento SSE são JSON inválido."
            ) from exc

        return self._extract_image_reference(data)

    def _extract_image_reference(self, data: Any) -> str:
        if isinstance(data, (list, tuple)):
            if not data:
                raise RuntimeError("Resultado vazio.")
            errors: list[str] = []
            for item in data:
                try:
                    return self._extract_image_reference(item)
                except Exception as exc:
                    errors.append(str(exc))
            raise RuntimeError(
                "Nenhum arquivo de imagem encontrado. "
                + " | ".join(errors[:3])
            )

        if isinstance(data, dict):
            for key in ("path", "url", "image", "file"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return self._normalize_remote_file(value)

            for key in ("data", "value", "output", "outputs"):
                value = data.get(key)
                if value is None:
                    continue
                try:
                    return self._extract_image_reference(value)
                except Exception:
                    pass

        if isinstance(data, str) and data.strip():
            return self._normalize_remote_file(data.strip())

        raise RuntimeError(
            "Não foi possível localizar a imagem na resposta. "
            f"Resposta: {str(data)[:5000]}"
        )

    def _normalize_remote_file(self, value: str) -> str:
        value = value.strip()
        if value.startswith(("http://", "https://")):
            return value
        encoded = requests.utils.quote(value, safe="")
        return self._url(f"/gradio_api/file={encoded}")

    def _download_image(self, image_reference: str) -> bytes:
        print("[IMAGE] Hugging Face: downloading result", flush=True)

        try:
            response = self.session.get(
                image_reference,
                timeout=self._request_timeout(),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Erro de rede baixando imagem: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                "Hugging Face image download failed: "
                f"HTTP {response.status_code}: {response.text[:3000]}"
            )

        content = response.content
        if not content:
            raise RuntimeError("Hugging Face retornou imagem vazia.")

        content_type = response.headers.get(
            "content-type", ""
        ).lower()
        if "text/html" in content_type or "application/json" in content_type:
            raise RuntimeError(
                "Hugging Face retornou conteúdo que não parece imagem. "
                f"content-type={content_type}"
            )

        return content

    def generate(self, request: Any, prompt: str):
        job_id = str(uuid.uuid4())

        try:
            print(
                "[IMAGE] Hugging Face: starting official "
                "FLUX.2 Klein 4B generation",
                flush=True,
            )

            if not self._endpoint:
                self._discover_endpoint()

            payload = self._build_payload(prompt, request)

            print(
                "[IMAGE] Hugging Face: mode="
                f"{payload.get('mode_choice', self.DEFAULT_MODE)} "
                f"width={payload.get('width', 1024)} "
                f"height={payload.get('height', 1024)} "
                f"steps={payload.get('num_inference_steps', 4)} "
                f"guidance={payload.get('guidance_scale', 1.0)} "
                f"seed={payload.get('seed', 42)} "
                f"reference={bool(payload.get('input_images'))}",
                flush=True,
            )

            image_reference = self._call_generation(payload)
            image_bytes = self._download_image(image_reference)

            print(
                "[IMAGE] Hugging Face: generation completed "
                f"successfully bytes={len(image_bytes)}",
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
                f"{type(exc).__name__}: {exc}"
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
        candidates = (
            "app.images.models",
            "app.images.schemas",
            "app.images.provider",
            "app.images.base",
            "app.providers.image",
            "app.providers.base",
        )

        for module_name in candidates:
            try:
                module = __import__(
                    module_name,
                    fromlist=["ImageResult"],
                )
                result_class = getattr(module, "ImageResult", None)
                if result_class is None:
                    continue

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
                    try:
                        return result_class(
                            success=success,
                            provider=provider,
                            job_id=job_id,
                            image_url=image_url,
                            image_bytes=image_bytes,
                            error=error,
                        )
                    except TypeError:
                        pass
            except Exception:
                continue

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
                    f"face_swapped={self.face_swapped!r}"
                    ")"
                )

        return ProviderResult()

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            text = str(value)

        if len(text) > 6000:
            text = text[:6000] + "...[truncated]"
        return text

    def health_check(self) -> bool:
        try:
            response = self.session.get(
                self._url("/gradio_api/info"),
                timeout=(15.0, 30.0),
            )
            self._available = response.status_code == 200
            return bool(self._available)
        except Exception:
            self._available = False
            return False


__all__ = ["HuggingFaceImageProvider"]
