import asyncio
import base64
import tempfile
from pathlib import Path

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """
    Face swap free-first (sem upscale Pillow — evita "quadriculado"):

      1. Hugging Face Gradio FaceFusion/Hyperswap
         - tenta PRIMEIRO com face restore neural (gfpgan/codeformer no Space)
      2. Opcional: 2º Space GRÁTIS só de enhance (CodeFormer/GFPGAN)
      3. Replicate (opcional / pago se token)

    NÃO faz resize/unsharp local. Qualidade vem do modelo no Space.
    """

    def __init__(
        self,
        reference_path: str,
        required: bool = True,
        provider: str = "huggingface",
        hf_space: str = "tonyassi/face-swap",
        hf_api_name: str = "/swap_faces",
        hf_token: str | None = None,
        hf_swap_model: str = "hyperswap_1b_256.onnx",
        hf_target_index: int = 0,
        hf_restore_model: str = "gfpgan_1.4",
        hf_restore_strength: float = 0.5,
        # 2º passo grátis: enhance neural após o swap (pode ser lento / fila)
        hf_enhance_enabled: bool = True,
        hf_enhance_space: str = "sczhou/CodeFormer",
        hf_enhance_api_name: str = "/predict",
        replicate_token: str | None = None,
        replicate_version: str = (
            "codeplugtech/face-swap:"
            "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"
        ),
        timeout: int = 180,
    ):
        self.reference_path = Path(reference_path)
        self.required = bool(required)
        self.provider = provider
        self.hf_space = hf_space
        self.hf_api_name = hf_api_name
        self.hf_token = hf_token
        self.hf_swap_model = hf_swap_model
        self.hf_target_index = int(hf_target_index)
        self.hf_restore_model = (hf_restore_model or "none").strip()
        self.hf_restore_strength = float(hf_restore_strength)
        self.hf_enhance_enabled = bool(hf_enhance_enabled)
        self.hf_enhance_space = (hf_enhance_space or "").strip()
        self.hf_enhance_api_name = hf_enhance_api_name or "/predict"
        self.replicate_token = replicate_token
        self.replicate_version = replicate_version
        self.timeout = int(timeout)

    async def available(self) -> bool:
        if not self.reference_path.is_file():
            return False
        if self.provider in {"none", "disabled"}:
            return not self.required
        if self.provider in {"huggingface", "auto"}:
            try:
                import gradio_client  # noqa: F401
                return True
            except ImportError:
                pass
        if self.provider in {"replicate", "auto"}:
            return bool(self.replicate_token)
        return False

    async def apply(self, generated: ImageResult) -> ImageResult:
        if not generated.success:
            return generated
        if self.provider in {"none", "disabled"}:
            return generated if not self.required else ImageResult(
                False, error="face_swap_disabled"
            )

        target_bytes = await self._get_image_bytes(generated)
        if not target_bytes:
            return (
                ImageResult(False, error="face_swap_target_unavailable")
                if self.required
                else generated
            )

        providers = self._provider_order()
        errors: list[str] = []
        for name in providers:
            try:
                if name == "huggingface":
                    output = await self._huggingface_swap(target_bytes)
                else:
                    output = await self._replicate_swap(target_bytes)
                if output:
                    # 2º passo GRÁTIS: enhance neural (se falhar, mantém o swap)
                    if self.hf_enhance_enabled and self.hf_enhance_space:
                        try:
                            enhanced = await self._huggingface_enhance(output)
                            if enhanced:
                                print(
                                    f"[FaceSwap] enhance OK space={self.hf_enhance_space} "
                                    f"bytes {len(output)}->{len(enhanced)}",
                                    flush=True,
                                )
                                output = enhanced
                                name = f"{name}+enhance"
                            else:
                                print("[FaceSwap] enhance sem output — mantém swap", flush=True)
                        except Exception as e:
                            print(f"[FaceSwap] enhance skip: {e}", flush=True)
                    return ImageResult(
                        success=True,
                        provider=f"{generated.provider or 'image'}+faceswap:{name}",
                        job_id=generated.job_id,
                        image_bytes=output,
                        face_swapped=True,
                    )
                errors.append(f"{name}:no_output")
            except Exception as exc:
                errors.append(f"{name}:{exc}")
                print(f"[FaceSwap] {name} error: {exc}", flush=True)

        if self.required:
            return ImageResult(
                False,
                provider=generated.provider,
                error="; ".join(errors) or "face_swap_failed",
            )
        return generated

    def _provider_order(self) -> list[str]:
        if self.provider == "huggingface":
            return ["huggingface"] + (["replicate"] if self.replicate_token else [])
        if self.provider == "replicate":
            return ["replicate"]
        if self.provider == "auto":
            order = ["huggingface"]
            if self.replicate_token:
                order.append("replicate")
            return order
        return []

    async def _get_image_bytes(self, generated: ImageResult) -> bytes | None:
        if generated.image_bytes:
            return generated.image_bytes
        if not generated.image_url:
            return None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(generated.image_url)
            response.raise_for_status()
            return response.content

    async def _huggingface_swap(self, target_bytes: bytes) -> bytes | None:
        return await asyncio.to_thread(self._huggingface_swap_sync, target_bytes)

    def _huggingface_swap_sync(self, target_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        model = self.hf_swap_model
        model_plain = model.replace(".onnx", "") if model.endswith(".onnx") else model
        model_onnx = model if model.endswith(".onnx") else f"{model}.onnx"
        restore = self.hf_restore_model
        strength = self.hf_restore_strength
        idx = self.hf_target_index

        with tempfile.TemporaryDirectory(prefix="face-swap-") as tmp:
            source = Path(tmp) / "source.jpg"
            target = Path(tmp) / "target.jpg"
            source.write_bytes(self.reference_path.read_bytes())
            target.write_bytes(target_bytes)

            client_kwargs = {}
            if self.hf_token:
                client_kwargs["token"] = self.hf_token

            client = Client(self.hf_space, **client_kwargs)
            print(
                f"[FaceSwap] HF space={self.hf_space} api={self.hf_api_name} "
                f"model={model} restore={restore} strength={strength}",
                flush=True,
            )

            # IMPORTANTE: restore/model PRIMEIRO.
            # Se a API simples for primeiro, o swap "cru" (rosto 256) vence e o restore nunca roda.
            attempts = []

            if restore and restore.lower() != "none":
                for m in (model, model_onnx, model_plain, "hyperswap_1b_256.onnx", "hyperswap_1b_256"):
                    for r in (restore, "gfpgan_1.4", "codeformer", "gpen_bfr_512"):
                        def _mk(mm=m, rr=r):
                            return lambda: client.predict(
                                handle_file(str(source)),
                                handle_file(str(target)),
                                idx,
                                mm,
                                rr,
                                strength,
                                api_name=self.hf_api_name,
                            )
                        attempts.append(_mk())
                        def _mk2(mm=m, rr=r):
                            return lambda: client.predict(
                                handle_file(str(source)),
                                handle_file(str(target)),
                                idx,
                                mm,
                                rr,
                                strength,
                                api_name="/generate_image",
                            )
                        attempts.append(_mk2())

            # Fallbacks simples (sem restore) — por último
            attempts.extend(
                [
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        idx,
                        model,
                        "none",
                        0.0,
                        api_name=self.hf_api_name,
                    ),
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        api_name=self.hf_api_name,
                    ),
                    lambda: client.predict(
                        src_img=handle_file(str(source)),
                        dest_img=handle_file(str(target)),
                        api_name=self.hf_api_name,
                    ),
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        api_name="/generate_image",
                    ),
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        api_name="/predict",
                    ),
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        api_name="/swap_faces",
                    ),
                ]
            )

            last_err = None
            # evita 100 tentativas: no máx 12
            for i, attempt in enumerate(attempts[:12]):
                try:
                    result = attempt()
                    data = self._read_result_sync(result)
                    if data:
                        print(f"[FaceSwap] sucesso na tentativa {i+1}", flush=True)
                        return data
                except Exception as e:
                    last_err = e
                    print(f"[FaceSwap] tentativa {i+1} falhou: {e}", flush=True)
                    continue

            if last_err:
                raise last_err
            return None

    async def _huggingface_enhance(self, image_bytes: bytes) -> bytes | None:
        """2º Space GRÁTIS: CodeFormer/GFPGAN — neural, sem LANCZOS."""
        return await asyncio.to_thread(self._huggingface_enhance_sync, image_bytes)

    def _huggingface_enhance_sync(self, image_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        spaces = []
        if self.hf_enhance_space:
            spaces.append(self.hf_enhance_space)
        # fallbacks grátis conhecidos
        for s in (
            "sczhou/CodeFormer",
            "xinntao/GFPGAN",
            "akhaliq/GFPGAN",
            "TencentARC/GFPGAN",
        ):
            if s not in spaces:
                spaces.append(s)

        client_kwargs = {}
        if self.hf_token:
            client_kwargs["token"] = self.hf_token

        with tempfile.TemporaryDirectory(prefix="face-enhance-") as tmp:
            img_path = Path(tmp) / "in.jpg"
            img_path.write_bytes(image_bytes)

            for space in spaces[:3]:
                try:
                    print(f"[FaceSwap] enhance try space={space}", flush=True)
                    client = Client(space, **client_kwargs)
                    attempts = [
                        # CodeFormer típico: image, fidelity, upscale?
                        # upscale=1 para NÃO aumentar (evita artefato)
                        lambda c=client: c.predict(
                            handle_file(str(img_path)),
                            0.5,  # fidelity / weight
                            1,    # upscale factor = 1 (sem zoom)
                            api_name=self.hf_enhance_api_name,
                        ),
                        lambda c=client: c.predict(
                            handle_file(str(img_path)),
                            0.5,
                            1,
                            api_name="/predict",
                        ),
                        lambda c=client: c.predict(
                            handle_file(str(img_path)),
                            api_name="/predict",
                        ),
                        lambda c=client: c.predict(
                            handle_file(str(img_path)),
                            api_name="/",
                        ),
                        # GFPGAN spaces: image only or image + version
                        lambda c=client: c.predict(
                            handle_file(str(img_path)),
                            "v1.4",
                            1,
                            api_name="/predict",
                        ),
                        lambda c=client: c.predict(
                            img=handle_file(str(img_path)),
                            api_name="/predict",
                        ),
                    ]
                    for j, attempt in enumerate(attempts):
                        try:
                            result = attempt()
                            data = self._read_result_sync(result)
                            if data and len(data) > 1000:
                                print(
                                    f"[FaceSwap] enhance ok space={space} try={j+1}",
                                    flush=True,
                                )
                                return data
                        except Exception as e:
                            print(
                                f"[FaceSwap] enhance {space} try={j+1}: {e}",
                                flush=True,
                            )
                            continue
                except Exception as e:
                    print(f"[FaceSwap] enhance space fail {space}: {e}", flush=True)
                    continue
        return None

    async def _replicate_swap(self, target_bytes: bytes) -> bytes | None:
        if not self.replicate_token:
            return None
        source_bytes = self.reference_path.read_bytes()

        def _b64(data: bytes) -> str:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            create = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {self.replicate_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": self.replicate_version.split(":")[-1]
                    if ":" in self.replicate_version
                    else self.replicate_version,
                    "input": {
                        "swap_image": _b64(source_bytes),
                        "input_image": _b64(target_bytes),
                    },
                },
            )
            if create.status_code >= 400:
                create = await client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Authorization": f"Bearer {self.replicate_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "version": self.replicate_version.split(":")[-1]
                        if ":" in self.replicate_version
                        else self.replicate_version,
                        "input": {
                            "source_image": _b64(source_bytes),
                            "target_image": _b64(target_bytes),
                        },
                    },
                )
            create.raise_for_status()
            prediction = create.json()
            prediction_id = prediction.get("id")
            output = prediction.get("output")

            for _ in range(60):
                if prediction.get("status") in {"succeeded", "failed", "canceled"}:
                    break
                await asyncio.sleep(2)
                poll = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Bearer {self.replicate_token}"},
                )
                poll.raise_for_status()
                prediction = poll.json()
                output = prediction.get("output")

            if not output:
                raise RuntimeError(prediction.get("error") or "replicate_no_output")
            return await self._download_output(output)

    async def _download_output(self, output) -> bytes | None:
        if isinstance(output, list):
            output = output[0] if output else None
        if isinstance(output, dict):
            output = output.get("url") or output.get("path") or output.get("data")
        if not isinstance(output, str):
            return None
        if output.startswith("data:"):
            return base64.b64decode(output.split(",", 1)[1])
        if output.startswith("http://") or output.startswith("https://"):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(output)
                response.raise_for_status()
                return response.content
        path = Path(output)
        return path.read_bytes() if path.is_file() else None

    def _read_result_sync(self, result) -> bytes | None:
        if isinstance(result, (list, tuple)):
            for item in result:
                data = self._read_result_sync(item)
                if data:
                    return data
            return None
        if isinstance(result, dict):
            for key in ("path", "url", "data", "value"):
                if key in result:
                    data = self._read_result_sync(result[key])
                    if data:
                        return data
            return None
        if not isinstance(result, str):
            return None
        if result.startswith("data:"):
            return base64.b64decode(result.split(",", 1)[1])
        if result.startswith("http://") or result.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(result, timeout=self.timeout) as response:
                return response.read()
        path = Path(result)
        return path.read_bytes() if path.is_file() else None
