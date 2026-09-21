import asyncio
import base64
import os
import tempfile
from pathlib import Path

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """
    Caminho GRÁTIS (log real de 2026-09):

    1) tonyassi/face-swap  — swap OK
    2) sczhou/CodeFormer /inference — melhora o rosto (ZeroGPU)
       * PRECISA de HF_TOKEN grátis ou a cota zera (erro do log)
    3) Fallback: bookbot/Image-Upscaling-Playground (se CodeFormer sem cota)

    Sem upscale Pillow (não quadrícula).
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
        hf_restore_model: str = "none",
        hf_restore_strength: float = 0.5,
        hf_enhance_enabled: bool = True,
        hf_enhance_space: str = "sczhou/CodeFormer",
        hf_enhance_api_name: str = "/inference",
        hf_enhance_upscale: int = 2,
        hf_enhance_fidelity: float = 0.5,
        replicate_token: str | None = None,
        replicate_version: str = (
            "codeplugtech/face-swap:"
            "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"
        ),
        timeout: int = 240,
    ):
        self.reference_path = Path(reference_path)
        self.required = bool(required)
        self.provider = provider
        self.hf_space = (hf_space or "tonyassi/face-swap").strip()
        self.hf_api_name = hf_api_name or "/swap_faces"
        # Token GRÁTIS do HF — essencial pro ZeroGPU do CodeFormer
        self.hf_token = (
            (hf_token or "").strip()
            or (os.getenv("HF_TOKEN") or "").strip()
            or (os.getenv("HUGGINGFACE_TOKEN") or "").strip()
            or (os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
            or None
        )
        self.hf_swap_model = hf_swap_model
        self.hf_target_index = int(hf_target_index)
        self.hf_restore_model = (hf_restore_model or "none").strip()
        self.hf_restore_strength = float(hf_restore_strength)
        self.hf_enhance_enabled = bool(hf_enhance_enabled)
        self.hf_enhance_space = (hf_enhance_space or "sczhou/CodeFormer").strip()
        self.hf_enhance_api_name = hf_enhance_api_name or "/inference"
        self.hf_enhance_upscale = int(hf_enhance_upscale or 2)
        self.hf_enhance_fidelity = float(hf_enhance_fidelity or 0.5)
        self.replicate_token = replicate_token
        self.replicate_version = replicate_version
        self.timeout = int(timeout)

        if self.hf_token:
            # algumas libs / Spaces leem só do ambiente
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)
        tok = "yes" if self.hf_token else "NO"
        print(
            f"[FaceSwap] init space={self.hf_space} enhance={self.hf_enhance_space} "
            f"hf_token={tok}",
            flush=True,
        )
        if self.hf_enhance_enabled and not self.hf_token:
            print(
                "[FaceSwap] AVISO: sem HF_TOKEN — CodeFormer ZeroGPU vai falhar por cota. "
                "Crie token grátis em https://huggingface.co/settings/tokens "
                "e defina HF_TOKEN no Render.",
                flush=True,
            )

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
                if not output:
                    errors.append(f"{name}:no_output")
                    continue

                tag = name
                if self.hf_enhance_enabled:
                    try:
                        enhanced = await self._enhance_free(output)
                        if enhanced:
                            print(
                                f"[FaceSwap] enhance OK bytes {len(output)}->{len(enhanced)}",
                                flush=True,
                            )
                            output = enhanced
                            tag = f"{name}+enhance"
                        else:
                            print(
                                "[FaceSwap] enhance sem output — mantém swap "
                                "(defina HF_TOKEN grátis no Render se CodeFormer falhou por cota)",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[FaceSwap] enhance skip: {e}", flush=True)

                return ImageResult(
                    success=True,
                    provider=f"{generated.provider or 'image'}+faceswap:{tag}",
                    job_id=generated.job_id,
                    image_bytes=output,
                    face_swapped=True,
                )
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

    def _client_kwargs(self) -> dict:
        # gradio_client >=1.x usa hf_token= (NÃO token=)
        kw = {}
        if self.hf_token:
            kw["hf_token"] = self.hf_token
        return kw

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

        swap_spaces = []
        if self.hf_space:
            swap_spaces.append((self.hf_space, self.hf_api_name))
        if "tonyassi/face-swap" not in {s for s, _ in swap_spaces}:
            swap_spaces.append(("tonyassi/face-swap", "/swap_faces"))

        with tempfile.TemporaryDirectory(prefix="face-swap-") as tmp:
            source = Path(tmp) / "source.jpg"
            target = Path(tmp) / "target.jpg"
            source.write_bytes(self.reference_path.read_bytes())
            target.write_bytes(target_bytes)

            last_err = None
            for space, api in swap_spaces:
                try:
                    client = Client(space, **self._client_kwargs())
                    print(f"[FaceSwap] swap space={space} api={api}", flush=True)
                    attempts = [
                        lambda c=client, a=api: c.predict(
                            handle_file(str(source)),
                            handle_file(str(target)),
                            api_name=a or "/swap_faces",
                        ),
                        lambda c=client: c.predict(
                            handle_file(str(source)),
                            handle_file(str(target)),
                            api_name="/swap_faces",
                        ),
                        lambda c=client: c.predict(
                            handle_file(str(source)),
                            handle_file(str(target)),
                            api_name="/swap_faces_1",
                        ),
                        lambda c=client: c.predict(
                            src_img=handle_file(str(source)),
                            dest_img=handle_file(str(target)),
                            api_name="/swap_faces",
                        ),
                    ]
                    for i, attempt in enumerate(attempts):
                        try:
                            result = attempt()
                            data = self._read_result_sync(result)
                            if data:
                                print(
                                    f"[FaceSwap] swap OK space={space} try={i+1} "
                                    f"bytes={len(data)}",
                                    flush=True,
                                )
                                return data
                        except Exception as e:
                            last_err = e
                            print(f"[FaceSwap] swap {space} try={i+1}: {e}", flush=True)
                except Exception as e:
                    last_err = e
                    print(f"[FaceSwap] swap space fail {space}: {e}", flush=True)

            if last_err:
                raise last_err
            return None

    async def _enhance_free(self, image_bytes: bytes) -> bytes | None:
        return await asyncio.to_thread(self._enhance_free_sync, image_bytes)

    def _enhance_free_sync(self, image_bytes: bytes) -> bytes | None:
        # 1) CodeFormer (melhor pro rosto) — precisa HF_TOKEN p/ cota ZeroGPU
        data = self._codeformer_enhance_sync(image_bytes)
        if data:
            return data
        # 2) Fallback Real-ESRGAN playground (grátis, outro space)
        data = self._bookbot_upscale_sync(image_bytes)
        if data:
            return data
        return None

    def _codeformer_enhance_sync(self, image_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        space = self.hf_enhance_space or "sczhou/CodeFormer"
        upscale = max(1, min(4, int(self.hf_enhance_upscale or 2)))
        fidelity = float(self.hf_enhance_fidelity or 0.5)

        if not self.hf_token:
            print(
                "[FaceSwap] CodeFormer: SEM HF_TOKEN — ZeroGPU quase sempre recusa. "
                "Pulando para fallback / configure HF_TOKEN.",
                flush=True,
            )
            # ainda tenta uma vez (às vezes sobra cota anônima)
        
        with tempfile.TemporaryDirectory(prefix="face-enhance-") as tmp:
            img_path = Path(tmp) / "in.jpg"
            img_path.write_bytes(image_bytes)

            print(
                f"[FaceSwap] CodeFormer space={space} api=/inference "
                f"upscale={upscale} fidelity={fidelity} token={'yes' if self.hf_token else 'NO'}",
                flush=True,
            )
            try:
                client = Client(space, **self._client_kwargs())
            except Exception as e:
                print(f"[FaceSwap] CodeFormer client fail: {e}", flush=True)
                return None

            attempts = [
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True, True, True, upscale, fidelity,
                    api_name="/inference",
                ),
                lambda: client.predict(
                    image=handle_file(str(img_path)),
                    face_align=True,
                    background_enhance=True,
                    face_upsample=True,
                    upscale=upscale,
                    codeformer_fidelity=fidelity,
                    api_name="/inference",
                ),
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True, True, True, 1, fidelity,
                    api_name="/inference",
                ),
            ]

            last_err = None
            for i, attempt in enumerate(attempts):
                try:
                    result = attempt()
                    data = self._read_result_sync(result)
                    if data and len(data) > 1000:
                        print(
                            f"[FaceSwap] CodeFormer sucesso try={i+1} bytes={len(data)}",
                            flush=True,
                        )
                        return data
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    print(f"[FaceSwap] CodeFormer try={i+1}: {msg}", flush=True)
                    if "ZeroGPU quota" in msg or "quota" in msg.lower():
                        print(
                            "[FaceSwap] COTA ZeroGPU esgotada. "
                            "Solução GRÁTIS: crie token em huggingface.co/settings/tokens "
                            "e coloque HF_TOKEN no Render (Environment). "
                            "Sem token o enhance não roda e o rosto fica em baixa res.",
                            flush=True,
                        )
                        break  # não spamma 4x a mesma cota

            if last_err:
                print(f"[FaceSwap] CodeFormer failed: {last_err}", flush=True)
            return None

    def _bookbot_upscale_sync(self, image_bytes: bytes) -> bytes | None:
        """Fallback grátis se CodeFormer sem cota."""
        from gradio_client import Client, handle_file
        import base64 as b64

        space = "bookbot/Image-Upscaling-Playground"
        print(f"[FaceSwap] fallback upscale space={space}", flush=True)
        try:
            client = Client(space, **self._client_kwargs())
        except Exception as e:
            print(f"[FaceSwap] bookbot client fail: {e}", flush=True)
            return None

        with tempfile.TemporaryDirectory(prefix="face-up-") as tmp:
            img_path = Path(tmp) / "in.jpg"
            img_path.write_bytes(image_bytes)
            # API pede base64 string às vezes; handle_file costuma funcionar
            b64img = "data:image/jpeg;base64," + b64.b64encode(image_bytes).decode()
            for upscaler in ("modelx2", "modelx4", "RealESRGAN_x2", "RealESRGAN_x4plus"):
                for payload in (
                    lambda u=upscaler: client.predict(
                        handle_file(str(img_path)), u, api_name="/predict"
                    ),
                    lambda u=upscaler: client.predict(
                        b64img, u, api_name="/predict"
                    ),
                ):
                    try:
                        result = payload()
                        data = self._read_result_sync(result)
                        if data and len(data) > 1000:
                            print(
                                f"[FaceSwap] bookbot OK upscaler={upscaler} bytes={len(data)}",
                                flush=True,
                            )
                            return data
                    except Exception as e:
                        print(f"[FaceSwap] bookbot {upscaler}: {e}", flush=True)
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
        # base64 raw
        if len(result) > 200 and not result.startswith("/") and " " not in result[:50]:
            try:
                return base64.b64decode(result)
            except Exception:
                pass
        path = Path(result)
        return path.read_bytes() if path.is_file() else None
