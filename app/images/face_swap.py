import asyncio
import base64
import tempfile
from pathlib import Path

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """
    Caminho GRÁTIS corrigido (log real):

    1) Swap em tonyassi/face-swap (API simples: src+dest) — confiável
    2) Enhance em sczhou/CodeFormer com api_name=/inference
       (face_upsample neural — NÃO é LANCZOS)

    O log mostrou:
    - restore no tonyassi NÃO existe (só 2 imagens) → tentativas com gfpgan eram no-op
    - CodeFormer falhava porque usávamos /predict (API real é /inference)
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
        self.hf_token = hf_token
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
                if self.hf_enhance_enabled and self.hf_enhance_space:
                    try:
                        enhanced = await self._codeformer_enhance(output)
                        if enhanced:
                            print(
                                f"[FaceSwap] CodeFormer OK bytes {len(output)}->{len(enhanced)}",
                                flush=True,
                            )
                            output = enhanced
                            tag = f"{name}+codeformer"
                        else:
                            print("[FaceSwap] CodeFormer sem output — mantém swap", flush=True)
                    except Exception as e:
                        print(f"[FaceSwap] CodeFormer skip: {e}", flush=True)

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

    def _client_kwargs(self) -> dict:
        kw = {}
        if self.hf_token:
            kw["token"] = self.hf_token
        return kw

    def _huggingface_swap_sync(self, target_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        # Spaces de swap simples (2 imagens) — tonyassi é o que o log usa de fato
        swap_spaces = []
        if self.hf_space:
            swap_spaces.append((self.hf_space, self.hf_api_name))
        # sempre tenta tonyassi se ainda não for ele (confiável + grátis)
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
                    print(
                        f"[FaceSwap] swap space={space} api={api}",
                        flush=True,
                    )
                    attempts = [
                        # tonyassi API real
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
                            print(
                                f"[FaceSwap] swap {space} try={i+1}: {e}",
                                flush=True,
                            )
                except Exception as e:
                    last_err = e
                    print(f"[FaceSwap] swap space fail {space}: {e}", flush=True)

            if last_err:
                raise last_err
            return None

    async def _codeformer_enhance(self, image_bytes: bytes) -> bytes | None:
        return await asyncio.to_thread(self._codeformer_enhance_sync, image_bytes)

    def _codeformer_enhance_sync(self, image_bytes: bytes) -> bytes | None:
        """
        sczhou/CodeFormer API real (view_api):
          /inference(
            image,
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=2,
            codeformer_fidelity=0.5,
          ) -> output
        """
        from gradio_client import Client, handle_file

        space = self.hf_enhance_space or "sczhou/CodeFormer"
        api = self.hf_enhance_api_name or "/inference"
        upscale = max(1, min(4, int(self.hf_enhance_upscale or 2)))
        fidelity = float(self.hf_enhance_fidelity or 0.5)

        with tempfile.TemporaryDirectory(prefix="face-enhance-") as tmp:
            img_path = Path(tmp) / "in.jpg"
            img_path.write_bytes(image_bytes)

            print(
                f"[FaceSwap] CodeFormer space={space} api={api} "
                f"upscale={upscale} fidelity={fidelity}",
                flush=True,
            )
            client = Client(space, **self._client_kwargs())

            attempts = [
                # assinatura oficial
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True,   # face_align
                    True,   # background_enhance
                    True,   # face_upsample  << neural no rosto
                    upscale,
                    fidelity,
                    api_name="/inference",
                ),
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True,
                    True,
                    True,
                    upscale,
                    fidelity,
                    api_name=api,
                ),
                # upscale 1 se 2 falhar por memória
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True,
                    True,
                    True,
                    1,
                    fidelity,
                    api_name="/inference",
                ),
                # kwargs
                lambda: client.predict(
                    image=handle_file(str(img_path)),
                    face_align=True,
                    background_enhance=True,
                    face_upsample=True,
                    upscale=upscale,
                    codeformer_fidelity=fidelity,
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
                    print(f"[FaceSwap] CodeFormer try={i+1}: {e}", flush=True)

            if last_err:
                print(f"[FaceSwap] CodeFormer all failed: {last_err}", flush=True)
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
