import asyncio
import base64
import io
import tempfile
from pathlib import Path

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """
    Face swap free-first:
      1. Hugging Face Gradio (FaceFusion / Hyperswap)
      2. Replicate (opcional)

    Qualidade do rosto / imagem:
      - modelo hyperswap_1b_256 (base 256; qualidade sobe com pixel_boost)
      - face restore (gfpgan / codeformer / gpen)
      - pixel_boost 512/768/1024 se o Space aceitar
      - upscale pos-swap (Pillow LANCZOS) da imagem inteira

    Source = rosto da personagem (reference)
    Target = foto do album / gerada / web
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
        hf_restore_strength: float = 0.65,
        hf_pixel_boost: str = "512x512",
        post_upscale: float = 1.5,
        min_target_side: int = 768,
        max_output_side: int = 2048,
        jpeg_quality: int = 95,
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
        self.hf_restore_model = hf_restore_model or "none"
        self.hf_restore_strength = float(hf_restore_strength)
        self.hf_pixel_boost = (hf_pixel_boost or "512x512").strip()
        self.post_upscale = float(post_upscale or 1.0)
        self.min_target_side = int(min_target_side or 0)
        self.max_output_side = int(max_output_side or 2048)
        self.jpeg_quality = int(jpeg_quality or 95)
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

        # Prepara target em resolucao decente ANTES do swap
        try:
            target_bytes = self._prepare_target_bytes(target_bytes)
        except Exception as e:
            print(f"[FaceSwap] prepare_target skip: {e}", flush=True)

        providers = self._provider_order()
        errors: list[str] = []
        for name in providers:
            try:
                if name == "huggingface":
                    output = await self._huggingface_swap(target_bytes)
                else:
                    output = await self._replicate_swap(target_bytes)
                if output:
                    try:
                        output = self._enhance_output_bytes(output)
                    except Exception as e:
                        print(f"[FaceSwap] enhance_output skip: {e}", flush=True)
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

    def _prepare_target_bytes(self, data: bytes) -> bytes:
        """Garante lado minimo no target (rosto maior = swap melhor)."""
        if self.min_target_side <= 0:
            return data
        try:
            from PIL import Image
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        side = max(w, h)
        if side >= self.min_target_side:
            # so re-encode em qualidade alta
            return self._to_jpeg_bytes(im)
        scale = self.min_target_side / float(side)
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        print(
            f"[FaceSwap] target upscale {w}x{h} -> {nw}x{nh} (min_side={self.min_target_side})",
            flush=True,
        )
        return self._to_jpeg_bytes(im)

    def _enhance_output_bytes(self, data: bytes) -> bytes:
        """Upscale pos-swap da imagem inteira + JPEG alta qualidade."""
        try:
            from PIL import Image
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        scale = float(self.post_upscale or 1.0)
        if scale > 1.01:
            nw = min(int(w * scale), self.max_output_side)
            nh = min(int(h * scale), self.max_output_side)
            # mantem proporcao se bater no max
            ratio = min(nw / w, nh / h)
            nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
            if nw > w or nh > h:
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
                print(
                    f"[FaceSwap] output upscale {w}x{h} -> {nw}x{nh} (x{scale})",
                    flush=True,
                )
        # clamp max side
        w2, h2 = im.size
        m = max(w2, h2)
        if m > self.max_output_side:
            r = self.max_output_side / float(m)
            im = im.resize((max(1, int(w2 * r)), max(1, int(h2 * r))), Image.Resampling.LANCZOS)
        return self._to_jpeg_bytes(im)

    def _to_jpeg_bytes(self, im) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True, subsampling=0)
        return buf.getvalue()

    async def _huggingface_swap(self, target_bytes: bytes) -> bytes | None:
        return await asyncio.to_thread(self._huggingface_swap_sync, target_bytes)

    def _huggingface_swap_sync(self, target_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        # nomes de modelo com e sem .onnx (Spaces variam)
        model = self.hf_swap_model
        model_alt = model.replace(".onnx", "") if model.endswith(".onnx") else f"{model}.onnx"
        restore = self.hf_restore_model
        strength = self.hf_restore_strength
        boost = self.hf_pixel_boost
        idx = self.hf_target_index

        with tempfile.TemporaryDirectory(prefix="face-swap-") as tmp:
            source = Path(tmp) / "source.jpg"
            target = Path(tmp) / "target.jpg"
            # referencia: se muito pequena, sobe um pouco
            ref = self.reference_path.read_bytes()
            try:
                ref = self._prepare_reference_bytes(ref)
            except Exception:
                pass
            source.write_bytes(ref)
            target.write_bytes(target_bytes)

            client_kwargs = {}
            if self.hf_token:
                client_kwargs["token"] = self.hf_token

            client = Client(self.hf_space, **client_kwargs)
            print(
                f"[FaceSwap] HF space={self.hf_space} api={self.hf_api_name} "
                f"model={model} restore={restore} boost={boost}",
                flush=True,
            )

            # Tentativas: do mais completo (boost+restore) ao mais simples
            attempts = [
                # FaceFusion-like: source, target, index, model, restore, strength, pixel_boost
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model,
                    restore,
                    strength,
                    boost,
                    api_name=self.hf_api_name,
                ),
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model_alt,
                    restore,
                    strength,
                    boost,
                    api_name=self.hf_api_name,
                ),
                # sem boost
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model,
                    restore,
                    strength,
                    api_name=self.hf_api_name,
                ),
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model_alt,
                    restore,
                    strength,
                    api_name=self.hf_api_name,
                ),
                # gfpgan alternativo se restore falhar no space
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model,
                    "codeformer",
                    strength,
                    api_name=self.hf_api_name,
                ),
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    idx,
                    model,
                    "gpen_bfr_512",
                    strength,
                    api_name=self.hf_api_name,
                ),
                # kwargs named
                lambda: client.predict(
                    src_img=handle_file(str(source)),
                    dest_img=handle_file(str(target)),
                    api_name=self.hf_api_name,
                ),
                # simples 2 args
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    api_name=self.hf_api_name,
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
                lambda: client.predict(
                    handle_file(str(source)),
                    handle_file(str(target)),
                    api_name="/generate_image",
                ),
            ]

            last_err = None
            for i, attempt in enumerate(attempts):
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

    def _prepare_reference_bytes(self, data: bytes) -> bytes:
        """Referencia facial em boa resolucao (lado min ~768 se menor)."""
        try:
            from PIL import Image
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        side = max(w, h)
        # referencia ja 923x1536 no repo — so reencode alta qualidade
        if side < 512:
            scale = 768 / float(side)
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            print(f"[FaceSwap] reference upscale -> {im.size}", flush=True)
        return self._to_jpeg_bytes(im)

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
            data = await self._download_output(output)
            if data:
                try:
                    data = self._enhance_output_bytes(data)
                except Exception:
                    pass
            return data

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
