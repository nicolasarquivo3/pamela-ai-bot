import asyncio
import base64
import io
import tempfile
from pathlib import Path

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """
    Face swap + reforço de qualidade do rosto.

    HyperSwap nativo e 256px; qualidade sobe com:
      1) target grande (min_side)
      2) pixel_boost (512/768/1024) no Space FaceFusion
      3) face restore (gfpgan / gpen / codeformer)
      4) pos-upscale 2x + unsharp na regiao do rosto (Pillow)
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
        hf_restore_strength: float = 0.8,
        hf_pixel_boost: str = "1024x1024",
        post_upscale: float = 2.0,
        min_target_side: int = 1024,
        max_output_side: int = 2048,
        jpeg_quality: int = 97,
        face_sharpen: float = 1.8,
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
        self.hf_space = hf_space
        self.hf_api_name = hf_api_name
        self.hf_token = hf_token
        self.hf_swap_model = hf_swap_model
        self.hf_target_index = int(hf_target_index)
        self.hf_restore_model = hf_restore_model or "none"
        self.hf_restore_strength = float(hf_restore_strength)
        self.hf_pixel_boost = (hf_pixel_boost or "1024x1024").strip()
        self.post_upscale = float(post_upscale or 1.0)
        self.min_target_side = int(min_target_side or 0)
        self.max_output_side = int(max_output_side or 2048)
        self.jpeg_quality = int(jpeg_quality or 95)
        self.face_sharpen = float(face_sharpen or 1.0)
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
                        before = len(output)
                        output = self._enhance_output_bytes(output)
                        print(
                            f"[FaceSwap] HQ enhance ok provider={name} "
                            f"bytes {before}->{len(output)}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[FaceSwap] enhance_output fail: {e}", flush=True)
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
        """Sobe o target para o rosto ocupar mais pixels no swap."""
        if self.min_target_side <= 0:
            return data
        try:
            from PIL import Image
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        side = max(w, h)
        if side < self.min_target_side:
            scale = self.min_target_side / float(side)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
            print(
                f"[FaceSwap] target upscale {w}x{h} -> {nw}x{nh}",
                flush=True,
            )
        else:
            print(f"[FaceSwap] target ok {w}x{h}", flush=True)
        return self._to_jpeg_bytes(im)

    def _prepare_reference_bytes(self, data: bytes) -> bytes:
        try:
            from PIL import Image
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        side = max(w, h)
        # referencia facial: no minimo ~1024 no maior lado
        want = max(1024, self.min_target_side)
        if side < want:
            scale = want / float(side)
            im = im.resize(
                (int(round(w * scale)), int(round(h * scale))),
                Image.Resampling.LANCZOS,
            )
            print(f"[FaceSwap] reference upscale -> {im.size}", flush=True)
        else:
            print(f"[FaceSwap] reference ok {w}x{h}", flush=True)
        return self._to_jpeg_bytes(im)

    def _enhance_output_bytes(self, data: bytes) -> bytes:
        """
        Upscale 2x + nitidez forte na regiao tipica do rosto (retrato).
        Isso mitiga o rosto 256px 'borrado' que o HyperSwap devolve.
        """
        from PIL import Image, ImageFilter, ImageEnhance, ImageOps

        im = Image.open(io.BytesIO(data)).convert("RGB")
        w0, h0 = im.size
        print(f"[FaceSwap] raw output {w0}x{h0}", flush=True)

        scale = float(self.post_upscale or 1.0)
        if scale < 1.5:
            scale = 1.5  # minimo util

        # 1) upscale geral
        nw = int(round(w0 * scale))
        nh = int(round(h0 * scale))
        m = max(nw, nh)
        if m > self.max_output_side:
            r = self.max_output_side / float(m)
            nw, nh = max(1, int(round(nw * r))), max(1, int(round(nh * r)))
        if nw != w0 or nh != h0:
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
            print(f"[FaceSwap] output upscale {w0}x{h0} -> {nw}x{nh}", flush=True)

        w, h = im.size

        # 2) unsharp global leve
        im = im.filter(
            ImageFilter.UnsharpMask(radius=1.6, percent=140, threshold=2)
        )

        # 3) reforco EXTRA na faixa do rosto (centro-superior em retrato)
        #    x: 15%-85%  y: 5%-55%  — cobre rosto na maioria das fotos do bot
        fx0, fy0 = int(w * 0.12), int(h * 0.04)
        fx1, fy1 = int(w * 0.88), int(h * 0.58)
        if fx1 > fx0 + 40 and fy1 > fy0 + 40:
            face = im.crop((fx0, fy0, fx1, fy1))
            # upscale local interno 1.25x e volta (alucina um pouco de detalhe)
            fw, fh = face.size
            face_hi = face.resize(
                (int(fw * 1.25), int(fh * 1.25)), Image.Resampling.LANCZOS
            )
            face_hi = face_hi.resize((fw, fh), Image.Resampling.LANCZOS)
            face_hi = face_hi.filter(
                ImageFilter.UnsharpMask(radius=2.2, percent=180, threshold=1)
            )
            face_hi = ImageEnhance.Sharpness(face_hi).enhance(
                max(1.0, self.face_sharpen)
            )
            face_hi = ImageEnhance.Contrast(face_hi).enhance(1.08)
            # blend para nao ficar artificial
            face_blend = Image.blend(face, face_hi, 0.72)
            im.paste(face_blend, (fx0, fy0))
            print(
                f"[FaceSwap] face-region sharpen box=({fx0},{fy0})-({fx1},{fy1}) "
                f"sharp={self.face_sharpen}",
                flush=True,
            )

        # 4) nitidez global final controlada
        im = ImageEnhance.Sharpness(im).enhance(1.15)
        im = ImageEnhance.Color(im).enhance(1.05)

        w2, h2 = im.size
        print(f"[FaceSwap] final output {w2}x{h2} q={self.jpeg_quality}", flush=True)
        return self._to_jpeg_bytes(im)

    def _to_jpeg_bytes(self, im) -> bytes:
        buf = io.BytesIO()
        # PNG internamente e melhor, mas Telegram espera jpg no filename;
        # usamos JPEG alta qualidade + chroma full (subsampling=0)
        im.save(
            buf,
            format="JPEG",
            quality=max(90, min(98, self.jpeg_quality)),
            optimize=True,
            subsampling=0,
            progressive=True,
        )
        return buf.getvalue()

    async def _huggingface_swap(self, target_bytes: bytes) -> bytes | None:
        return await asyncio.to_thread(self._huggingface_swap_sync, target_bytes)

    def _huggingface_swap_sync(self, target_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        model = self.hf_swap_model
        model_alt = model.replace(".onnx", "") if model.endswith(".onnx") else f"{model}.onnx"
        # preferir variantes 512 se o space listar
        model_candidates = [
            model,
            model_alt,
            "simswap_unofficial_512",
            "simswap_unofficial_512.onnx",
            "hyperswap_1b_256",
            "hyperswap_1b_256.onnx",
            "hyperswap_1c_256",
            "hyperswap_1c_256.onnx",
        ]
        # unique preserve order
        seen = set()
        models = []
        for m in model_candidates:
            if m not in seen:
                seen.add(m)
                models.append(m)

        restores = [
            self.hf_restore_model,
            "gfpgan_1.4",
            "gpen_bfr_512",
            "gpen_bfr_1024",
            "codeformer",
            "gfpgan_1.3",
            "none",
        ]
        seen_r = set()
        restores = [r for r in restores if r and not (r in seen_r or seen_r.add(r))]

        boosts = [
            self.hf_pixel_boost,
            "1024x1024",
            "768x768",
            "512x512",
            "256x256",
        ]
        seen_b = set()
        boosts = [b for b in boosts if b and not (b in seen_b or seen_b.add(b))]

        strength = self.hf_restore_strength
        idx = self.hf_target_index

        with tempfile.TemporaryDirectory(prefix="face-swap-") as tmp:
            source = Path(tmp) / "source.jpg"
            target = Path(tmp) / "target.jpg"
            ref = self.reference_path.read_bytes()
            try:
                ref = self._prepare_reference_bytes(ref)
            except Exception as e:
                print(f"[FaceSwap] ref prep: {e}", flush=True)
            source.write_bytes(ref)
            target.write_bytes(target_bytes)

            client_kwargs = {}
            if self.hf_token:
                client_kwargs["token"] = self.hf_token

            client = Client(self.hf_space, **client_kwargs)
            print(
                f"[FaceSwap] HF space={self.hf_space} api={self.hf_api_name} "
                f"model={model} restore={self.hf_restore_model} boost={self.hf_pixel_boost}",
                flush=True,
            )

            attempts = []

            # Ordem: boost alto + restore bom primeiro
            for boost in boosts[:3]:
                for restore in restores[:4]:
                    for mname in models[:4]:
                        def _mk(m=mname, r=restore, b=boost):
                            return lambda: client.predict(
                                handle_file(str(source)),
                                handle_file(str(target)),
                                idx,
                                m,
                                r,
                                strength,
                                b,
                                api_name=self.hf_api_name,
                            )
                        attempts.append(_mk())

            # sem boost
            for restore in restores[:3]:
                for mname in models[:3]:
                    def _mk2(m=mname, r=restore):
                        return lambda: client.predict(
                            handle_file(str(source)),
                            handle_file(str(target)),
                            idx,
                            m,
                            r,
                            strength,
                            api_name=self.hf_api_name,
                        )
                    attempts.append(_mk2())

            # kwargs / apis simples
            attempts.extend(
                [
                    lambda: client.predict(
                        handle_file(str(source)),
                        handle_file(str(target)),
                        idx,
                        model,
                        self.hf_restore_model,
                        strength,
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
            # limita tentativas longas (Spaces lentos) — top 12 + 6 fallbacks
            max_try = min(len(attempts), 18)
            for i, attempt in enumerate(attempts[:max_try]):
                try:
                    result = attempt()
                    data = self._read_result_sync(result)
                    if data:
                        print(f"[FaceSwap] sucesso na tentativa {i+1}", flush=True)
                        # log dim
                        try:
                            from PIL import Image
                            im = Image.open(io.BytesIO(data))
                            print(
                                f"[FaceSwap] swap result dim={im.size} bytes={len(data)}",
                                flush=True,
                            )
                        except Exception:
                            pass
                        return data
                except Exception as e:
                    last_err = e
                    print(f"[FaceSwap] tentativa {i+1} falhou: {e}", flush=True)
                    continue

            if last_err:
                raise last_err
            return None

    async def _replicate_swap(self, target_bytes: bytes) -> bytes | None:
        if not self.replicate_token:
            return None
        source_bytes = self.reference_path.read_bytes()
        try:
            source_bytes = self._prepare_reference_bytes(source_bytes)
        except Exception:
            pass

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

            for _ in range(90):
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
