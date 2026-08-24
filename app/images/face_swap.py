import asyncio
import base64
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.images.models import ImageResult


class FaceSwapService:
    """Runs a post-generation face swap against the configured character reference.

    Free-first order:
      1. Hugging Face Gradio Space (public/private, CPU-first)
      2. Optional Replicate fallback

    When required=True, an image is never returned to the user unless the swap
    succeeded. This is important when the caller expects the reference face to
    be enforced rather than merely requested in the generation prompt.
    """

    def __init__(
        self,
        reference_path: str,
        required: bool = True,
        provider: str = "huggingface",
        hf_space: str = "V0pr0S/FaceFusion-Face-Swap-Hyperswap",
        hf_api_name: str = "/generate_image",
        hf_token: str | None = None,
        hf_swap_model: str = "hyperswap_1b_256.onnx",
        hf_target_index: int = 0,
        hf_restore_model: str = "none",
        hf_restore_strength: float = 0.7,
        replicate_token: str | None = None,
        replicate_version: str = "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34",
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
        self.hf_restore_model = hf_restore_model
        self.hf_restore_strength = float(hf_restore_strength)
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
            return generated if not self.required else ImageResult(False, error="face_swap_disabled")

        target_bytes = await self._get_image_bytes(generated)
        if not target_bytes:
            return ImageResult(False, error="face_swap_target_unavailable") if self.required else generated

        providers = self._provider_order()
        errors: list[str] = []
        for name in providers:
            try:
                if name == "huggingface":
                    output = await self._huggingface_swap(target_bytes)
                else:
                    output = await self._replicate_swap(target_bytes)
                if output:
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

        if self.required:
            return ImageResult(False, provider=generated.provider, error="; ".join(errors) or "face_swap_failed")
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
        # gradio_client is synchronous; keep it off FastAPI's event loop.
        return await asyncio.to_thread(self._huggingface_swap_sync, target_bytes)

    def _huggingface_swap_sync(self, target_bytes: bytes) -> bytes | None:
        from gradio_client import Client, handle_file

        with tempfile.TemporaryDirectory(prefix="face-swap-") as tmp:
            source = Path(tmp) / "source.jpg"
            target = Path(tmp) / "target.jpg"
            source.write_bytes(self.reference_path.read_bytes())
            target.write_bytes(target_bytes)

            client_kwargs = {}
            if self.hf_token:
                client_kwargs["token"] = self.hf_token
            client = Client(self.hf_space, **client_kwargs)
            result = client.predict(
                handle_file(str(source)),
                handle_file(str(target)),
                self.hf_target_index,
                self.hf_swap_model,
                self.hf_restore_model,
                self.hf_restore_strength,
                api_name=self.hf_api_name,
            )
            return self._read_result_sync(result)

    async def _replicate_swap(self, target_bytes: bytes) -> bytes | None:
        if not self.replicate_token:
            return None
        source_bytes = self.reference_path.read_bytes()
        if len(source_bytes) > 256 * 1024 or len(target_bytes) > 256 * 1024:
            # Replicate recommends hosted files above 256 KB. The project does
            # not expose an upload bucket, so data URLs are used for small files
            # and a clear error is returned for larger ones.
            raise RuntimeError("replicate_data_uri_limit")

        def data_uri(data: bytes, mime: str = "image/jpeg") -> str:
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

        headers = {
            "Authorization": f"Bearer {self.replicate_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        payload = {
            "version": self.replicate_version,
            "input": {
                "swap_image": data_uri(source_bytes),
                "input_image": data_uri(target_bytes),
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
            response.raise_for_status()
            prediction = response.json()
            output = prediction.get("output")
            if not output and prediction.get("status") not in {"succeeded", "failed", "canceled"}:
                prediction_id = prediction.get("id")
                if prediction_id:
                    for _ in range(30):
                        await asyncio.sleep(2)
                        poll = await client.get(
                            f"https://api.replicate.com/v1/predictions/{prediction_id}",
                            headers={"Authorization": f"Bearer {self.replicate_token}"},
                        )
                        poll.raise_for_status()
                        prediction = poll.json()
                        output = prediction.get("output")
                        if prediction.get("status") in {"succeeded", "failed", "canceled"}:
                            break
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
