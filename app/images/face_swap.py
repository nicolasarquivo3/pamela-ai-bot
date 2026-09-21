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
        hf_enhance_fidelity: float = 0.75,
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
        self.hf_enhance_fidelity = float(hf_enhance_fidelity if hf_enhance_fidelity is not None else 0.75)
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

        # Multi-pessoa: forcar swap no rosto da MULHER
        original_target = target_bytes
        restore_boxes: list = []
        try:
            target_bytes, restore_boxes = await self._prefer_woman_faces(target_bytes)
        except Exception as e:
            print(f"[FaceSwap] prefer_woman skip: {e}", flush=True)
            target_bytes = original_target
            restore_boxes = []

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

                # restaura rostos de homens / nao-alvo
                if restore_boxes:
                    try:
                        output = self._restore_face_boxes(
                            output, original_target, restore_boxes
                        )
                    except Exception as e:
                        print(f"[FaceSwap] restore faces skip: {e}", flush=True)

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



    def _prepare_identity_reference(self, data: bytes, mode: str = "raw") -> bytes:
        """
        mode:
          raw   = referencia original (preferido — nao quebra deteccao)
          mild  = so reencode JPEG HQ (sem crop)
          crop  = crop cabeca so como fallback se raw falhar
        O crop agressivo anterior fazia "No faces detected" no tonyassi.
        """
        try:
            from PIL import Image
            import io
        except ImportError:
            return data

        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size

        if mode == "crop" and h >= int(w * 1.15):
            # crop mais generoso (nao corta o rosto)
            x0, x1 = int(w * 0.05), int(w * 0.95)
            y0, y1 = int(h * 0.0), int(h * 0.55)
            if y1 > y0 + 100 and x1 > x0 + 100:
                im = im.crop((x0, y0, x1, y1))
                print(f"[FaceSwap] ref crop mild {w}x{h} -> {im.size}", flush=True)
                w, h = im.size

        # raw/mild: no crop; so reencode
        side = max(w, h)
        if side > 1600:
            scale = 1400 / float(side)
            im = im.resize(
                (int(round(w * scale)), int(round(h * scale))),
                Image.Resampling.LANCZOS,
            )
            print(f"[FaceSwap] ref downscale -> {im.size}", flush=True)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        return buf.getvalue()

    def _prepare_target_for_identity(self, data: bytes) -> bytes:
        """Reencode target; evita target gigante. Sem crop (nao remover rosto)."""
        try:
            from PIL import Image
            import io
        except ImportError:
            return data
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        side = max(w, h)
        if side > 1800:
            scale = 1600 / float(side)
            im = im.resize(
                (int(round(w * scale)), int(round(h * scale))),
                Image.Resampling.LANCZOS,
            )
            print(f"[FaceSwap] target downscale -> {im.size}", flush=True)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        return buf.getvalue()


    def _gemini_keys(self) -> list[str]:
        keys = []
        try:
            from app.config import settings
            if getattr(settings, "gemini_api_key", None):
                keys.append(str(settings.gemini_api_key).strip())
            multi = getattr(settings, "gemini_api_keys", None) or ""
            for part in str(multi).split(","):
                p = part.strip()
                if p and p not in keys:
                    keys.append(p)
        except Exception:
            pass
        for env_k in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
            import os
            v = (os.getenv(env_k) or "").strip()
            if not v:
                continue
            for part in v.split(","):
                p = part.strip()
                if p and p not in keys:
                    keys.append(p)
        return keys

    async def _prefer_woman_faces(self, target_bytes: bytes) -> tuple[bytes, list]:
        """
        Se houver varias pessoas, mascara rostos NAO-femininos para o swap
        pegar sempre a mulher. Depois restauramos esses rostos no resultado.

        Retorna (target_preparado, lista de boxes normalizados 0-1 a restaurar).
        box = (x0, y0, x1, y1) em fracao da imagem.
        """
        try:
            faces = await self._detect_faces_gender(target_bytes)
        except Exception as e:
            print(f"[FaceSwap] woman-detect fail: {e}", flush=True)
            return target_bytes, []

        if not faces:
            print("[FaceSwap] woman-detect: 0 faces (segue normal)", flush=True)
            return target_bytes, []

        women = [f for f in faces if f.get("gender") == "female"]
        others = [f for f in faces if f.get("gender") != "female"]
        print(
            f"[FaceSwap] faces={len(faces)} women={len(women)} others={len(others)} "
            f"detail={faces}",
            flush=True,
        )

        if len(faces) <= 1:
            return target_bytes, []
        if not women:
            # sem mulher clara: nao mascara (evita swap errado pior)
            print("[FaceSwap] nenhuma mulher detectada — swap normal", flush=True)
            return target_bytes, []
        if not others:
            # so mulheres: pega a principal (maior area / mais central)
            print("[FaceSwap] so mulheres — prioriza rosto principal", flush=True)
            # ainda assim se >1 mulher, mascara as menores para trocar a principal
            if len(women) == 1:
                return target_bytes, []
            primary = self._pick_primary_woman(women)
            restore = [tuple(w["box"]) for w in women if w is not primary]
            masked = self._mask_face_boxes(target_bytes, restore)
            return masked, restore

        # multi: mulher(es) + homem(ns) → mascara homens (e mulheres extras)
        primary = self._pick_primary_woman(women)
        restore = []
        for f in faces:
            if f is primary:
                continue
            restore.append(tuple(f["box"]))
        masked = self._mask_face_boxes(target_bytes, restore)
        print(
            f"[FaceSwap] mascara {len(restore)} rosto(s) nao-alvo; "
            f"swap so na mulher primary={primary.get('box')}",
            flush=True,
        )
        return masked, restore

    def _pick_primary_woman(self, women: list) -> dict:
        """Maior rosto; empate → mais central."""
        def score(f):
            x0, y0, x1, y1 = f["box"]
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            center = 1.0 - ((cx - 0.5) ** 2 + (cy - 0.45) ** 2) ** 0.5
            return area * 2.0 + center
        return max(women, key=score)

    def _mask_face_boxes(self, image_bytes: bytes, boxes: list) -> bytes:
        """Cobre rostos nao-alvo com blur forte p/ o detector nao achar face."""
        if not boxes:
            return image_bytes
        try:
            from PIL import Image, ImageFilter, ImageDraw
            import io
        except ImportError:
            return image_bytes
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        for box in boxes:
            try:
                x0, y0, x1, y1 = box
            except Exception:
                continue
            # padding generoso
            px0 = max(0, int((x0 - 0.02) * w))
            py0 = max(0, int((y0 - 0.02) * h))
            px1 = min(w, int((x1 + 0.02) * w))
            py1 = min(h, int((y1 + 0.02) * h))
            if px1 <= px0 + 4 or py1 <= py0 + 4:
                continue
            region = im.crop((px0, py0, px1, py1))
            region = region.filter(ImageFilter.GaussianBlur(radius=28))
            # escurece um pouco p/ matar landmarks
            from PIL import ImageEnhance
            region = ImageEnhance.Brightness(region).enhance(0.55)
            im.paste(region, (px0, py0))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        return buf.getvalue()

    def _restore_face_boxes(
        self, swapped_bytes: bytes, original_bytes: bytes, boxes: list
    ) -> bytes:
        """Cola de volta os rostos originais (homens etc.) apos o swap na mulher."""
        if not boxes or not swapped_bytes or not original_bytes:
            return swapped_bytes
        try:
            from PIL import Image, ImageDraw
            import io
        except ImportError:
            return swapped_bytes
        out = Image.open(io.BytesIO(swapped_bytes)).convert("RGB")
        orig = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        # se tamanhos diferem (prep), redimensiona orig para out
        if orig.size != out.size:
            orig = orig.resize(out.size, Image.Resampling.LANCZOS)
        w, h = out.size
        for box in boxes:
            try:
                x0, y0, x1, y1 = box
            except Exception:
                continue
            # padding um pouco maior na restauracao
            px0 = max(0, int((x0 - 0.03) * w))
            py0 = max(0, int((y0 - 0.03) * h))
            px1 = min(w, int((x1 + 0.03) * w))
            py1 = min(h, int((y1 + 0.03) * h))
            if px1 <= px0 + 4 or py1 <= py0 + 4:
                continue
            patch = orig.crop((px0, py0, px1, py1))
            # mascara eliptica suave nas bordas
            mask = Image.new("L", (px1 - px0, py1 - py0), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(
                (2, 2, px1 - px0 - 3, py1 - py0 - 3),
                fill=255,
            )
            try:
                from PIL import ImageFilter
                mask = mask.filter(ImageFilter.GaussianBlur(radius=4))
            except Exception:
                pass
            out.paste(patch, (px0, py0), mask)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        print(f"[FaceSwap] restaurou {len(boxes)} rosto(s) nao-alvo", flush=True)
        return buf.getvalue()

    async def _detect_faces_gender(self, image_bytes: bytes) -> list:
        """
        Gemini Vision: lista faces com gender + box normalizado [x0,y0,x1,y1] 0..1.
        """
        import json
        import re as _re
        import base64 as _b64
        import httpx

        keys = self._gemini_keys()
        if not keys:
            print("[FaceSwap] woman-detect: sem GEMINI key", flush=True)
            return []

        raw = image_bytes[:3_500_000]
        b64 = _b64.b64encode(raw).decode("ascii")
        prompt = (
            "Analise a imagem. Liste TODAS as faces de pessoas adultas visiveis. "
            "Responda APENAS JSON valido, sem markdown, neste formato:\n"
            '{"faces":[{"gender":"female"|"male"|"unknown","box":[x0,y0,x1,y1]}]}\n'
            "box e normalizado 0 a 1 (fracao da largura/altura da imagem), "
            "x0,y0 canto superior esquerdo, x1,y1 inferior direito. "
            "gender=female para mulher, male para homem. "
            "Se so uma pessoa, ainda assim retorne 1 face. "
            "Se nao houver face, {\"faces\":[]}."
        )

        models = [
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash",
        ]
        try:
            from app.config import settings
            m = (getattr(settings, "gemini_model", None) or "").strip()
            if m:
                models = [m] + [x for x in models if x != m]
        except Exception:
            pass

        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 512,
            },
        }

        text_out = ""
        async with httpx.AsyncClient(timeout=60) as client:
            for key in keys[:3]:
                for model in models[:5]:
                    for ver in ("v1beta", "v1"):
                        url = (
                            f"https://generativelanguage.googleapis.com/{ver}/"
                            f"models/{model}:generateContent"
                        )
                        try:
                            r = await client.post(
                                url,
                                headers={
                                    "x-goog-api-key": key,
                                    "Content-Type": "application/json",
                                },
                                json=payload,
                            )
                            if r.status_code != 200:
                                continue
                            data = r.json()
                            cands = data.get("candidates") or []
                            if not cands:
                                continue
                            parts_out = (
                                (cands[0].get("content") or {}).get("parts") or []
                            )
                            text_out = "".join(
                                (p.get("text") or "") for p in parts_out
                            ).strip()
                            if text_out:
                                print(
                                    f"[FaceSwap] woman-detect model={model} ok",
                                    flush=True,
                                )
                                break
                        except Exception as e:
                            print(f"[FaceSwap] woman-detect err: {e}", flush=True)
                            continue
                    if text_out:
                        break
                if text_out:
                    break

        if not text_out:
            return []

        # extrai JSON
        m = _re.search(r"\{[\s\S]*\}", text_out)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return []
        faces_raw = obj.get("faces") or []
        faces = []
        for f in faces_raw:
            if not isinstance(f, dict):
                continue
            g = (f.get("gender") or "unknown").lower().strip()
            if g in ("woman", "girl", "f", "mulher", "female"):
                g = "female"
            elif g in ("man", "boy", "m", "homem", "male"):
                g = "male"
            else:
                g = "unknown"
            box = f.get("box") or f.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in box]
            except Exception:
                continue
            # clamp
            x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
            x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            faces.append({"gender": g, "box": [x0, y0, x1, y1]})
        return faces

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
            raw_ref = self.reference_path.read_bytes()
            try:
                tgt = self._prepare_target_for_identity(target_bytes)
            except Exception as e:
                print(f"[FaceSwap] target prep skip: {e}", flush=True)
                tgt = target_bytes
            target.write_bytes(tgt)

            # RAW primeiro (o crop agressivo quebrava: No faces detected)
            ref_variants = []
            for mode in ("raw", "mild", "crop"):
                try:
                    rb = self._prepare_identity_reference(raw_ref, mode=mode)
                    ref_variants.append((mode, rb))
                except Exception as e:
                    print(f"[FaceSwap] ref mode={mode} skip: {e}", flush=True)
            if not ref_variants:
                ref_variants = [("raw", raw_ref)]

            last_err = None
            for space, api in swap_spaces:
                try:
                    client = Client(space, **self._client_kwargs())
                    print(f"[FaceSwap] swap space={space} api={api}", flush=True)
                    for mode, ref_bytes in ref_variants:
                        source.write_bytes(ref_bytes)
                        print(
                            f"[FaceSwap] try ref_mode={mode} ref={len(ref_bytes)} "
                            f"target={len(tgt)}",
                            flush=True,
                        )
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
                                        f"[FaceSwap] swap OK space={space} "
                                        f"ref_mode={mode} try={i+1} bytes={len(data)}",
                                        flush=True,
                                    )
                                    return data
                            except Exception as e:
                                last_err = e
                                print(
                                    f"[FaceSwap] swap {space} mode={mode} "
                                    f"try={i+1}: {e}",
                                    flush=True,
                                )
                                continue
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

            # fidelity ALTO = preserva mais o rosto do swap (identidade)
            # fidelity BAIXO = "melhora" generica e pode mudar o rosto
            fid_hi = min(0.9, max(fidelity, 0.75))
            attempts = [
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True, True, True, upscale, fid_hi,
                    api_name="/inference",
                ),
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True, False, True, upscale, fid_hi,  # sem background enhance
                    api_name="/inference",
                ),
                lambda: client.predict(
                    image=handle_file(str(img_path)),
                    face_align=True,
                    background_enhance=False,
                    face_upsample=True,
                    upscale=upscale,
                    codeformer_fidelity=fid_hi,
                    api_name="/inference",
                ),
                lambda: client.predict(
                    handle_file(str(img_path)),
                    True, True, True, 1, fid_hi,
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
