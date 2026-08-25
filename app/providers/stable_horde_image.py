"""
Stable Horde — geração FREE distribuída (sem cartão).
API: https://stablehorde.net/api/
Anon key: 0000000000 (lento) | ou STABLE_HORDE_API_KEY grátis em stablehorde.net
"""
from __future__ import annotations

import asyncio
import base64
import random
import time
from typing import Any

import httpx

from app.images.models import ImageResult


# Modelos realistas com workers frequentes (ordem de preferência)
DEFAULT_MODELS = [
    "ICBINP - I Can't Believe It's Not Photography",
    "AbsoluteReality",
    "Realistic Vision",
    "Juggernaut XL",
    "majicMIX realistic",
    "Flux.1-Schnell fp8 (Compact)",
    "Epic Diffusion",
]


class StableHordeImageProvider:
    name = "stable_horde"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 180,
        width: int = 512,
        height: int = 768,
        steps: int = 25,
        cfg_scale: float = 7.0,
        models: list[str] | None = None,
        nsfw: bool = True,
    ):
        # key grátis no site; "0000000000" = anônimo
        self.api_key = (api_key or "").strip() or "0000000000"
        self.timeout = int(timeout)
        self.width = int(width) - (int(width) % 64)
        self.height = int(height) - (int(height) % 64)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.models = models or list(DEFAULT_MODELS)
        self.nsfw = bool(nsfw)
        self.base = "https://stablehorde.net/api/v2"

    async def available(self) -> bool:
        return True

    def _seed(self, request) -> int:
        s = getattr(request, "seed", None)
        if s is None:
            s = int(time.time() * 1000) ^ random.randint(1, 9_999_999)
        return abs(int(s)) % 2_147_483_647

    async def generate(self, request, prompt: str) -> ImageResult:
        seed = self._seed(request)
        seed = (seed + random.randint(0, 999_999)) % 2_147_483_647
        model = random.choice(self.models[:5])
        print(
            f"[IMAGE] StableHorde: start model={model!r} seed={seed}",
            flush=True,
        )

        payload: dict[str, Any] = {
            "prompt": prompt[:2000],
            "params": {
                "sampler_name": "k_euler_a",
                "cfg_scale": self.cfg_scale,
                "denoising_strength": 0.7,
                "seed": str(seed),
                "height": self.height,
                "width": self.width,
                "steps": self.steps,
                "n": 1,
                "karras": True,
            },
            "nsfw": self.nsfw,
            "censor_nsfw": False,
            "trusted_workers": False,
            "models": [model],
            "r2": True,
            "shared": False,
        }

        headers = {
            "apikey": self.api_key,
            "Client-Agent": "pamela-ai-bot:1.0:github.com/nicolasarquivo3/pamela-ai-bot",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base}/generate/async",
                    json=payload,
                    headers=headers,
                )
                if r.status_code not in (200, 202):
                    print(
                        f"[IMAGE] StableHorde submit HTTP {r.status_code}: {r.text[:400]}",
                        flush=True,
                    )
                    # tenta modelo generico
                    payload["models"] = []
                    r = await client.post(
                        f"{self.base}/generate/async",
                        json=payload,
                        headers=headers,
                    )
                    if r.status_code not in (200, 202):
                        return ImageResult(
                            False,
                            error=f"stable_horde:submit_{r.status_code}",
                            provider="stable_horde",
                        )

                data = r.json()
                job_id = data.get("id")
                if not job_id:
                    return ImageResult(
                        False,
                        error=f"stable_horde:no_id:{data}",
                        provider="stable_horde",
                    )
                print(f"[IMAGE] StableHorde job={job_id}", flush=True)

                # poll
                deadline = time.time() + self.timeout
                while time.time() < deadline:
                    await asyncio.sleep(3.5)
                    st = await client.get(
                        f"{self.base}/generate/status/{job_id}",
                        headers=headers,
                    )
                    if st.status_code != 200:
                        continue
                    body = st.json()
                    if body.get("faulted"):
                        return ImageResult(
                            False,
                            error="stable_horde:faulted",
                            provider="stable_horde",
                        )
                    if not body.get("done"):
                        wait = body.get("wait_time") or body.get("queue_position")
                        print(
                            f"[IMAGE] StableHorde wait queue={wait}",
                            flush=True,
                        )
                        continue

                    gens = body.get("generations") or []
                    if not gens:
                        return ImageResult(
                            False,
                            error="stable_horde:empty",
                            provider="stable_horde",
                        )
                    g0 = gens[0]
                    img = g0.get("img")
                    if not img:
                        return ImageResult(
                            False,
                            error="stable_horde:no_img",
                            provider="stable_horde",
                        )

                    # R2 URL ou base64
                    if str(img).startswith("http"):
                        ir = await client.get(img, timeout=60)
                        if ir.status_code != 200 or len(ir.content) < 5000:
                            return ImageResult(
                                False,
                                error="stable_horde:download_fail",
                                provider="stable_horde",
                            )
                        raw = ir.content
                    else:
                        raw = base64.b64decode(img)

                    print(
                        f"[IMAGE] StableHorde ok bytes={len(raw)} model={model}",
                        flush=True,
                    )
                    return ImageResult(
                        success=True,
                        provider=f"stable_horde:{model}",
                        image_bytes=raw,
                        image_url=None
                    )

                return ImageResult(
                    False,
                    error="stable_horde:timeout",
                    provider="stable_horde",
                )
        except Exception as e:
            print(f"[IMAGE] StableHorde exception: {e}", flush=True)
            return ImageResult(
                False, error=f"stable_horde:{e}", provider="stable_horde"
            )
