"""
Pollinations.ai — geração FREE (sem key ou com POLLINATIONS_API_KEY).
GET image.pollinations.ai/prompt/{prompt}?seed=...
"""
from __future__ import annotations

import random
import time
import urllib.parse
from typing import Any

import httpx

from app.images.models import ImageResult


class PollinationsImageProvider:
    name = "pollinations"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "flux",
        timeout: int = 120,
        width: int = 768,
        height: int = 1024,
        max_retries: int = 4,
    ):
        self.api_key = (api_key or "").strip() or None
        self.model = model or "flux"
        self.timeout = int(timeout)
        self.width = int(width)
        self.height = int(height)
        self.max_retries = int(max_retries)

    async def available(self) -> bool:
        return True

    def _seed(self, request) -> int:
        s = getattr(request, "seed", None)
        if s is None:
            s = int(time.time() * 1000) ^ random.randint(1, 1_000_000)
        return int(s) % 2_147_483_647

    def _urls(self, prompt: str, seed: int) -> list[str]:
        enc = urllib.parse.quote(prompt[:1200], safe="")
        qs = (
            f"width={self.width}&height={self.height}"
            f"&seed={seed}&nologo=true&enhance=true&model={self.model}"
        )
        urls = [
            f"https://image.pollinations.ai/prompt/{enc}?{qs}",
        ]
        if self.api_key:
            urls.insert(
                0,
                f"https://gen.pollinations.ai/image/{enc}?{qs}&key={self.api_key}",
            )
        return urls

    async def generate(self, request, prompt: str) -> ImageResult:
        seed = self._seed(request)
        # seed extra jitter para nunca repetir
        seed = (seed + random.randint(0, 999_999)) % 2_147_483_647
        print(
            f"[IMAGE] Pollinations: starting model={self.model} seed={seed}",
            flush=True,
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
            "Referer": "https://pollinations.ai/",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err = "pollinations_failed"
        for attempt in range(self.max_retries):
            attempt_seed = (seed + attempt * 17_777 + random.randint(0, 9999)) % 2_147_483_647
            for url in self._urls(prompt, attempt_seed):
                try:
                    print(
                        f"[IMAGE] Pollinations: GET attempt={attempt+1} seed={attempt_seed}",
                        flush=True,
                    )
                    async with httpx.AsyncClient(
                        timeout=self.timeout, follow_redirects=True, headers=headers
                    ) as client:
                        r = await client.get(url)
                    ct = (r.headers.get("content-type") or "").lower()
                    data = r.content
                    if r.status_code == 429:
                        last_err = "pollinations:rate_limit_429"
                        print(f"[IMAGE] Pollinations 429 — wait", flush=True)
                        time.sleep(2 + attempt * 2)
                        continue
                    if r.status_code != 200:
                        last_err = f"pollinations:http_{r.status_code}"
                        print(
                            f"[IMAGE] Pollinations HTTP {r.status_code} {data[:200]!r}",
                            flush=True,
                        )
                        continue
                    if "image" not in ct and not (
                        data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"
                    ):
                        last_err = "pollinations:not_image"
                        continue
                    if len(data) < 8000:
                        last_err = "pollinations:too_small"
                        continue
                    print(
                        f"[IMAGE] Pollinations ok bytes={len(data)} seed={attempt_seed}",
                        flush=True,
                    )
                    return ImageResult(
                        success=True,
                        provider="pollinations",
                        image_bytes=data,
                        image_url=None
                    )
                except Exception as e:
                    last_err = f"pollinations:{e}"
                    print(f"[IMAGE] Pollinations error: {e}", flush=True)
            await _async_sleep(1.5 + attempt)

        return ImageResult(False, error=last_err, provider="pollinations")


async def _async_sleep(sec: float) -> None:
    import asyncio

    await asyncio.sleep(sec)
