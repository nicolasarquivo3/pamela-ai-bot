"""
Perchance image provider.

Por que a key do browser da invalid_key no Render:
  - userKey sozinha NAO basta
  - o request do browser manda cookies (cf_clearance etc.)
  - sem cookie, a API devolve {"status":"invalid_key"}

Env:
  PERCHANCE_USER_KEY   = hex da query userKey=
  PERCHANCE_COOKIES    = header Cookie completo do MESMO request generate
  PERCHANCE_CHANNELS   = lista separada por virgula
  PERCHANCE_CHANNEL    = canal preferido (vai pro final da fila)
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from app.images.models import ImageResult

DEFAULT_CHANNELS = [
    "ai-photo-generator",
    "image-generator-professional",
    "ai-text-to-image-generator",
    "5yf90s8rdo",
]


class PerchanceImageProvider:
    name = "perchance"

    def __init__(
        self,
        user_key: str | None = None,
        cookies: str | None = None,
        channel: str | None = None,
        channels: list[str] | None = None,
        timeout: int = 180,
        resolution: str = "512x768",
        guidance_scale: float = 7.0,
        negative_prompt: str = (
            "deformed, bad anatomy, extra limbs, blurry, low quality, "
            "distorted face, mutated hands, child, minor, cartoon, anime, cgi"
        ),
    ):
        self.user_key = (user_key or "").strip() or None
        self.cookies = (cookies or "").strip() or None
        self.timeout = int(timeout)
        self.resolution = resolution
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt
        self.base = "https://image-generation.perchance.org/api"

        ordered: list[str] = []
        if channels:
            for c in channels:
                c = (c or "").strip()
                if c and c not in ordered:
                    ordered.append(c)
        for c in DEFAULT_CHANNELS:
            if c not in ordered:
                ordered.append(c)
        if channel and str(channel).strip():
            ch = str(channel).strip()
            if ch in ordered:
                ordered = [x for x in ordered if x != ch]
            ordered.append(ch)
        self.channels = ordered

    async def available(self) -> bool:
        return bool(self.user_key)

    def _headers(self, channel: str) -> dict[str, str]:
        h = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": f"https://perchance.org/{channel}",
            "Origin": "https://perchance.org",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }
        if self.cookies:
            # aceita "Cookie: a=b" ou so "a=b; c=d"
            raw = self.cookies
            if raw.lower().startswith("cookie:"):
                raw = raw.split(":", 1)[1].strip()
            h["Cookie"] = raw
        return h

    async def generate(self, request, prompt: str) -> ImageResult:
        if not self.user_key:
            return ImageResult(
                False,
                error="perchance:missing_PERCHANCE_USER_KEY",
                provider="perchance",
            )

        if not self.cookies:
            print(
                "[IMAGE] Perchance AVISO: sem PERCHANCE_COOKIES "
                "(cf_clearance). Muitas vezes da invalid_key no Render.",
                flush=True,
            )

        seed = getattr(request, "seed", None)
        if seed is None:
            seed = -1
        else:
            seed = int(seed) % 2_147_483_647

        print(
            f"[IMAGE] Perchance: channels={self.channels} seed={seed} "
            f"key={self.user_key[:8]}... cookies={'yes' if self.cookies else 'no'}",
            flush=True,
        )

        last_err = "perchance_failed"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True
            ) as client:
                for ch in self.channels:
                    result = await self._try_channel(client, ch, prompt, seed)
                    if result.success:
                        return result
                    err = result.error or ""
                    last_err = err
                    if "invalid_key" in err or "not_verified" in err:
                        print(
                            "[IMAGE] Perchance: key/cookies invalidos — "
                            "renove USER_KEY + COOKIES no Render (mesmo request)",
                            flush=True,
                        )
                        break
                    print(
                        f"[IMAGE] Perchance channel={ch!r} falhou: {err} -> proximo",
                        flush=True,
                    )
        except Exception as e:
            last_err = f"perchance:{e}"
            print(f"[IMAGE] Perchance exception: {e}", flush=True)

        return ImageResult(False, error=last_err, provider="perchance")

    async def _try_channel(
        self,
        client: httpx.AsyncClient,
        channel: str,
        prompt: str,
        seed: int,
    ) -> ImageResult:
        headers = self._headers(channel)
        params = {
            "prompt": prompt[:1500],
            "negativePrompt": self.negative_prompt,
            "userKey": self.user_key,
            "__cache_bust": str(random.random()),
            "seed": str(seed if seed >= 0 else -1),
            "resolution": self.resolution,
            "guidanceScale": str(self.guidance_scale),
            "channel": channel,
            "subChannel": "public",
            "requestId": str(random.random()),
        }

        image_id = None
        last_err = f"perchance:{channel}:no_image"

        for method in ("GET", "POST"):
            try:
                if method == "GET":
                    r = await client.get(
                        f"{self.base}/generate",
                        params=params,
                        headers=headers,
                    )
                else:
                    r = await client.post(
                        f"{self.base}/generate",
                        params={
                            "userKey": self.user_key,
                            "requestId": params["requestId"],
                            "__cacheBust": params["__cache_bust"],
                        },
                        json={
                            "prompt": prompt[:1500],
                            "negativePrompt": self.negative_prompt,
                            "seed": seed if seed >= 0 else -1,
                            "resolution": self.resolution,
                            "guidanceScale": self.guidance_scale,
                            "channel": channel,
                            "subChannel": "public",
                            "generatorName": channel,
                        },
                        headers=headers,
                    )
            except Exception as e:
                last_err = f"perchance:{channel}:{method}:{e}"
                continue

            text = r.text or ""
            print(
                f"[IMAGE] Perchance {method} ch={channel} "
                f"HTTP {r.status_code} body[:160]={text[:160]!r}",
                flush=True,
            )

            if r.status_code == 403 or "Just a moment" in text:
                return ImageResult(
                    False,
                    error=f"perchance:{channel}:cloudflare_403",
                    provider="perchance",
                )
            if "invalid_key" in text or "not_verified" in text:
                return ImageResult(
                    False,
                    error="perchance:invalid_key",
                    provider="perchance",
                )
            if r.status_code not in (200, 202):
                last_err = f"perchance:{channel}:http_{r.status_code}"
                continue

            try:
                data: dict[str, Any] = r.json()
            except Exception:
                last_err = f"perchance:{channel}:bad_json"
                continue

            for _ in range(15):
                if data.get("imageId") or data.get("filePath"):
                    break
                status = str(data.get("status") or data.get("message") or "")
                if any(x in status.lower() for x in ("wait", "queue", "pending")):
                    await asyncio.sleep(4)
                    try:
                        r2 = await client.get(
                            f"{self.base}/generate",
                            params=params,
                            headers=headers,
                        )
                        data = r2.json()
                    except Exception:
                        break
                else:
                    break

            image_id = data.get("imageId") or data.get("filePath")
            if image_id:
                break
            if data.get("status") == "invalid_data":
                last_err = f"perchance:{channel}:invalid_data"
                break

        if not image_id:
            return ImageResult(False, error=last_err, provider="perchance")

        try:
            dl = await client.get(
                f"{self.base}/downloadTemporaryImage",
                params={"imageId": image_id},
                headers=headers,
            )
        except Exception as e:
            return ImageResult(
                False,
                error=f"perchance:{channel}:download:{e}",
                provider="perchance",
            )

        if dl.status_code != 200 or len(dl.content) < 5000:
            return ImageResult(
                False,
                error=f"perchance:{channel}:download_fail",
                provider="perchance",
            )

        raw = dl.content
        print(
            f"[IMAGE] Perchance ok channel={channel!r} bytes={len(raw)} id={image_id}",
            flush=True,
        )
        return ImageResult(
            success=True,
            provider=f"perchance:{channel}",
            image_bytes=raw,
            image_url=None,
        )
