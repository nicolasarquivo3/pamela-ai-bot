"""
Busca fotos reais no Pexels (100% free) e devolve a melhor URL
para face swap.
"""
from __future__ import annotations

import random
from typing import Any

import httpx


class PexelsSearchService:
    name = "pexels"

    def __init__(
        self,
        api_key: str | None,
        timeout: int = 30,
        per_page: int = 15,
        orientation: str = "portrait",
    ):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.per_page = min(int(per_page), 80)
        self.orientation = orientation
        self.base_url = "https://api.pexels.com/v1/search"

    async def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = (query or "").strip()
        if not q:
            q = "beautiful woman portrait natural light"

        if "woman" not in q.lower() and "girl" not in q.lower() and "female" not in q.lower():
            q = f"beautiful young woman {q} portrait"

        params = {
            "query": q,
            "per_page": self.per_page,
            "orientation": self.orientation,
            "size": "large",
        }
        headers = {"Authorization": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(self.base_url, params=params, headers=headers)

            if r.status_code == 429:
                print("[PEXELS] rate limit", flush=True)
                return None
            if r.status_code != 200:
                print(f"[PEXELS] status {r.status_code}: {r.text[:200]}", flush=True)
                return None

            data = r.json()
            photos = data.get("photos") or []
            if not photos:
                print(f"[PEXELS] zero results for: {q}", flush=True)
                return None

            random.shuffle(photos)
            photo = photos[0]

            src = photo.get("src") or {}
            url = (
                src.get("large2x")
                or src.get("large")
                or src.get("original")
                or src.get("medium")
            )
            if not url:
                return None

            return {
                "url": url,
                "photographer": photo.get("photographer") or "Pexels",
                "photo_id": photo.get("id"),
                "alt": photo.get("alt") or q,
                "query": q,
            }
        except Exception as e:
            print(f"[PEXELS] error: {e}", flush=True)
            return None
