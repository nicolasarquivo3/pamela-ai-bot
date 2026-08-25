"""Pixabay — PIXABAY_API_KEY. Respeita OUTFIT."""
from __future__ import annotations

from typing import Any
import random
import httpx

from app.images.outfit import outfit_from_scene
from app.images.recent_guard import RECENT


class PixabaySearchService:
    name = "pixabay"

    def __init__(self, api_key: str | None, timeout: int = 30, per_page: int = 20):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.per_page = min(int(per_page), 200)
        self.base_url = "https://pixabay.com/api/"

    async def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None
        q = outfit_from_scene(query)
        print(f"[PIXABAY] query: {q}", flush=True)
        params = {
            "key": self.api_key,
            "q": q,
            "image_type": "photo",
            "orientation": "vertical",
            "safesearch": "false",
            "per_page": self.per_page,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(self.base_url, params=params)
            if r.status_code != 200:
                return None
            hits = (r.json() or {}).get("hits") or []
            if not hits:
                return None
            random.shuffle(hits)
            for hit in hits[:8]:
                if RECENT.seen(photo_id=hit.get("id")):
                    continue
                url = hit.get("largeImageURL") or hit.get("fullHDURL") or hit.get("webformatURL")
                if not url:
                    continue
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    img = await client.get(url)
                if img.status_code != 200 or len(img.content) < 8000:
                    continue
                print(f"[PIXABAY] ok id={hit.get('id')}", flush=True)
                return {
                    "url": url,
                    "bytes": img.content,
                    "photographer": hit.get("user") or "pixabay",
                    "photo_id": hit.get("id"),
                    "alt": hit.get("tags") or q,
                    "query": q,
                }
        except Exception as e:
            print(f"[PIXABAY] error: {e}", flush=True)
        return None
