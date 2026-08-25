"""Pexels — ultimo fallback stock. Respeita OUTFIT."""
from __future__ import annotations

import random
from typing import Any

import httpx

from app.images.outfit import outfit_from_scene
from app.images.recent_guard import RECENT, make_jitter_seed


class PexelsSearchService:
    name = "pexels"

    def __init__(
        self,
        api_key: str | None,
        timeout: int = 30,
        per_page: int = 40,
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
        random.seed(make_jitter_seed())
        q = outfit_from_scene(query)
        page = random.randint(1, 4)
        print(f"[PEXELS] query: {q} page={page}", flush=True)
        params = {
            "query": q,
            "per_page": self.per_page,
            "orientation": self.orientation,
            "size": "large",
            "page": page,
        }
        headers = {"Authorization": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(self.base_url, params=params, headers=headers)
            if r.status_code != 200:
                print(f"[PEXELS] status={r.status_code}", flush=True)
                return None
            photos = (r.json().get("photos") or [])
            if not photos:
                return None
            fresh = [p for p in photos if not RECENT.seen(photo_id=p.get("id"))] or photos
            random.shuffle(fresh)
            for photo in fresh[:12]:
                src = photo.get("src") or {}
                url = src.get("large2x") or src.get("large") or src.get("original") or src.get("medium")
                if not url or RECENT.seen(url=url, photo_id=photo.get("id")):
                    continue
                print(f"[PEXELS] id={photo.get('id')}", flush=True)
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
