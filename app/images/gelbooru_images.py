"""Gelbooru NSFW (anime). Respeita bits de OUTFIT quando possivel."""
from __future__ import annotations

import random
from typing import Any

import httpx

from app.images.outfit import extract_outfit_bits


class GelbooruSearchService:
    name = "gelbooru"

    BASE_TAGS = (
        "1girl solo miniskirt high_heels rating:questionable",
        "1girl solo short_dress high_heels rating:questionable",
        "1girl solo microskirt crop_top rating:questionable",
    )

    def __init__(self, timeout: int = 40, limit: int = 30):
        self.timeout = int(timeout)
        self.limit = min(int(limit), 100)
        self.api = "https://gelbooru.com/index.php"

    async def available(self) -> bool:
        return True

    def _tags(self, query: str) -> str:
        bits = extract_outfit_bits(query)
        tags = ["1girl", "solo", "rating:questionable"]
        blob = " ".join(bits).lower()
        if "lingerie" in blob:
            tags.append("lingerie")
        elif "skirt" in blob:
            tags.append("miniskirt")
        elif "dress" in blob:
            tags.append("short_dress")
        else:
            return random.choice(self.BASE_TAGS)
        if "heels" in blob or "heel" in blob:
            tags.append("high_heels")
        return " ".join(tags)

    async def search(self, query: str) -> dict[str, Any] | None:
        tags = self._tags(query)
        print(f"[GELBOORU] tags: {tags}", flush=True)
        params = {
            "page": "dapi", "s": "post", "q": "index", "json": "1",
            "limit": str(self.limit), "tags": tags,
        }
        headers = {"User-Agent": "pamela-ai-bot/1.0"}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                r = await client.get(self.api, params=params)
                if r.status_code != 200:
                    return None
                data = r.json()
                posts = data.get("post") if isinstance(data, dict) else data
                if not posts:
                    return None
                if isinstance(posts, dict):
                    posts = [posts]
                random.shuffle(posts)
                for post in posts[:12]:
                    url = post.get("file_url") or post.get("sample_url")
                    if not url or not str(url).startswith("http"):
                        continue
                    img = await client.get(url)
                    if img.status_code != 200 or len(img.content) < 5000:
                        continue
                    return {
                        "url": url,
                        "bytes": img.content,
                        "photographer": "gelbooru",
                        "photo_id": post.get("id"),
                        "alt": post.get("tags") or tags,
                        "query": tags,
                    }
        except Exception as e:
            print(f"[GELBOORU] error: {e}", flush=True)
        return None
