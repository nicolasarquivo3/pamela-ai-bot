"""Bing via ddgs — respeita OUTFIT da scene."""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from app.images.outfit import outfit_from_scene
from app.images.recent_guard import RECENT, make_jitter_seed

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore


class BingImageSearchService:
    name = "bing"

    def __init__(self, timeout: int = 45, max_results: int = 20):
        self.timeout = int(timeout)
        self.max_results = int(max_results)

    async def available(self) -> bool:
        return DDGS is not None

    def _search_sync(self, q: str) -> list[dict[str, Any]]:
        if DDGS is None:
            return []
        rows = []
        try:
            ddgs = DDGS()
            try:
                results = ddgs.images(
                    q, max_results=self.max_results, safesearch="off", backend="bing"
                )
            except TypeError:
                results = ddgs.images(q, max_results=self.max_results, safesearch="off")
            for item in results or []:
                url = item.get("image") or item.get("url")
                if not url or not str(url).startswith("http"):
                    continue
                if RECENT.seen(url=url):
                    continue
                rows.append({
                    "url": url,
                    "title": item.get("title") or "",
                    "source": item.get("source") or "bing",
                })
        except Exception as e:
            print(f"[BING] search error: {e}", flush=True)
        return rows

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None
        random.seed(make_jitter_seed())
        q = outfit_from_scene(query)
        print(f"[BING] query: {q}", flush=True)
        rows = await asyncio.to_thread(self._search_sync, q)
        if not rows:
            return None
        random.shuffle(rows)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                for item in rows[:12]:
                    url = item["url"]
                    if RECENT.seen(url=url):
                        continue
                    try:
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        ctype = (r.headers.get("content-type") or "").lower()
                        if "image" not in ctype and not url.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".webp")
                        ):
                            continue
                        data = r.content
                        if len(data) < 8000:
                            continue
                        print(f"[BING] ok bytes={len(data)}", flush=True)
                        return {
                            "url": url,
                            "bytes": data,
                            "photographer": item.get("source") or "bing",
                            "photo_id": None,
                            "alt": item.get("title") or q,
                            "query": q,
                        }
                    except Exception as e:
                        print(f"[BING] download fail: {e}", flush=True)
        except Exception as e:
            print(f"[BING] client error: {e}", flush=True)
        return None
