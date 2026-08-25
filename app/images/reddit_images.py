"""Reddit JSON — fashion. Pode 403 no Render."""
from __future__ import annotations

import random
from typing import Any

import httpx

from app.images.outfit import extract_outfit_bits


class RedditImageSearchService:
    name = "reddit"

    SUBREDDITS = (
        "Fashion_Sexiness", "NSFW_Fashion", "Sexy", "clubwear",
        "highheels", "selfies", "GoneMild",
    )

    def __init__(self, timeout: int = 40, limit: int = 30):
        self.timeout = int(timeout)
        self.limit = min(int(limit), 50)

    async def available(self) -> bool:
        return True

    def _pick_subs(self, query: str) -> list[str]:
        bits = " ".join(extract_outfit_bits(query)).lower()
        if "lingerie" in bits:
            return ["lingerie", "GoneMild", "Sexy"]
        if "skirt" in bits or "dress" in bits:
            return ["Fashion_Sexiness", "NSFW_Fashion", "Sexy"]
        subs = list(self.SUBREDDITS)
        random.shuffle(subs)
        return subs[:3]

    def _extract_image_url(self, post: dict) -> str | None:
        data = post.get("data") or {}
        url = (data.get("url_overridden_by_dest") or data.get("url") or "").strip()
        if not url:
            return None
        lower = url.lower()
        if any(lower.endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp")):
            return url
        if "i.redd.it" in lower or "i.imgur.com" in lower:
            return url
        preview = data.get("preview") or {}
        images = preview.get("images") or []
        if images:
            source = (images[0].get("source") or {}).get("url") or ""
            if source:
                return source.replace("&amp;", "&")
        return None

    async def search(self, query: str) -> dict[str, Any] | None:
        subs = self._pick_subs(query)
        print(f"[REDDIT] subs={subs}", flush=True)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        candidates = []
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                for sub in subs:
                    for base in ("https://www.reddit.com", "https://old.reddit.com"):
                        try:
                            r = await client.get(
                                f"{base}/r/{sub}/hot.json",
                                params={"limit": self.limit, "raw_json": 1},
                            )
                            if r.status_code != 200:
                                continue
                            for post in ((r.json().get("data") or {}).get("children") or []):
                                img = self._extract_image_url(post)
                                if img:
                                    title = (post.get("data") or {}).get("title") or ""
                                    candidates.append({"url": img, "title": title, "sub": sub})
                            break
                        except Exception:
                            continue
                if not candidates:
                    print("[REDDIT] zero candidates", flush=True)
                    return None
                random.shuffle(candidates)
                for item in candidates[:12]:
                    try:
                        r = await client.get(item["url"])
                        if r.status_code != 200 or len(r.content) < 8000:
                            continue
                        return {
                            "url": item["url"],
                            "bytes": r.content,
                            "photographer": f"reddit/r/{item['sub']}",
                            "photo_id": None,
                            "alt": item["title"],
                            "query": f"reddit:{item['sub']}",
                        }
                    except Exception:
                        continue
        except Exception as e:
            print(f"[REDDIT] error: {e}", flush=True)
        return None
