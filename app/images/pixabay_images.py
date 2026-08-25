"""
Pixabay — stock grátis (PIXABAY_API_KEY em pixabay.com/api/docs).
Padrao: micro saia / micro vestido.
"""
from __future__ import annotations

import random
import re
from typing import Any

import httpx


class PixabaySearchService:
    name = "pixabay"

    STYLE_PACKS = (
        "woman micro mini skirt fashion",
        "woman mini dress high heels",
        "woman bodycon dress portrait",
        "woman short dress party",
        "woman denim mini skirt",
        "woman tight dress fashion model",
    )

    def __init__(self, api_key: str | None, timeout: int = 30, per_page: int = 20):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.per_page = min(int(per_page), 200)
        self.base_url = "https://pixabay.com/api/"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _query(self, raw: str) -> str:
        text = (raw or "").lower()
        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.I | re.S)
        if m:
            text = m.group(1)

        bits = []
        mapping = (
            (r"\blingerie\b|\blina\b", "lingerie woman"),
            (r"\bvestido\b", "woman mini dress"),
            (r"\bsaia\b|\bminissaia\b", "woman mini skirt"),
            (r"\bcropped\b", "woman crop top mini skirt"),
            (r"\bsalto\b", "woman high heels mini dress"),
            (r"\bbalada\b|\bfesta\b", "woman party mini dress"),
            (r"\bpraia\b", "woman bikini beach"),
            (r"\bacademia\b", "woman fitness gym"),
        )
        for pat, eng in mapping:
            if re.search(pat, text, re.I):
                bits.append(eng)

        if not bits:
            q = random.choice(self.STYLE_PACKS)
        elif re.search(r"\blingerie\b|\blina\b", text, re.I):
            q = " ".join(bits[:3])
        else:
            q = " ".join(bits[:2]) + " mini skirt mini dress fashion"
            if "lingerie" in q:
                q = random.choice(self.STYLE_PACKS)

        print(f"[PIXABAY] query: {q}", flush=True)
        return q

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._query(query)
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
                print(f"[PIXABAY] status={r.status_code}", flush=True)
                return None
            hits = (r.json() or {}).get("hits") or []
            if not hits:
                print("[PIXABAY] zero hits", flush=True)
                return None

            random.shuffle(hits)
            for hit in hits[:8]:
                url = (
                    hit.get("largeImageURL")
                    or hit.get("fullHDURL")
                    or hit.get("webformatURL")
                )
                if not url:
                    continue
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        img = await client.get(url)
                    if img.status_code != 200 or len(img.content) < 8000:
                        continue
                    print(
                        f"[PIXABAY] ok id={hit.get('id')} bytes={len(img.content)}",
                        flush=True,
                    )
                    return {
                        "url": url,
                        "bytes": img.content,
                        "photographer": hit.get("user") or "pixabay",
                        "photo_id": hit.get("id"),
                        "alt": hit.get("tags") or q,
                        "query": q,
                    }
                except Exception as e:
                    print(f"[PIXABAY] download fail: {e}", flush=True)
        except Exception as e:
            print(f"[PIXABAY] error: {e}", flush=True)
        return None
