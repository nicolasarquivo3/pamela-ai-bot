"""
Busca fotos reais no Pexels (100% free) e devolve a melhor URL
para face swap.

Estilo da personagem: sensual / cropped / minissaia / salto / lingerie.
"""
from __future__ import annotations

import re
import random
from typing import Any

import httpx


class PexelsSearchService:
    name = "pexels"

    STYLE_PACKS = (
        "beautiful young woman tight mini dress crop top bodycon mirror selfie",
        "beautiful young woman leather jacket crop top sequin mini skirt night club selfie",
        "beautiful young woman white lace mini dress high heels mirror selfie",
        "beautiful young woman lace top denim mini skirt mirror selfie",
        "beautiful young woman black bodycon mini dress looking over shoulder",
        "beautiful young woman mint two piece backless crop ruched mini skirt selfie",
        "beautiful young woman cropped denim jacket sequin mini skirt midriff selfie",
        "beautiful young woman blue sequin mini dress high heels bedroom selfie",
        "beautiful young woman gym black crop top tight shorts fitness mirror selfie",
        "beautiful young woman black lace lingerie high heels sitting portrait",
        "beautiful young woman black lace bralette leather mini skirt stockings heels",
        "beautiful young woman white bandeau top white denim mini skirt mirror selfie",
        "beautiful young woman leather bustier denim shorts thigh high boots",
        "beautiful young woman black backless catsuit looking over shoulder",
        "beautiful young woman black plunging mini dress body chain platform heels",
        "beautiful young woman denim corset matching mini skirt fashion selfie",
    )

    def __init__(
        self,
        api_key: str | None,
        timeout: int = 30,
        per_page: int = 20,
        orientation: str = "portrait",
    ):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.per_page = min(int(per_page), 80)
        self.orientation = orientation
        self.base_url = "https://api.pexels.com/v1/search"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _clean_query(self, raw: str) -> str:
        style = random.choice(self.STYLE_PACKS)
        text = (raw or "").strip()

        extra: list[str] = []
        replacements = {
            r"\bsorrindo\b": "smiling",
            r"\bsorriso\b": "smile",
            r"\bna praia\b": "beach",
            r"\bpraia\b": "beach",
            r"\bdeitada\b": "lying on bed",
            r"\bdeitado\b": "lying on bed",
            r"\bcama\b": "bed",
            r"\bquarto\b": "bedroom",
            r"\bcasa\b": "home",
            r"\bsofá\b": "couch",
            r"\bsofa\b": "couch",
            r"\bsexy\b": "sensual",
            r"\blina\b": "lingerie",
            r"\blingerie\b": "lingerie",
            r"\bsalto\b": "high heels",
            r"\bsapatos?\b": "high heels",
            r"\bnoite\b": "night",
            r"\bselfie\b": "selfie",
            r"\bacademia\b": "gym",
            r"\bbalada\b": "night club",
        }
        for pat, rep in replacements.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                extra.append(rep)

        parts = [style]
        if extra:
            parts.append(" ".join(dict.fromkeys(extra)))

        q = " ".join(parts)
        if len(q) > 120:
            q = q[:120].rsplit(" ", 1)[0]
        return q.strip()

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._clean_query(query)
        print(f"[PEXELS] query estilo: {q}", flush=True)

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

            photos = (r.json().get("photos") or [])
            if not photos:
                print(f"[PEXELS] zero results, fallback pack", flush=True)
                fallback = random.choice(self.STYLE_PACKS)
                return await self._search_raw(fallback)

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

    async def _search_raw(self, q: str) -> dict[str, Any] | None:
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
            if r.status_code != 200:
                return None
            photos = r.json().get("photos") or []
            if not photos:
                return None
            photo = random.choice(photos)
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
        except Exception:
            return None
