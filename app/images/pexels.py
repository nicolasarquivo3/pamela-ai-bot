"""
Pexels — ULTIMO fallback stock.
Padrao: micro saia / micro vestido.
"""
from __future__ import annotations

import re
import random
from typing import Any

import httpx


class PexelsSearchService:
    name = "pexels"

    STYLE_PACKS = (
        "sexy young woman micro mini skirt crop top mirror selfie",
        "sexy young woman tight bodycon micro mini dress high heels",
        "sexy young woman black micro mini dress mirror selfie",
        "sexy young woman white micro mini dress high heels",
        "sexy young woman denim micro mini skirt crop top selfie",
        "sexy young woman sequin micro mini dress high heels party",
        "sexy young woman leather jacket micro mini skirt boots",
        "sexy young woman backless micro mini dress over shoulder",
        "sexy young woman red bodycon micro mini dress heels",
        "sexy young woman plaid micro mini skirt crop top",
        "sexy young woman lace micro mini dress high heels selfie",
        "sexy young woman club micro mini skirt tight top",
        "sexy young woman micro skirt stockings high heels",
        "sexy young woman short bodycon dress heels fashion",
    )

    CLOTHING_MAP = (
        (r"\bvestido\s+preto\b", "black micro mini dress"),
        (r"\bvestido\s+branco\b", "white micro mini dress"),
        (r"\bvestido\s+azul\b", "blue micro mini dress"),
        (r"\bvestido\s+vermelho\b", "red micro mini dress"),
        (r"\bvestido\s+de\s+renda\b", "lace micro mini dress"),
        (r"\bvestido\s+justo\b", "bodycon micro mini dress"),
        (r"\bvestido\s+paet[eê]\b", "sequin micro mini dress"),
        (r"\bvestido\b", "micro mini dress"),
        (r"\bsaia\s+jeans\b", "denim micro mini skirt"),
        (r"\bsaia\s+preta\b", "black micro mini skirt"),
        (r"\bminissaia\b", "micro mini skirt"),
        (r"\bsaia\s+curta\b", "micro mini skirt"),
        (r"\bsaia\b", "micro mini skirt"),
        (r"\bcropped\b", "crop top"),
        (r"\blingerie\b", "lingerie"),
        (r"\blina\b", "lingerie"),
        (r"\bsalto\b", "high heels"),
        (r"\bbota\b", "boots"),
        (r"\bbalada\b", "night club"),
        (r"\bfesta\b", "party"),
        (r"\bacademia\b", "gym fitness"),
        (r"\bpraia\b", "beach"),
        (r"\bdecote\b", "cleavage"),
        (r"\bpreto\b", "black"),
        (r"\bbranco\b", "white"),
        (r"\bvermelh[oa]\b", "red"),
    )

    def __init__(
        self,
        api_key: str | None,
        timeout: int = 30,
        per_page: int = 30,
        orientation: str = "portrait",
    ):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.per_page = min(int(per_page), 80)
        self.orientation = orientation
        self.base_url = "https://api.pexels.com/v1/search"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _extract_user_bits(self, raw: str) -> list[str]:
        text = (raw or "").strip()
        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1)
        for bad in (
            "Criar uma fotografia",
            "fotografia espontânea",
            "personagem Pâmela",
            "personagem Pamela",
            "mulher adulta",
            "identidade visual",
            "Interpretar o pedido",
            "Preservar os detalhes",
            "PHOTOREALISTIC",
            "Create a photorealistic",
        ):
            text = re.sub(re.escape(bad), " ", text, flags=re.IGNORECASE)

        text = re.sub(r"\s+", " ", text).strip()
        found: list[str] = []
        for pat, eng in self.CLOTHING_MAP:
            if re.search(pat, text, flags=re.IGNORECASE):
                if eng not in found:
                    found.append(eng)
        return found

    def _clean_query(self, raw: str) -> str:
        bits = self._extract_user_bits(raw)
        text = raw or ""
        wants_lingerie = bool(
            re.search(r"\blingerie\b|\blina\b|\bcalcinha\b", text, re.I)
        )
        if not wants_lingerie:
            bits = [b for b in bits if "lingerie" not in b.lower()]

        if bits:
            core = " ".join(bits[:5])
            if wants_lingerie:
                q = f"sexy young woman {core} portrait"
            else:
                q = (
                    f"sexy young woman {core} micro mini dress "
                    f"micro mini skirt fashion portrait"
                )
            print(f"[PEXELS] pedido: {bits}", flush=True)
        else:
            q = random.choice(self.STYLE_PACKS)
            print(f"[PEXELS] pack micro: {q}", flush=True)

        if len(q) > 110:
            q = q[:110].rsplit(" ", 1)[0]
        return q.strip()

    def _score_photo(self, photo: dict) -> int:
        alt = (photo.get("alt") or "").lower()
        score = 0
        for w in (
            "dress", "skirt", "mini", "heel", "fashion",
            "model", "woman", "portrait", "club",
        ):
            if w in alt:
                score += 2
        for w in ("lingerie", "bra", "underwear", "man", "boy", "child", "food"):
            if w in alt:
                score -= 5
        return score

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._clean_query(query)
        print(f"[PEXELS] query final: {q}", flush=True)

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
                print("[PEXELS] zero results, fallback pack", flush=True)
                return await self._search_raw(random.choice(self.STYLE_PACKS))

            photos.sort(key=self._score_photo, reverse=True)
            top = photos[: max(5, min(8, len(photos)))]
            photo = random.choice(top)

            src = photo.get("src") or {}
            url = (
                src.get("large2x")
                or src.get("large")
                or src.get("original")
                or src.get("medium")
            )
            if not url:
                return None

            print(
                f"[PEXELS] escolhida id={photo.get('id')} "
                f"score={self._score_photo(photo)} alt={(photo.get('alt') or '')[:60]!r}",
                flush=True,
            )

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
            photos.sort(key=self._score_photo, reverse=True)
            photo = random.choice(photos[:5] if len(photos) >= 5 else photos)
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
