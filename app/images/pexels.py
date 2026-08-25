"""
Busca fotos reais no Pexels (100% free) e devolve a melhor URL
para face swap.
"""
from __future__ import annotations

import re
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

    def _clean_query(self, raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return "beautiful young woman portrait natural light"

        for bad in [
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
            "adult female character",
        ]:
            text = re.sub(re.escape(bad), " ", text, flags=re.IGNORECASE)

        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.IGNORECASE)
        if m:
            text = m.group(1)

        text = re.sub(r"[^.]*pedido do usuário[^.]*\.?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" .,-")

        replacements = {
            r"\bfoto\b": "photo",
            r"\bselfie\b": "selfie",
            r"\bsorrindo\b": "smiling",
            r"\bsorriso\b": "smile",
            r"\bna praia\b": "on the beach",
            r"\bpraia\b": "beach",
            r"\bdeitada\b": "lying down",
            r"\bdeitado\b": "lying down",
            r"\bcama\b": "bed",
            r"\bquarto\b": "bedroom",
            r"\bcasa\b": "home",
            r"\bsofá\b": "couch",
            r"\bsofa\b": "couch",
            r"\bsexy\b": "sexy",
            r"\blina\b": "lingerie",
            r"\bme manda uma foto\b": "portrait selfie",
            r"\bmanda uma foto\b": "portrait selfie",
            r"\bquero ver você\b": "portrait looking at camera",
            r"\btira uma selfie\b": "selfie",
        }
        for pat, rep in replacements.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text) if len(w) > 2]
        stop = {
            "criar", "fotografia", "espontanea", "personagem", "pamela", "pamera",
            "mulher", "adulta", "como", "estivesse", "tirando", "naquele", "momento",
            "especificamente", "para", "enviar", "usuario", "deve", "parecer",
            "natural", "coerente", "com", "conversa", "mantendo", "identidade",
            "visual", "estabelecida", "interpretar", "pedido", "contexto",
            "preservar", "detalhes", "relevantes", "roupa", "pose", "expressao",
            "local", "enquadramento", "ambiente", "mencionados", "pelo",
        }
        words = [w for w in words if w.lower() not in stop]

        if not words:
            return "beautiful young woman portrait natural light smiling"

        core = " ".join(words[:8])
        q = f"beautiful young woman {core} portrait"
        if len(q) > 100:
            q = q[:100].rsplit(" ", 1)[0]
        return q.strip()

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._clean_query(query)
        print(f"[PEXELS] query limpa: {q}", flush=True)

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
                return await self._search_raw("beautiful young woman portrait natural light")

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
            photos = (r.json().get("photos") or [])
            if not photos:
                return None
            photo = random.choice(photos)
            src = photo.get("src") or {}
            url = src.get("large2x") or src.get("large") or src.get("original") or src.get("medium")
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
