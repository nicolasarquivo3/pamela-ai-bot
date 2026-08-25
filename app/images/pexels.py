"""
Busca fotos reais no Pexels (100% free) e devolve a melhor URL
para face swap.

Prioridade:
1) roupa / local / pose que o usuário PEDIU na mensagem
2) senão, um STYLE_PACK sensual da personagem
"""
from __future__ import annotations

import re
import random
from typing import Any

import httpx


class PexelsSearchService:
    name = "pexels"

    # Looks base da personagem (só quando o usuário NÃO pediu roupa específica)
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

    # PT/EN → termos curtos de busca no Pexels (roupa, local, pose)
    CLOTHING_MAP = (
        # vestidos
        (r"\bvestido\s+preto\b", "black mini dress"),
        (r"\bvestido\s+branco\b", "white mini dress"),
        (r"\bvestido\s+azul\b", "blue mini dress"),
        (r"\bvestido\s+vermelho\b", "red mini dress"),
        (r"\bvestido\s+verde\b", "green mini dress"),
        (r"\bvestido\s+rosa\b", "pink mini dress"),
        (r"\bvestido\s+de\s+renda\b", "lace mini dress"),
        (r"\bvestido\s+curto\b", "mini dress"),
        (r"\bvestido\s+longo\b", "long dress"),
        (r"\bvestido\s+justo\b", "bodycon mini dress"),
        (r"\bvestido\s+paet[eê]\b", "sequin mini dress"),
        (r"\bvestido\b", "mini dress"),
        # saias
        (r"\bsaia\s+jeans\b", "denim mini skirt"),
        (r"\bsaia\s+preta\b", "black mini skirt"),
        (r"\bsaia\s+branca\b", "white mini skirt"),
        (r"\bsaia\s+de\s+paet[eê]\b", "sequin mini skirt"),
        (r"\bminissaia\b", "mini skirt"),
        (r"\bsaia\s+curta\b", "mini skirt"),
        (r"\bsaia\b", "mini skirt"),
        # tops
        (r"\bcropped\b", "crop top"),
        (r"\bcrop\s*top\b", "crop top"),
        (r"\bblusa\s+de\s+renda\b", "lace top"),
        (r"\bblusa\b", "blouse"),
        (r"\btop\s+faixa\b", "bandeau top"),
        (r"\btop\b", "crop top"),
        (r"\bcorset\b", "corset"),
        (r"\bcors[eé]\b", "corset"),
        # conjuntos / looks
        (r"\blingerie\b", "lingerie"),
        (r"\blina\b", "lingerie"),
        (r"\bcalcinha\b", "lingerie"),
        (r"\bsuti[aã]n\b", "lingerie bra"),
        (r"\bbody\b", "bodysuit"),
        (r"\bmacac[aã]o\b", "catsuit jumpsuit"),
        (r"\bjumpsuit\b", "jumpsuit"),
        (r"\bjeans\b", "denim"),
        (r"\bcouro\b", "leather"),
        (r"\bjaqueta\s+de\s+couro\b", "leather jacket"),
        (r"\bjaqueta\s+jeans\b", "denim jacket"),
        (r"\bjaqueta\b", "jacket"),
        (r"\bshort\b", "shorts"),
        (r"\bshorts?\s+jeans\b", "denim shorts"),
        (r"\bmeia\s*[-\s]?cal[cç]a\b", "stockings"),
        (r"\bmeia\s*7\b", "thigh high stockings"),
        (r"\bfishnet\b", "fishnet stockings"),
        (r"\brede\b", "fishnet"),
        # sapatos
        (r"\bsalto\s+alto\b", "high heels"),
        (r"\bsalto\b", "high heels"),
        (r"\bscarpin\b", "high heels"),
        (r"\bbota\s+over\b", "thigh high boots"),
        (r"\bbota\s+alta\b", "thigh high boots"),
        (r"\bbota\b", "boots"),
        (r"\bsand[aá]lia\b", "sandals heels"),
        (r"\bt[eê]nis\b", "sneakers"),
        # locais
        (r"\bpraia\b", "beach"),
        (r"\bquarto\b", "bedroom"),
        (r"\bcama\b", "bed"),
        (r"\bcasa\b", "home"),
        (r"\bbalada\b", "night club"),
        (r"\bfesta\b", "party night"),
        (r"\bacademia\b", "gym fitness"),
        (r"\bbanheiro\b", "bathroom mirror"),
        (r"\bespelho\b", "mirror selfie"),
        (r"\bcozinha\b", "kitchen"),
        (r"\brua\b", "street outdoor"),
        (r"\bcarro\b", "car"),
        (r"\bpiscina\b", "pool"),
        # pose / enquadramento
        (r"\bselfie\b", "selfie"),
        (r"\bdeitada\b", "lying on bed"),
        (r"\bdeitado\b", "lying on bed"),
        (r"\bsorrindo\b", "smiling"),
        (r"\bde\s+costas\b", "looking over shoulder"),
        (r"\bcostas\s+nuas\b", "backless looking over shoulder"),
        (r"\bagachad[oa]\b", "squatting pose"),
        (r"\bsentinad[oa]\b", "sitting pose"),
        # cores avulsas
        (r"\bpreto\b", "black"),
        (r"\bpreta\b", "black"),
        (r"\bbranco\b", "white"),
        (r"\bbranca\b", "white"),
        (r"\bvermelho\b", "red"),
        (r"\bvermelha\b", "red"),
        (r"\bazul\b", "blue"),
        (r"\brosa\b", "pink"),
        (r"\bverde\b", "green"),
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

    def _extract_user_bits(self, raw: str) -> list[str]:
        """Extrai roupa/local/pose do texto do usuário (PT → EN)."""
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

        if bits:
            core = " ".join(bits[:6])
            q = f"beautiful young woman {core} portrait mirror selfie"
            print(f"[PEXELS] pedido do usuário: {bits}", flush=True)
        else:
            style = random.choice(self.STYLE_PACKS)
            q = style
            print(f"[PEXELS] sem roupa no pedido → pack: {style}", flush=True)

        if len(q) > 120:
            q = q[:120].rsplit(" ", 1)[0]
        return q.strip()

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
                print(f"[PEXELS] zero results for: {q}", flush=True)
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
