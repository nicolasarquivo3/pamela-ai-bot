"""
Busca fotos reais no Pexels (100% free) e devolve a melhor URL
para face swap.

Viés fixo: sensual / tight / bodycon / mirror selfie / heels.
Prioridade:
1) roupa / local / pose que o usuário PEDIU
2) senão STYLE_PACK sensual da personagem
"""
from __future__ import annotations

import re
import random
from typing import Any

import httpx


class PexelsSearchService:
    name = "pexels"

    STYLE_PACKS = (
        "sexy young woman tight bodycon mini dress deep cleavage mirror selfie",
        "sexy young woman black lace lingerie high heels bedroom mirror selfie",
        "sexy young woman crop top micro mini skirt midriff night club selfie",
        "sexy young woman leather jacket crop top sequin mini skirt thigh boots",
        "sexy young woman white lace mini dress high heels curvy mirror selfie",
        "sexy young woman backless black mini dress looking over shoulder",
        "sexy young woman denim mini skirt white crop top midriff mirror selfie",
        "sexy young woman blue sequin mini dress high slit heels bedroom",
        "sexy young woman black lace bralette leather mini skirt stockings heels",
        "sexy young woman bandeau top micro denim shorts thigh high boots",
        "sexy young woman black plunging mini dress body chain platform heels",
        "sexy young woman denim corset tight mini skirt curvy fashion selfie",
        "sexy young woman mint ruched mini two piece backless crop selfie",
        "sexy young woman fishnet stockings black mini dress high heels",
        "sexy young woman gym crop top tight shorts fitness mirror curvy",
        "sexy young woman satin lingerie robe high heels bedroom selfie",
    )

    CLOTHING_MAP = (
        (r"\bvestido\s+preto\b", "black tight mini dress"),
        (r"\bvestido\s+branco\b", "white tight mini dress"),
        (r"\bvestido\s+azul\b", "blue tight mini dress"),
        (r"\bvestido\s+vermelho\b", "red tight mini dress"),
        (r"\bvestido\s+verde\b", "green tight mini dress"),
        (r"\bvestido\s+rosa\b", "pink tight mini dress"),
        (r"\bvestido\s+de\s+renda\b", "white lace tight mini dress"),
        (r"\bvestido\s+curto\b", "tight mini dress"),
        (r"\bvestido\s+longo\b", "long tight dress"),
        (r"\bvestido\s+justo\b", "bodycon mini dress"),
        (r"\bvestido\s+paet[eê]\b", "sequin mini dress"),
        (r"\bvestido\b", "tight bodycon mini dress"),
        (r"\bsaia\s+jeans\b", "denim micro mini skirt"),
        (r"\bsaia\s+preta\b", "black micro mini skirt"),
        (r"\bsaia\s+branca\b", "white micro mini skirt"),
        (r"\bsaia\s+de\s+paet[eê]\b", "sequin micro mini skirt"),
        (r"\bminissaia\b", "micro mini skirt"),
        (r"\bsaia\s+curta\b", "micro mini skirt"),
        (r"\bsaia\b", "mini skirt"),
        (r"\bcropped\b", "crop top midriff"),
        (r"\bcrop\s*top\b", "crop top midriff"),
        (r"\bblusa\s+de\s+renda\b", "lace crop top"),
        (r"\bblusa\b", "tight blouse"),
        (r"\btop\s+faixa\b", "bandeau crop top"),
        (r"\btop\b", "crop top midriff"),
        (r"\bcorset\b", "corset bustier"),
        (r"\bcors[eé]\b", "corset bustier"),
        (r"\blingerie\b", "sexy lingerie lace"),
        (r"\blina\b", "sexy lingerie lace"),
        (r"\bcalcinha\b", "sexy lingerie"),
        (r"\bsuti[aã]n\b", "lace bra lingerie"),
        (r"\bbody\b", "sexy bodysuit"),
        (r"\bmacac[aã]o\b", "tight catsuit jumpsuit"),
        (r"\bjumpsuit\b", "tight jumpsuit"),
        (r"\bjeans\b", "denim"),
        (r"\bcouro\b", "leather"),
        (r"\bjaqueta\s+de\s+couro\b", "leather jacket"),
        (r"\bjaqueta\s+jeans\b", "cropped denim jacket"),
        (r"\bjaqueta\b", "jacket"),
        (r"\bshort\b", "tight shorts"),
        (r"\bshorts?\s+jeans\b", "denim micro shorts"),
        (r"\bmeia\s*[-\s]?cal[cç]a\b", "stockings"),
        (r"\bmeia\s*7\b", "thigh high stockings"),
        (r"\bfishnet\b", "fishnet stockings"),
        (r"\brede\b", "fishnet stockings"),
        (r"\bsalto\s+alto\b", "high heels"),
        (r"\bsalto\b", "high heels"),
        (r"\bscarpin\b", "high heels"),
        (r"\bbota\s+over\b", "thigh high boots"),
        (r"\bbota\s+alta\b", "thigh high boots"),
        (r"\bbota\b", "thigh high boots"),
        (r"\bsand[aá]lia\b", "strappy high heels"),
        (r"\bt[eê]nis\b", "sneakers"),
        (r"\bpraia\b", "beach bikini"),
        (r"\bquarto\b", "bedroom"),
        (r"\bcama\b", "bed bedroom"),
        (r"\bcasa\b", "home mirror"),
        (r"\bbalada\b", "night club party"),
        (r"\bfesta\b", "night party club"),
        (r"\bacademia\b", "gym fitness tight"),
        (r"\bbanheiro\b", "bathroom mirror selfie"),
        (r"\bespelho\b", "mirror selfie"),
        (r"\bcozinha\b", "kitchen mirror"),
        (r"\brua\b", "street fashion"),
        (r"\bcarro\b", "car selfie"),
        (r"\bpiscina\b", "pool swimsuit"),
        (r"\bselfie\b", "mirror selfie"),
        (r"\bdeitada\b", "lying on bed sensual"),
        (r"\bdeitado\b", "lying on bed sensual"),
        (r"\bsorrindo\b", "smiling"),
        (r"\bde\s+costas\b", "looking over shoulder"),
        (r"\bcostas\s+nuas\b", "backless looking over shoulder"),
        (r"\bagachad[oa]\b", "squatting pose sexy"),
        (r"\bsentinad[oa]\b", "sitting pose sexy"),
        (r"\bsexy\b", "sexy"),
        (r"\bsensual\b", "sensual"),
        (r"\bgostosa\b", "sexy curvy"),
        (r"\bprovocante\b", "sexy provocative"),
        (r"\bdecote\b", "deep cleavage"),
        (r"\bcurvy\b", "curvy"),
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

        if bits:
            core = " ".join(bits[:5])
            q = f"sexy young woman {core} tight sensual mirror selfie portrait"
            print(f"[PEXELS] pedido: {bits}", flush=True)
        else:
            style = random.choice(self.STYLE_PACKS)
            q = style
            print(f"[PEXELS] pack sensual: {style}", flush=True)

        if len(q) > 110:
            q = q[:110].rsplit(" ", 1)[0]
        return q.strip()

    def _score_photo(self, photo: dict) -> int:
        alt = (photo.get("alt") or "").lower()
        score = 0
        for w in (
            "dress", "skirt", "lingerie", "heel", "sexy", "fashion",
            "model", "crop", "bikini", "body", "mirror", "portrait",
            "woman", "girl", "club", "night", "leather", "lace",
        ):
            if w in alt:
                score += 2
        for w in (
            "man", "men", "boy", "child", "baby", "dog", "cat",
            "food", "landscape", "building", "car interior",
        ):
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
