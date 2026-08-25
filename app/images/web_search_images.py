"""
Busca imagens grátis via DuckDuckGo (sem API key).
Pacote novo: ddgs (antigo duckduckgo_search renomeado).
"""
from __future__ import annotations

import asyncio
import random
import re
from typing import Any

import httpx

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore


class WebImageSearchService:
    name = "duckduckgo"

    STYLE_PACKS = (
        "sexy woman tight bodycon mini dress cleavage mirror selfie",
        "sexy woman black lace lingerie high heels bedroom selfie",
        "sexy woman crop top micro mini skirt night club mirror",
        "sexy woman leather jacket sequin mini skirt thigh boots",
        "sexy woman white lace mini dress high heels curvy selfie",
        "sexy woman backless black mini dress looking over shoulder",
        "sexy woman denim micro skirt crop top midriff mirror",
        "sexy woman sequin mini dress high slit heels party",
        "sexy woman lingerie stockings high heels sitting",
        "sexy woman plunging mini dress body chain platform heels",
        "sexy woman corset mini skirt fashion Instagram mirror",
        "sexy woman fishnet stockings black mini dress heels",
        "sexy woman satin lingerie bedroom mirror selfie",
        "sexy woman tight dress club night selfie fashion model",
    )

    CLOTHING_MAP = (
        (r"\bvestido\s+preto\b", "black tight mini dress"),
        (r"\bvestido\s+branco\b", "white tight mini dress"),
        (r"\bvestido\s+azul\b", "blue mini dress"),
        (r"\bvestido\s+vermelho\b", "red mini dress"),
        (r"\bvestido\s+de\s+renda\b", "lace mini dress"),
        (r"\bvestido\s+justo\b", "bodycon mini dress"),
        (r"\bvestido\s+paet[eê]\b", "sequin mini dress"),
        (r"\bvestido\b", "tight bodycon mini dress sexy"),
        (r"\bsaia\s+jeans\b", "denim micro mini skirt"),
        (r"\bminissaia\b", "micro mini skirt"),
        (r"\bsaia\b", "mini skirt sexy"),
        (r"\bcropped\b", "crop top midriff"),
        (r"\blingerie\b", "sexy lingerie lace heels"),
        (r"\blina\b", "sexy lingerie"),
        (r"\bjaqueta\s+de\s+couro\b", "leather jacket crop top"),
        (r"\bcouro\b", "leather sexy"),
        (r"\bsalto\b", "high heels"),
        (r"\bbota\b", "thigh high boots"),
        (r"\bbalada\b", "night club party"),
        (r"\bacademia\b", "gym crop top tight shorts"),
        (r"\bquarto\b", "bedroom mirror"),
        (r"\bpraia\b", "beach bikini sexy"),
        (r"\bsexy\b", "sexy"),
        (r"\bsensual\b", "sensual sexy"),
        (r"\bgostosa\b", "sexy curvy"),
        (r"\bdecote\b", "deep cleavage"),
        (r"\bpreto\b", "black"),
        (r"\bbranco\b", "white"),
        (r"\bvermelh[oa]\b", "red"),
    )

    def __init__(self, timeout: int = 45, max_results: int = 15):
        self.timeout = int(timeout)
        self.max_results = int(max_results)

    async def available(self) -> bool:
        return DDGS is not None

    def _build_query(self, raw: str) -> str:
        text = (raw or "").strip()
        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.I | re.S)
        if m:
            text = m.group(1)

        bits: list[str] = []
        for pat, eng in self.CLOTHING_MAP:
            if re.search(pat, text, re.I) and eng not in bits:
                bits.append(eng)

        if bits:
            q = f"sexy woman {' '.join(bits[:5])} mirror selfie fashion"
        else:
            q = random.choice(self.STYLE_PACKS)

        print(f"[DDG] query: {q}", flush=True)
        return q

    def _search_sync(self, q: str) -> list[dict[str, Any]]:
        if DDGS is None:
            print("[DDG] pacote ddgs nao instalado (pip install ddgs)", flush=True)
            return []

        rows: list[dict[str, Any]] = []
        try:
            ddgs = DDGS()
            results = ddgs.images(
                q,
                max_results=self.max_results,
                safesearch="off",
            )
            for item in results or []:
                url = item.get("image") or item.get("url")
                if not url or not str(url).startswith("http"):
                    continue
                rows.append(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "source": item.get("source") or "duckduckgo",
                    }
                )
        except Exception as e:
            print(f"[DDG] search error: {e}", flush=True)

        return rows

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._build_query(query)
        rows = await asyncio.to_thread(self._search_sync, q)
        if not rows:
            print("[DDG] zero results", flush=True)
            return None

        random.shuffle(rows)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                for item in rows[:10]:
                    url = item["url"]
                    try:
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        ctype = (r.headers.get("content-type") or "").lower()
                        is_img = "image" in ctype or url.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".webp")
                        )
                        if not is_img:
                            continue
                        data = r.content
                        if len(data) < 8000:
                            continue
                        print(
                            f"[DDG] ok bytes={len(data)} src={item.get('source')} "
                            f"title={(item.get('title') or '')[:50]!r}",
                            flush=True,
                        )
                        return {
                            "url": url,
                            "bytes": data,
                            "photographer": item.get("source") or "web",
                            "photo_id": None,
                            "alt": item.get("title") or q,
                            "query": q,
                        }
                    except Exception as e:
                        print(f"[DDG] download fail: {e}", flush=True)
                        continue
        except Exception as e:
            print(f"[DDG] client error: {e}", flush=True)

        print("[DDG] nenhuma imagem baixavel", flush=True)
        return None
