"""
Bing Images via ddgs (backend bing) — grátis, sem API key.
Padrao: micro saia / micro vestido (lingerie so se pedir).
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


class BingImageSearchService:
    name = "bing"

    STYLE_PACKS = (
        "sexy woman micro mini skirt crop top midriff mirror selfie",
        "sexy woman tight bodycon micro mini dress high heels selfie",
        "sexy woman black micro mini dress cleavage mirror selfie",
        "sexy woman white micro mini dress high heels curvy selfie",
        "sexy woman denim micro mini skirt crop top midriff selfie",
        "sexy woman sequin micro mini dress high slit heels party",
        "sexy woman leather jacket micro mini skirt thigh boots",
        "sexy woman backless micro mini dress looking over shoulder",
        "sexy woman red bodycon micro mini dress high heels selfie",
        "sexy woman plaid micro mini skirt crop top heels selfie",
        "sexy woman lace micro mini dress high heels mirror selfie",
        "sexy woman club micro mini skirt tight top night selfie",
        "sexy woman micro skirt stockings high heels fashion selfie",
        "sexy woman short bodycon dress micro length heels selfie",
    )

    CLOTHING_MAP = (
        (r"\bvestido\s+preto\b", "black micro mini dress"),
        (r"\bvestido\s+branco\b", "white micro mini dress"),
        (r"\bvestido\s+azul\b", "blue micro mini dress"),
        (r"\bvestido\s+vermelho\b", "red micro mini dress"),
        (r"\bvestido\s+de\s+renda\b", "lace micro mini dress"),
        (r"\bvestido\s+justo\b", "bodycon micro mini dress"),
        (r"\bvestido\s+paet[eê]\b", "sequin micro mini dress"),
        (r"\bvestido\b", "micro mini dress bodycon"),
        (r"\bsaia\s+jeans\b", "denim micro mini skirt"),
        (r"\bsaia\s+preta\b", "black micro mini skirt"),
        (r"\bminissaia\b", "micro mini skirt"),
        (r"\bsaia\s+curta\b", "micro mini skirt"),
        (r"\bsaia\b", "micro mini skirt"),
        (r"\bcropped\b", "crop top midriff"),
        (r"\bcrop\s*top\b", "crop top midriff"),
        (r"\blingerie\b", "lingerie"),
        (r"\blina\b", "lingerie"),
        (r"\bcalcinha\b", "lingerie"),
        (r"\bjaqueta\s+de\s+couro\b", "leather jacket crop top"),
        (r"\bcouro\b", "leather"),
        (r"\bsalto\b", "high heels"),
        (r"\bbota\b", "thigh high boots"),
        (r"\bbalada\b", "night club"),
        (r"\bfesta\b", "party club"),
        (r"\bacademia\b", "gym crop top tight shorts"),
        (r"\bquarto\b", "bedroom mirror"),
        (r"\bpraia\b", "beach bikini"),
        (r"\bdecote\b", "deep cleavage"),
        (r"\bpreto\b", "black"),
        (r"\bpreta\b", "black"),
        (r"\bbranco\b", "white"),
        (r"\bbranca\b", "white"),
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

        user_wants_lingerie = bool(
            re.search(r"\blingerie\b|\blina\b|\bcalcinha\b|\bsuti", text, re.I)
        )
        if not user_wants_lingerie:
            bits = [
                b
                for b in bits
                if "lingerie" not in b.lower() and "bra" not in b.lower()
            ]

        if bits:
            core = " ".join(bits[:5])
            if user_wants_lingerie:
                q = f"sexy woman {core} fashion selfie"
            else:
                q = (
                    f"sexy woman {core} micro mini skirt "
                    f"micro mini dress short tight dress fashion selfie"
                )
        else:
            q = random.choice(self.STYLE_PACKS)

        print(f"[BING] query: {q}", flush=True)
        return q

    def _score(self, title: str, q: str) -> int:
        score = 0
        for w in (
            "mini", "skirt", "dress", "bodycon", "heels",
            "fashion", "club", "short", "tight",
        ):
            if w in title:
                score += 3
        for w in ("lingerie", "bra", "panties", "underwear", "nude", "naked"):
            if w in title and "lingerie" not in q.lower():
                score -= 8
        return score

    def _search_sync(self, q: str) -> list[dict[str, Any]]:
        if DDGS is None:
            return []
        rows: list[dict[str, Any]] = []
        try:
            ddgs = DDGS()
            try:
                results = ddgs.images(
                    q, max_results=self.max_results, safesearch="off", backend="bing"
                )
            except TypeError:
                results = ddgs.images(
                    q, max_results=self.max_results, safesearch="off"
                )
            for item in results or []:
                url = item.get("image") or item.get("url")
                if not url or not str(url).startswith("http"):
                    continue
                title = (item.get("title") or "").lower()
                rows.append(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "source": item.get("source") or "bing",
                        "score": self._score(title, q),
                    }
                )
        except Exception as e:
            print(f"[BING] search error: {e}", flush=True)
        rows.sort(key=lambda x: x.get("score", 0), reverse=True)
        return rows

    async def search(self, query: str) -> dict[str, Any] | None:
        if not await self.available():
            return None

        q = self._build_query(query)
        rows = await asyncio.to_thread(self._search_sync, q)
        if not rows:
            print("[BING] zero results", flush=True)
            return None

        top = rows[:8] if len(rows) >= 8 else rows
        random.shuffle(top)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                for item in top:
                    url = item["url"]
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
                        print(
                            f"[BING] ok bytes={len(data)} score={item.get('score')} "
                            f"title={(item.get('title') or '')[:50]!r}",
                            flush=True,
                        )
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
                        continue
        except Exception as e:
            print(f"[BING] client error: {e}", flush=True)

        print("[BING] nenhuma imagem baixavel", flush=True)
        return None
