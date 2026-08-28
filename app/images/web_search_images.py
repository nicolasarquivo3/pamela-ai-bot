from __future__ import annotations

def build_web_query(scene: str, extra: str = "") -> str:
    """Busca web a partir da roupa/cena da conversa."""
    import re as _re
    s = (scene or "")
    low = s.lower()
    bits = []
    m = _re.search(r"outfit:\s*([^|]+)", low, _re.I)
    if m:
        bits.append(m.group(1).strip())
    rules = [
        (r"vestido\s+preto\s*(curto|curtinho|mini)?|black\s+mini\s+dress", "black mini dress"),
        (r"vestido\s*(curto|curtinho|mini)|mini\s+dress", "mini dress"),
        (r"vestido\s+preto|black\s+dress", "black dress"),
        (r"saia\s+preta|black\s+skirt", "black mini skirt"),
        (r"arrumand|getting ready|espelho|mirror", "getting ready mirror"),
        (r"balada|festa|club|party", "night out party"),
        (r"casa|quarto|home|bedroom", "at home"),
        (r"salto|heels", "high heels"),
        (r"selfie", "selfie"),
    ]
    for pat, eng in rules:
        if _re.search(pat, low):
            if eng not in " ".join(bits):
                bits.append(eng)
    core = " ".join(x for x in bits if x).strip() or "sexy fashion outfit"
    if "woman" not in core:
        core = "sexy young woman " + core
    if extra:
        core = f"{core} {extra}"
    q = _re.sub(r"\s+", " ", core).strip()[:120]
    print(f"[WEB] build_web_query={q!r}", flush=True)
    return q


import asyncio
import random
from typing import Any

import httpx

from app.images.outfit import location_from_scene, outfit_from_scene
from app.images.recent_guard import RECENT, make_jitter_seed

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore


# Variacao forte para nao cair sempre na mesma stock photo
STYLE_VARIANTS = [
    "full body",
    "mirror selfie",
    "candid street photo",
    "nightlife photo",
    "instagram model",
    "fashion editorial",
    "club outfit",
    "party look",
    "standing pose",
    "sitting pose",
    "side profile",
    "looking at camera",
    "over shoulder",
    "night flash photo",
    "soft light portrait",
]

POSE_VARIANTS = [
    "hands on hips",
    "one hand in hair",
    "leaning wall",
    "walking",
    "dancing pose",
    "close up torso",
    "legs crossed",
    "sitting on stool",
]

CLOTHING_BIAS = [
    "micro mini dress high heels",
    "micro mini skirt crop top heels",
    "tight sequin mini dress heels",
    "bodycon mini dress stilettos",
    "leather mini skirt crop top boots",
    "lace mini dress high heels",
]


class WebImageSearchService:
    name = "duckduckgo"

    def __init__(self, timeout: int = 45, max_results: int = 40):
        self.timeout = int(timeout)
        self.max_results = int(max_results)

    async def available(self) -> bool:
        return DDGS is not None

    def _build_queries(self, scene: str) -> list[str]:
        base = outfit_from_scene(scene)
        loc = location_from_scene(scene)
        # se outfit generico demais, injeta bias sensual
        low = base.lower()
        if "micro mini" not in low and "dress" not in low and "skirt" not in low:
            base = f"sexy young woman {random.choice(CLOTHING_BIAS)}"

        queries: list[str] = []
        for _ in range(5):
            style = random.choice(STYLE_VARIANTS)
            pose = random.choice(POSE_VARIANTS)
            parts = [base, style, pose]
            if loc:
                parts.append(loc)
            # às vezes remove "fashion portrait photo" se veio do builder antigo
            q = " ".join(parts)
            q = " ".join(q.split())
            if q not in queries:
                queries.append(q)

        # query extra bem diferente
        queries.append(
            f"sexy woman {random.choice(CLOTHING_BIAS)} {random.choice(STYLE_VARIANTS)}"
        )
        return queries

    def _search_sync(self, q: str) -> list[dict[str, Any]]:
        if DDGS is None:
            return []
        rows = []
        try:
            # region aleatoria + safesearch off para variar ranking
            region = random.choice(["wt-wt", "us-en", "br-pt", "uk-en"])
            results = DDGS().images(
                q,
                max_results=self.max_results,
                safesearch="off",
                region=region,
            )
            for item in results or []:
                url = item.get("image") or item.get("url")
                if not url or not str(url).startswith("http"):
                    continue
                if RECENT.seen(url=url):
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

        random.seed(make_jitter_seed() ^ random.randint(0, 999999))
        queries = self._build_queries(query)
        print(f"[DDG] queries({len(queries)}): {queries[0]!r} ...", flush=True)

        all_rows: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for q in queries:
            rows = await asyncio.to_thread(self._search_sync, q)
            for r in rows:
                u = r["url"]
                if u in seen_urls or RECENT.seen(url=u):
                    continue
                seen_urls.add(u)
                r["_q"] = q
                all_rows.append(r)
            if len(all_rows) >= 25:
                break

        if not all_rows:
            print("[DDG] zero results (apos filtro recent)", flush=True)
            return None

        random.shuffle(all_rows)
        print(
            f"[DDG] candidatos={len(all_rows)} recent_size={RECENT.size()}",
            flush=True,
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                # tenta varios candidatos; marca os rejeitados
                tried = 0
                for item in all_rows:
                    if tried >= 18:
                        break
                    url = item["url"]
                    if RECENT.seen(url=url):
                        continue
                    tried += 1
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
                        if len(data) < 12000:
                            continue
                        if RECENT.seen(url=url, content=data):
                            print("[DDG] skip recent hash/url", flush=True)
                            continue

                        # grava JA na busca (antes do face swap) para nao reusar
                        RECENT.remember(url=url, content=data)

                        print(
                            f"[DDG] ok bytes={len(data)} q={item.get('_q', '')[:60]!r}",
                            flush=True,
                        )
                        return {
                            "url": url,
                            "bytes": data,
                            "photographer": item.get("source") or "web",
                            "photo_id": None,
                            "alt": item.get("title") or item.get("_q") or "",
                            "query": item.get("_q") or queries[0],
                        }
                    except Exception as e:
                        print(f"[DDG] download fail: {e}", flush=True)
        except Exception as e:
            print(f"[DDG] client error: {e}", flush=True)
        return None
