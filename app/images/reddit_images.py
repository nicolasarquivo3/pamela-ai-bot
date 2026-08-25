"""
Busca imagens no Reddit (JSON publico, sem API key).
Fashion / lingerie adult 18+.
No Render pode retornar 403 — o service cai no Pexels.
"""
from __future__ import annotations

import random
import re
from typing import Any

import httpx


class RedditImageSearchService:
    name = "reddit"

    SUBREDDITS = (
        "lingerie",
        "Fashion_Sexiness",
        "Sexy",
        "NSFW_Fashion",
        "selfies",
        "GoneMild",
        "clubwear",
        "highheels",
        "corsets",
        "bodysuits",
    )

    KEYWORD_SUBS = (
        (r"\blingerie\b|\blina\b", ("lingerie", "GoneMild")),
        (r"\bvestido\b|\bdress\b", ("Fashion_Sexiness", "NSFW_Fashion", "Sexy")),
        (r"\bsaia\b|\bminissaia\b", ("Fashion_Sexiness", "Sexy")),
        (r"\bcropped\b|\bcrop\b", ("Fashion_Sexiness", "clubwear")),
        (r"\bbalada\b|\bclub\b|\bfesta\b", ("clubwear", "Sexy")),
        (r"\bsalto\b|\bheels\b|\bbota\b", ("highheels", "Fashion_Sexiness")),
        (r"\bcorset\b|\bcors[eé]\b", ("corsets", "Fashion_Sexiness")),
        (r"\bacademia\b|\bgym\b", ("Sexy", "GoneMild")),
        (r"\bpraia\b|\bbikini\b", ("Sexy", "GoneMild")),
    )

    def __init__(self, timeout: int = 40, limit: int = 30):
        self.timeout = int(timeout)
        self.limit = min(int(limit), 50)

    async def available(self) -> bool:
        return True

    def _pick_subs(self, raw: str) -> list[str]:
        text = (raw or "").lower()
        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.I | re.S)
        if m:
            text = m.group(1)

        chosen: list[str] = []
        for pat, subs in self.KEYWORD_SUBS:
            if re.search(pat, text, re.I):
                for s in subs:
                    if s not in chosen:
                        chosen.append(s)

        if not chosen:
            chosen = list(self.SUBREDDITS)

        random.shuffle(chosen)
        return chosen[:3]

    def _extract_image_url(self, post: dict) -> str | None:
        data = post.get("data") or {}
        url = (data.get("url_overridden_by_dest") or data.get("url") or "").strip()
        if not url:
            return None

        lower = url.lower()
        if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
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
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }

        candidates: list[dict[str, Any]] = []
        bases = (
            "https://www.reddit.com",
            "https://old.reddit.com",
        )

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                for sub in subs:
                    for base in bases:
                        api = f"{base}/r/{sub}/hot.json"
                        try:
                            r = await client.get(
                                api,
                                params={"limit": self.limit, "raw_json": 1},
                            )
                            if r.status_code == 403:
                                print(f"[REDDIT] {sub} 403 via {base}", flush=True)
                                continue
                            if r.status_code != 200:
                                print(
                                    f"[REDDIT] {sub} status={r.status_code}",
                                    flush=True,
                                )
                                continue

                            children = (
                                (r.json().get("data") or {}).get("children") or []
                            )
                            for post in children:
                                img = self._extract_image_url(post)
                                if not img:
                                    continue
                                title = (post.get("data") or {}).get("title") or ""
                                candidates.append(
                                    {"url": img, "title": title, "sub": sub}
                                )
                            break
                        except Exception as e:
                            print(f"[REDDIT] fetch {sub} error: {e}", flush=True)

                if not candidates:
                    print(
                        "[REDDIT] zero candidates (comum no Render/cloud)",
                        flush=True,
                    )
                    return None

                random.shuffle(candidates)

                for item in candidates[:12]:
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
                            f"[REDDIT] ok r/{item['sub']} bytes={len(data)} "
                            f"title={item['title'][:50]!r}",
                            flush=True,
                        )
                        return {
                            "url": url,
                            "bytes": data,
                            "photographer": f"reddit/r/{item['sub']}",
                            "photo_id": None,
                            "alt": item["title"],
                            "query": f"reddit:{item['sub']}",
                        }
                    except Exception as e:
                        print(f"[REDDIT] download fail: {e}", flush=True)
                        continue
        except Exception as e:
            print(f"[REDDIT] client error: {e}", flush=True)

        print("[REDDIT] nenhuma imagem baixavel", flush=True)
        return None
