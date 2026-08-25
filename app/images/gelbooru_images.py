"""
Gelbooru — API grátis NSFW.
Majoridade anime. Preferencia: micro saia / micro vestido.
"""
from __future__ import annotations

import random
import re
from typing import Any

import httpx


class GelbooruSearchService:
    name = "gelbooru"

    BASE_TAGS = (
        "1girl solo miniskirt high_heels rating:questionable",
        "1girl solo dress short_dress high_heels rating:questionable",
        "1girl solo microskirt crop_top rating:questionable",
        "1girl solo bodycon_dress high_heels rating:questionable",
        "1girl solo miniskirt stockings high_heels rating:questionable",
        "1girl solo short_dress cleavage high_heels rating:questionable",
    )

    def __init__(self, timeout: int = 40, limit: int = 30):
        self.timeout = int(timeout)
        self.limit = min(int(limit), 100)
        self.api = "https://gelbooru.com/index.php"

    async def available(self) -> bool:
        return True

    def _tags(self, raw: str) -> str:
        text = (raw or "").lower()
        m = re.search(r"contexto da fotografia:\s*(.+)", text, re.I | re.S)
        if m:
            text = m.group(1)

        tags = ["1girl", "solo"]
        mapping = (
            (r"\blingerie\b|\blina\b", "lingerie"),
            (r"\bvestido\b", "short_dress"),
            (r"\bsaia\b|\bminissaia\b", "miniskirt"),
            (r"\bsalto\b", "high_heels"),
            (r"\bbota\b", "boots"),
            (r"\bpraia\b|\bbikini\b", "bikini"),
            (r"\bmeia\b|\bstocking\b", "stockings"),
            (r"\bdecote\b", "cleavage"),
            (r"\bpreto\b|\bpreta\b", "black_dress"),
            (r"\bbranco\b|\bbranca\b", "white_dress"),
            (r"\bquarto\b|\bcama\b", "bedroom"),
            (r"\bcropped\b", "crop_top"),
        )
        for pat, tag in mapping:
            if re.search(pat, text, re.I) and tag not in tags:
                tags.append(tag)

        if not re.search(r"\blingerie\b|\blina\b", text, re.I):
            if "lingerie" in tags:
                tags.remove("lingerie")
            if "miniskirt" not in tags and "short_dress" not in tags:
                tags.append(random.choice(["miniskirt", "short_dress", "microskirt"]))

        if len(tags) <= 2:
            return random.choice(self.BASE_TAGS)

        tags.append("rating:questionable")
        q = " ".join(tags)
        print(f"[GELBOORU] tags: {q}", flush=True)
        return q

    async def search(self, query: str) -> dict[str, Any] | None:
        tags = self._tags(query)
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "limit": str(self.limit),
            "tags": tags,
        }
        headers = {
            "User-Agent": "pamela-ai-bot/1.0 (face-swap; personal use)",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                r = await client.get(self.api, params=params)
                if r.status_code != 200:
                    print(f"[GELBOORU] status={r.status_code}", flush=True)
                    return None

                data = r.json()
                posts = data.get("post") if isinstance(data, dict) else data
                if not posts:
                    print("[GELBOORU] zero posts", flush=True)
                    return None
                if isinstance(posts, dict):
                    posts = [posts]

                random.shuffle(posts)
                for post in posts[:12]:
                    url = (
                        post.get("file_url")
                        or post.get("sample_url")
                        or post.get("preview_url")
                    )
                    if not url or not str(url).startswith("http"):
                        continue
                    try:
                        img = await client.get(url)
                        if img.status_code != 200 or len(img.content) < 5000:
                            continue
                        print(
                            f"[GELBOORU] ok id={post.get('id')} bytes={len(img.content)}",
                            flush=True,
                        )
                        return {
                            "url": url,
                            "bytes": img.content,
                            "photographer": "gelbooru",
                            "photo_id": post.get("id"),
                            "alt": post.get("tags") or tags,
                            "query": tags,
                        }
                    except Exception as e:
                        print(f"[GELBOORU] download fail: {e}", flush=True)
                        continue
        except Exception as e:
            print(f"[GELBOORU] error: {e}", flush=True)

        print("[GELBOORU] nenhuma imagem baixavel", flush=True)
        return None
