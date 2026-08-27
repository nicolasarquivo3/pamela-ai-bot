"""
Album canal Telegram — match por PEDIDO + sinonimos + Gemini Vision se tags vazias.
"""
from __future__ import annotations

import base64
import random
import re
from typing import Any

from sqlalchemy import text


_STOP = {
    "uma", "para", "com", "sem", "pelo", "pela", "mais", "foto", "manda",
    "envia", "selfie", "pedido", "outfit", "source", "default", "acao",
    "contexto", "user", "espontanea", "momento", "conversa", "the", "and",
    "from", "that", "this", "with", "photo", "image", "scene", "look",
    "como", "voce", "esta", "aqui", "agora", "quero", "pode", "mim", "me",
    "de", "da", "do", "das", "dos", "uns",
}

_SYNONYMS = {
    "vestido": {"dress", "vestido", "gown"},
    "dress": {"dress", "vestido"},
    "saia": {"skirt", "saia", "miniskirt"},
    "skirt": {"skirt", "saia", "miniskirt"},
    "minissaia": {"skirt", "mini", "miniskirt", "saia"},
    "branco": {"white", "branco", "branca"},
    "branca": {"white", "branco", "branca"},
    "white": {"white", "branco", "branca"},
    "preto": {"black", "preto", "preta"},
    "preta": {"black", "preto", "preta"},
    "black": {"black", "preto", "preta"},
    "vermelho": {"red", "vermelho", "vermelha"},
    "vermelha": {"red", "vermelho", "vermelha"},
    "red": {"red", "vermelho", "vermelha"},
    "azul": {"blue", "azul"},
    "blue": {"blue", "azul"},
    "rosa": {"pink", "rosa"},
    "pink": {"pink", "rosa"},
    "salto": {"heels", "salto", "pumps"},
    "heels": {"heels", "salto"},
    "lingerie": {"lingerie"},
    "biquini": {"bikini", "biquini"},
    "bikini": {"bikini", "biquini"},
    "espelho": {"mirror", "espelho", "selfie"},
    "mirror": {"mirror", "espelho"},
    "balada": {"club", "party", "balada", "night"},
    "festa": {"party", "festa", "club"},
    "curto": {"short", "mini", "curto", "curta"},
    "curta": {"short", "mini", "curto", "curta"},
    "micro": {"micro", "mini"},
    "mini": {"mini", "micro", "short"},
    "jeans": {"jeans", "denim"},
    "couro": {"leather", "couro"},
    "leather": {"leather", "couro"},
}


def _expand_token(t: str) -> set:
    t = t.lower().strip()
    out = {t}
    if t in _SYNONYMS:
        out |= _SYNONYMS[t]
    return out


def _tokenize(s: str) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9à-ü\s\-]", " ", s)
    parts = re.split(r"[\s\-_/,|]+", s)
    out = set()
    for p in parts:
        p = p.strip()
        if len(p) < 3 or p in _STOP:
            continue
        out |= _expand_token(p)
    return out


def _pedido_from_scene(scene: str) -> str:
    s = scene or ""
    m = re.search(r"PEDIDO:\s*(.+?)(?:\s*\|\s*source=|$)", s, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    s2 = re.sub(r"OUTFIT:\s*[^|]+", " ", s, flags=re.I)
    s2 = re.sub(r"\|\s*source=\S+", " ", s2, flags=re.I)
    return s2.strip() or s


def _outfit_from_scene(scene: str) -> str:
    m = re.search(r"OUTFIT:\s*([^|]+)", scene or "", flags=re.I)
    return (m.group(1).strip() if m else "")


class AlbumService:
    def __init__(
        self,
        session,
        channel_id=None,
        enabled=True,
        use_llm_match=True,
        use_vision_match=True,
        llm=None,
    ):
        self.session = session
        self.channel_id = int(channel_id) if channel_id else None
        self.enabled = bool(enabled) and bool(self.channel_id)
        self.use_llm_match = bool(use_llm_match)
        self.use_vision_match = bool(use_vision_match)
        self.llm = llm
        self._ready = False
        self._recent_file_ids = []

    async def available(self) -> bool:
        return self.enabled

    async def ensure_table(self) -> None:
        if self._ready:
            return
        await self.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS album_photos (
                    id BIGSERIAL PRIMARY KEY,
                    file_id TEXT NOT NULL UNIQUE,
                    file_unique_id TEXT,
                    caption TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    width INT,
                    height INT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        )
        await self.session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_album_photos_created "
                "ON album_photos (created_at DESC)"
            )
        )
        await self.session.commit()
        self._ready = True
        print("[ALBUM] tabela album_photos OK", flush=True)

    def is_album_channel(self, chat_id) -> bool:
        if not self.enabled or chat_id is None or self.channel_id is None:
            return False
        return int(chat_id) == int(self.channel_id)

    async def ingest_telegram_photo(
        self,
        file_id: str,
        file_unique_id=None,
        caption_hint=None,
        width=None,
        height=None,
        generate_caption=False,
    ) -> bool:
        if not self.enabled:
            return False
        file_id = (file_id or "").strip()
        if not file_id:
            return False
        await self.ensure_table()
        caption = (caption_hint or "").strip()
        tags = " ".join(sorted(_tokenize(caption))[:32])
        try:
            await self.session.execute(
                text(
                    """
                    INSERT INTO album_photos
                        (file_id, file_unique_id, caption, tags, width, height)
                    VALUES
                        (:file_id, :fuid, :caption, :tags, :w, :h)
                    ON CONFLICT (file_id) DO UPDATE SET
                        caption = CASE
                            WHEN EXCLUDED.caption <> '' THEN EXCLUDED.caption
                            ELSE album_photos.caption
                        END,
                        tags = CASE
                            WHEN EXCLUDED.tags <> '' THEN EXCLUDED.tags
                            ELSE album_photos.tags
                        END
                    """
                ),
                {
                    "file_id": file_id,
                    "fuid": file_unique_id or None,
                    "caption": caption,
                    "tags": tags,
                    "w": width,
                    "h": height,
                },
            )
            await self.session.commit()
            print(
                f"[ALBUM] salva file_id={file_id[:20]}... "
                f"tags={tags[:60]!r} caption={caption[:60]!r}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[ALBUM] ingest error: {e}", flush=True)
            try:
                await self.session.rollback()
            except Exception:
                pass
            return False

    async def count(self) -> int:
        if not self.enabled:
            return 0
        await self.ensure_table()
        r = await self.session.execute(text("SELECT COUNT(*) FROM album_photos"))
        return int(r.scalar() or 0)

    def _score_row(self, row, pedido_toks, outfit_toks) -> float:
        blob = f"{row.get('caption') or ''} {row.get('tags') or ''}".lower()
        photo_toks = _tokenize(blob)
        score = 0.0
        if pedido_toks:
            score += len(pedido_toks & photo_toks) * 3.0
            for t in pedido_toks:
                if t in blob:
                    score += 1.5
        if outfit_toks:
            score += len(outfit_toks & photo_toks) * 1.0
        color_pedido = pedido_toks & {
            "white", "branco", "branca", "black", "preto", "preta",
            "red", "vermelho", "blue", "azul", "pink", "rosa",
        }
        if color_pedido and photo_toks:
            opposites = {
                "white": {"black", "preto", "preta"},
                "branco": {"black", "preto", "preta"},
                "branca": {"black", "preto", "preta"},
                "black": {"white", "branco", "branca"},
                "preto": {"white", "branco", "branca"},
                "preta": {"white", "branco", "branca"},
            }
            for c in color_pedido:
                bad = opposites.get(c, set())
                if bad & photo_toks and not (color_pedido & photo_toks):
                    score -= 4.0
        if "dress" in pedido_toks or "vestido" in pedido_toks:
            if photo_toks & {"skirt", "saia", "miniskirt"} and not (
                photo_toks & {"dress", "vestido"}
            ):
                score -= 3.0
        if "skirt" in pedido_toks or "saia" in pedido_toks:
            if photo_toks & {"dress", "vestido"} and not (
                photo_toks & {"skirt", "saia"}
            ):
                score -= 3.0
        if not photo_toks and not blob.strip():
            score = 0.05
        if row.get("file_id") in self._recent_file_ids:
            score -= 2.5
        return score

    async def pick_best(self, scene: str):
        if not self.enabled:
            return None
        await self.ensure_table()
        r = await self.session.execute(
            text(
                "SELECT id, file_id, caption, tags FROM album_photos "
                "ORDER BY id DESC LIMIT 1200"
            )
        )
        rows = [dict(x) for x in r.mappings().all()]
        if not rows:
            print("[ALBUM] vazio", flush=True)
            return None

        pedido = _pedido_from_scene(scene)
        outfit = _outfit_from_scene(scene)
        pedido_toks = _tokenize(pedido)
        outfit_toks = _tokenize(outfit)
        print(
            f"[ALBUM] pedido={pedido[:80]!r} outfit={outfit[:60]!r} "
            f"pedido_toks={list(pedido_toks)[:15]}",
            flush=True,
        )

        scored = [
            (self._score_row(row, pedido_toks, outfit_toks), row) for row in rows
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:10]
        best_score = top[0][0] if top else 0.0
        labeled = sum(
            1 for row in rows if (row.get("caption") or row.get("tags") or "").strip()
        )
        print(
            f"[ALBUM] candidatos={len(rows)} top_scores="
            f"{[round(s, 2) for s, _ in top[:5]]} "
            f"fotos_com_tag={labeled}/{len(rows)}",
            flush=True,
        )

        need_vision = (
            self.use_vision_match
            and best_score < 2.5
            and bool(pedido_toks or outfit_toks)
        )
        if need_vision:
            pool_ids = []
            for s, d in top[:5]:
                pool_ids.append(d)
            recent = rows[: min(50, len(rows))]
            random.shuffle(recent)
            for d in recent:
                if d not in pool_ids:
                    pool_ids.append(d)
                if len(pool_ids) >= 6:
                    break
            print(
                f"[ALBUM] vision match em {len(pool_ids)} candidatas "
                f"(score texto={best_score:.2f})",
                flush=True,
            )
            vision_pick = await self._vision_choose(scene, pedido or outfit, pool_ids)
            if vision_pick is not None:
                print(f"[ALBUM] VISION escolheu id={vision_pick.get('id')}", flush=True)
                self._remember(vision_pick.get("file_id"))
                return vision_pick

        if best_score < 0.5:
            pool = [
                d
                for d in rows[:50]
                if d.get("file_id") not in self._recent_file_ids
            ] or rows[:50]
            chosen = random.choice(pool)
            print("[ALBUM] score baixo e vision falhou -> aleatoria", flush=True)
        else:
            pool = [d for s, d in top if s >= best_score - 1.0] or [top[0][1]]
            chosen = random.choice(pool)

        print(
            f"[ALBUM] escolhida id={chosen.get('id')} "
            f"cap={str(chosen.get('caption') or '')[:40]!r}",
            flush=True,
        )
        self._remember(chosen.get("file_id"))
        return chosen

    def _remember(self, fid):
        if not fid:
            return
        self._recent_file_ids.append(fid)
        self._recent_file_ids = self._recent_file_ids[-15:]

    async def _download_file(self, file_id: str):
        try:
            from app.config import settings
            import httpx

            token = (settings.telegram_bot_token or "").strip()
            if not token or not file_id:
                return None
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(
                    f"https://api.telegram.org/bot{token}/getFile",
                    params={"file_id": file_id},
                )
                if r.status_code != 200:
                    return None
                data = r.json()
                path = (data.get("result") or {}).get("file_path")
                if not path:
                    return None
                r2 = await client.get(
                    f"https://api.telegram.org/file/bot{token}/{path}"
                )
                if r2.status_code != 200 or len(r2.content) < 500:
                    return None
                return r2.content
        except Exception as e:
            print(f"[ALBUM] vision download err: {e}", flush=True)
            return None

    def _gemini_keys(self):
        try:
            from app.config import settings

            keys = []
            multi = getattr(settings, "gemini_api_keys", None) or ""
            single = getattr(settings, "gemini_api_key", None) or ""
            for part in str(multi).replace(";", ",").split(","):
                k = part.strip()
                if k and k not in keys:
                    keys.append(k)
            if single.strip() and single.strip() not in keys:
                keys.append(single.strip())
            return keys
        except Exception:
            return []

    async def _vision_choose(self, scene: str, pedido: str, pool: list):
        import httpx

        keys = self._gemini_keys()
        if not keys:
            print("[ALBUM] vision: sem GEMINI key", flush=True)
            return None

        images_b64 = []
        valid_pool = []
        for p in pool[:5]:
            raw = await self._download_file(p["file_id"])
            if not raw or len(raw) > 4_000_000:
                continue
            images_b64.append(base64.b64encode(raw).decode("ascii"))
            valid_pool.append(p)

        if not valid_pool:
            print("[ALBUM] vision: nenhuma imagem baixada", flush=True)
            return None
        if len(valid_pool) == 1:
            return valid_pool[0]

        prompt = (
            "Voce escolhe UMA foto de um album de uma mulher adulta.\n"
            f"Pedido do usuario: {pedido[:200]}\n"
            f"Contexto: {scene[:250]}\n"
            f"Ha {len(valid_pool)} fotos numeradas 1 a {len(valid_pool)}.\n"
            "Escolha a que MELHOR combina (roupa, cor, tipo de peca).\n"
            "Se pediu VESTIDO BRANCO, priorize vestido branco; ignore saia preta.\n"
            "Responda SOMENTE com o numero (ex: 2)."
        )
        parts = [{"text": prompt}]
        for i, b64 in enumerate(images_b64, 1):
            parts.append({"text": f"Foto {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        model = "gemini-2.0-flash"
        try:
            from app.config import settings as _s

            model = getattr(_s, "gemini_model", None) or model
            if "lite" in str(model):
                model = "gemini-2.0-flash"
        except Exception:
            pass

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0.1},
        }

        for key in keys[:4]:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            try:
                async with httpx.AsyncClient(timeout=90) as client:
                    r = await client.post(
                        url,
                        headers={
                            "x-goog-api-key": key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                print(
                    f"[ALBUM] vision HTTP {r.status_code} key={key[:6]}...",
                    flush=True,
                )
                if r.status_code == 429:
                    continue
                if r.status_code != 200:
                    print(f"[ALBUM] vision body={r.text[:300]}", flush=True)
                    continue
                data = r.json()
                cands = data.get("candidates") or []
                if not cands:
                    continue
                parts_out = (cands[0].get("content") or {}).get("parts") or []
                text_out = "".join(
                    x.get("text", "") for x in parts_out if x.get("text")
                ).strip()
                print(f"[ALBUM] vision resposta={text_out!r}", flush=True)
                m = re.search(r"([1-9][0-9]*)", text_out)
                if not m:
                    continue
                n = int(m.group(1)) - 1
                if 0 <= n < len(valid_pool):
                    return valid_pool[n]
            except Exception as e:
                print(f"[ALBUM] vision err: {e}", flush=True)
                continue
        return None
