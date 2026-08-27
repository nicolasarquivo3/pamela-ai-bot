"""
Album canal Telegram.
- Ingest: Gemini Vision legenda automatica (roupa/cor/local)
- Match: PEDIDO + tags + Vision se precisar
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
    out = set()
    for p in re.split(r"[\s\-_/,|]+", s):
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
    return m.group(1).strip() if m else ""


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
        if generate_caption and not caption:
            try:
                raw = await self._download_file(file_id)
                if raw:
                    ai_cap = await self._auto_caption_vision(raw)
                    if ai_cap:
                        caption = ai_cap
                        print(f"[ALBUM] auto-caption={caption[:80]!r}", flush=True)
            except Exception as e:
                print(f"[ALBUM] auto-caption fail: {e}", flush=True)
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
        if ("dress" in pedido_toks or "vestido" in pedido_toks) and (
            photo_toks & {"skirt", "saia", "miniskirt"}
        ) and not (photo_toks & {"dress", "vestido"}):
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

        if (
            self.use_vision_match
            and best_score < 2.5
            and bool(pedido_toks or outfit_toks)
        ):
            pool_ids = [d for _, d in top[:5]]
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
                d for d in rows[:50] if d.get("file_id") not in self._recent_file_ids
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
            print(f"[ALBUM] download err: {e}", flush=True)
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
            if keys:
                print(
                    f"[ALBUM] gemini keys={len(keys)} "
                    f"prefixes={[k[:8]+'...' for k in keys[:4]]}",
                    flush=True,
                )
            return keys
        except Exception:
            return []

    _models_cache: list | None = None

    def _vision_models(self) -> list:
        """Prefer o mesmo modelo do chat + descoberta via ListModels."""
        if AlbumService._models_cache:
            return list(AlbumService._models_cache)

        models = []
        try:
            from app.config import settings

            m = (getattr(settings, "gemini_model", None) or "").strip()
            if m:
                models.append(m)
        except Exception:
            pass
        # nomes atuais (2026) + legados
        for m in (
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ):
            if m not in models:
                models.append(m)
        return models

    async def _discover_models(self, key: str) -> list:
        """Pergunta a API quais modelos a key realmente ve."""
        import httpx

        found = []
        for base in (
            "https://generativelanguage.googleapis.com/v1beta/models",
            "https://generativelanguage.googleapis.com/v1/models",
        ):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(
                        base,
                        headers={"x-goog-api-key": key},
                        params={"pageSize": 100},
                    )
                print(
                    f"[ALBUM] listModels {base.split('.com')[-1]} "
                    f"HTTP {r.status_code} key={key[:8]}...",
                    flush=True,
                )
                if r.status_code != 200:
                    print(f"[ALBUM] listModels body={r.text[:250]}", flush=True)
                    continue
                data = r.json()
                for m in data.get("models") or []:
                    name = (m.get("name") or "").replace("models/", "")
                    methods = m.get("supportedGenerationMethods") or m.get(
                        "supported_generation_methods"
                    ) or []
                    # se nao listar methods, ainda tenta
                    if methods and "generateContent" not in methods:
                        continue
                    if name and name not in found:
                        found.append(name)
                if found:
                    print(
                        f"[ALBUM] modelos disponiveis ({len(found)}): "
                        f"{found[:12]}",
                        flush=True,
                    )
                    # prioriza flash com vision
                    preferred = []
                    for needle in (
                        "flash-lite",
                        "2.5-flash",
                        "3.7-flash",
                        "3.5-flash",
                        "flash",
                    ):
                        for n in found:
                            if needle in n and n not in preferred:
                                preferred.append(n)
                    for n in found:
                        if n not in preferred:
                            preferred.append(n)
                    AlbumService._models_cache = preferred[:20]
                    return preferred[:20]
            except Exception as e:
                print(f"[ALBUM] listModels err: {e}", flush=True)
        return []

    async def _auto_caption_vision(self, image_bytes: bytes) -> str:
        """Legenda com Gemini Vision — suporta key AQ. (Auth) e AIza (legacy)."""
        import httpx
        import asyncio
        import base64 as _b64

        keys = self._gemini_keys()
        if not keys or not image_bytes:
            print("[ALBUM] caption abort: sem key ou bytes", flush=True)
            return ""

        raw = image_bytes[:3_500_000]
        b64 = _b64.b64encode(raw).decode("ascii")

        # modelo do chat primeiro
        models = self._vision_models()
        try:
            discovered = await self._discover_models(keys[0])
            if discovered:
                models = discovered + [m for m in models if m not in discovered]
        except Exception as e:
            print(f"[ALBUM] discover skip: {e}", flush=True)

        prompt = (
            "Descreva esta foto de mulher adulta em UMA linha curta em portugues, "
            "so fatos visuais: tipo de roupa, cor, acessorios, local. "
            "Exemplo: vestido branco curto salto alto espelho. "
            "Sem nome. Sem emoji. Maximo 12 palavras."
        )

        # dois formatos de part (snake e camel) — APIs variam
        parts_snake = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]
        parts_camel = [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
        ]

        def build_payload(parts):
            return {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "maxOutputTokens": 40,
                    "temperature": 0.2,
                },
            }

        def urls_for(model: str, key: str) -> list:
            """AQ. keys costumam ser Auth/Vertex Express; AIza = AI Studio."""
            out = []
            if key.startswith("AQ.") or key.startswith("AQ_"):
                out.extend(
                    [
                        f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent",
                        f"https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model}:generateContent",
                    ]
                )
            out.extend(
                [
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
                ]
            )
            return out

        # 1) tenta SDK google-genai se instalado (melhor com AQ.)
        try:
            from google import genai
            from google.genai import types

            for key in keys[:3]:
                for vertex_flag in (True, False):
                    try:
                        client = genai.Client(api_key=key, vertexai=vertex_flag)
                        for model in models[:6]:
                            try:
                                print(
                                    f"[ALBUM] caption SDK vertex={vertex_flag} "
                                    f"model={model} key={key[:8]}...",
                                    flush=True,
                                )
                                resp = client.models.generate_content(
                                    model=model,
                                    contents=[
                                        types.Content(
                                            role="user",
                                            parts=[
                                                types.Part.from_text(text=prompt),
                                                types.Part.from_bytes(
                                                    data=raw,
                                                    mime_type="image/jpeg",
                                                ),
                                            ],
                                        )
                                    ],
                                )
                                text = (getattr(resp, "text", None) or "").strip()
                                if text:
                                    text = text.replace("\n", " ")[:120]
                                    print(
                                        f"[ALBUM] caption ok SDK model={model}",
                                        flush=True,
                                    )
                                    return text
                            except Exception as e:
                                print(
                                    f"[ALBUM] caption SDK fail model={model}: {e}",
                                    flush=True,
                                )
                                continue
                    except Exception as e:
                        print(f"[ALBUM] caption SDK client fail: {e}", flush=True)
        except ImportError:
            print(
                "[ALBUM] google-genai nao instalado — usando REST",
                flush=True,
            )

        # 2) REST multi-endpoint
        for model in models[:8]:
            for key in keys[:3]:
                for url in urls_for(model, key):
                    for parts in (parts_camel, parts_snake):
                        payload = build_payload(parts)
                        try:
                            async with httpx.AsyncClient(timeout=60) as client:
                                r = await client.post(
                                    url,
                                    headers={
                                        "x-goog-api-key": key,
                                        "Content-Type": "application/json",
                                    },
                                    params={"key": key},
                                    json=payload,
                                )
                            short = url.split(".com")[-1][:60]
                            print(
                                f"[ALBUM] caption REST {short} "
                                f"model={model} HTTP {r.status_code} "
                                f"key={key[:8]}...",
                                flush=True,
                            )
                            if r.status_code == 404:
                                print(
                                    f"[ALBUM] 404 body={r.text[:160]}",
                                    flush=True,
                                )
                                # troca modelo/url
                                break  # next url format/parts still try? break parts
                            if r.status_code in (429, 503):
                                await asyncio.sleep(2)
                                continue
                            if r.status_code != 200:
                                print(
                                    f"[ALBUM] body={r.text[:200]}",
                                    flush=True,
                                )
                                continue
                            data = r.json()
                            cands = data.get("candidates") or []
                            if not cands:
                                continue
                            pr = (cands[0].get("content") or {}).get("parts") or []
                            text = "".join(
                                x.get("text", "") for x in pr if x.get("text")
                            ).strip()
                            text = text.replace("\n", " ").strip().strip('"')[:120]
                            if text:
                                print(
                                    f"[ALBUM] caption ok REST model={model}",
                                    flush=True,
                                )
                                return text
                        except Exception as e:
                            print(f"[ALBUM] caption REST err: {e}", flush=True)
                            continue
                    else:
                        continue
                    # 404 broke inner parts loop
                    continue
        print("[ALBUM] caption: todas tentativas falharam", flush=True)
        return ""

    async def backfill_captions(self, limit: int = 20) -> int:
        await self.ensure_table()
        r = await self.session.execute(
            text(
                """
                SELECT id, file_id FROM album_photos
                WHERE (caption IS NULL OR caption = '')
                   OR (tags IS NULL OR tags = '')
                ORDER BY id DESC
                LIMIT :lim
                """
            ),
            {"lim": int(limit)},
        )
        rows = list(r.mappings().all())
        done = 0
        for row in rows:
            raw = await self._download_file(row["file_id"])
            if not raw:
                continue
            cap = await self._auto_caption_vision(raw)
            if not cap:
                continue
            tags = " ".join(sorted(_tokenize(cap))[:32])
            await self.session.execute(
                text(
                    "UPDATE album_photos SET caption=:c, tags=:t WHERE id=:id"
                ),
                {"c": cap, "t": tags, "id": row["id"]},
            )
            await self.session.commit()
            done += 1
            print(f"[ALBUM] backfill id={row['id']} cap={cap[:60]!r}", flush=True)
        return done

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
            return None
        if len(valid_pool) == 1:
            return valid_pool[0]

        prompt = (
            "Voce escolhe UMA foto de um album de uma mulher adulta.\n"
            f"Pedido: {pedido[:200]}\n"
            f"Contexto: {scene[:250]}\n"
            f"Fotos 1 a {len(valid_pool)}. Escolha a que melhor combina "
            "(roupa, cor). Se pediu VESTIDO BRANCO, ignore saia preta.\n"
            "Responda SOMENTE o numero."
        )
        parts = [{"text": prompt}]
        for i, b64 in enumerate(images_b64, 1):
            parts.append({"text": f"Foto {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0.1},
        }
        for model in self._vision_models():
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
                        f"[ALBUM] vision model={model} HTTP {r.status_code} "
                        f"key={key[:6]}...",
                        flush=True,
                    )
                    if r.status_code == 404:
                        break
                    if r.status_code == 429:
                        continue
                    if r.status_code != 200:
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
