"""
Album de fotos via CANAL do Telegram.

1) Voce posta fotos no canal (bot e admin)
2) Bot salva file_id + caption/tags no Postgres
3) Na hora da foto: escolhe a mais parecida com a scene do chat

Env:
  ALBUM_CHANNEL_ID=-1004349291324
  ALBUM_ENABLED=true
  ALBUM_USE_LLM_MATCH=true   # opcional, grátis (Gemini/OpenRouter)
  ALBUM_FIRST=true           # tenta album antes de IA
"""
from __future__ import annotations

import re
import random
from typing import Any

from sqlalchemy import text


# palavras de cena que extraimos do contexto do chat
_STOP = {
    "uma", "para", "com", "sem", "pelo", "pela", "mais", "foto", "manda",
    "envia", "selfie", "pedido", "outfit", "source", "default", "acao",
    "contexto", "user", "espontanea", "momento", "conversa", "the", "and",
    "from", "that", "this", "with", "photo", "image", "scene",
}


def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9à-ü\s\-]", " ", s)
    parts = re.split(r"[\s\-_/,|]+", s)
    out = set()
    for p in parts:
        p = p.strip()
        if len(p) < 3 or p in _STOP:
            continue
        out.add(p)
    return out


class AlbumService:
    def __init__(
        self,
        session,
        channel_id: int | None = None,
        enabled: bool = True,
        use_llm_match: bool = True,
        llm=None,
    ):
        self.session = session
        self.channel_id = int(channel_id) if channel_id else None
        self.enabled = bool(enabled) and bool(self.channel_id)
        self.use_llm_match = bool(use_llm_match)
        self.llm = llm
        self._ready = False
        self._recent_file_ids: list[str] = []

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

    def is_album_channel(self, chat_id: int | None) -> bool:
        if not self.enabled or chat_id is None or self.channel_id is None:
            return False
        return int(chat_id) == int(self.channel_id)

    async def ingest_telegram_photo(
        self,
        file_id: str,
        file_unique_id: str | None = None,
        caption_hint: str | None = None,
        width: int | None = None,
        height: int | None = None,
        generate_caption: bool = False,
    ) -> bool:
        """Salva foto do channel_post. generate_caption=False no import em massa."""
        if not self.enabled:
            return False
        file_id = (file_id or "").strip()
        if not file_id:
            return False

        await self.ensure_table()

        caption = (caption_hint or "").strip()
        tags = ""

        # import em massa: nao chama LLM (economiza cota)
        if generate_caption and self.llm and not caption:
            try:
                caption = await self._auto_caption()
            except Exception as e:
                print(f"[ALBUM] caption fail: {e}", flush=True)

        tags = self._tags_from_text(caption)

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

    async def _auto_caption(self) -> str:
        """1 linha de tags — so se generate_caption=True (nao no bulk)."""
        # sem baixar a imagem: caption generica; match real vem das tags do chat
        return ""

    def _tags_from_text(self, text_in: str) -> str:
        toks = sorted(_tokenize(text_in))
        return " ".join(toks[:24])

    async def count(self) -> int:
        if not self.enabled:
            return 0
        await self.ensure_table()
        r = await self.session.execute(text("SELECT COUNT(*) FROM album_photos"))
        return int(r.scalar() or 0)

    async def pick_best(self, scene: str) -> dict[str, Any] | None:
        """
        Escolhe a foto mais alinhada a scene do chat.
        Retorna dict com file_id, caption, score.
        """
        if not self.enabled:
            return None
        await self.ensure_table()

        r = await self.session.execute(
            text(
                "SELECT id, file_id, caption, tags FROM album_photos "
                "ORDER BY id DESC LIMIT 800"
            )
        )
        rows = list(r.mappings().all())
        if not rows:
            print("[ALBUM] vazio — nenhuma foto indexada", flush=True)
            return None

        scene_toks = _tokenize(scene)
        # boost palavras de roupa/local que voce usa muito
        for extra in (
            "mini", "dress", "vestido", "saia", "heels", "salto", "balada",
            "club", "mirror", "espelho", "selfie", "lingerie", "sensual",
            "party", "noite", "casa", "sofa", "gym", "academia",
        ):
            if extra in (scene or "").lower():
                scene_toks.add(extra)

        scored: list[tuple[float, dict]] = []
        for row in rows:
            blob = f"{row.get('caption') or ''} {row.get('tags') or ''}".lower()
            photo_toks = _tokenize(blob)
            if not photo_toks and not blob.strip():
                # sem tags: score baixo mas entra no pool
                score = 0.1
            else:
                inter = scene_toks & photo_toks
                score = float(len(inter))
                # bonus substring
                for t in list(scene_toks)[:20]:
                    if t in blob:
                        score += 0.35
            # evita repetir a mesma logo em seguida
            if row["file_id"] in self._recent_file_ids:
                score -= 2.0
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:8]
        print(
            f"[ALBUM] candidatos={len(rows)} top_scores="
            f"{[round(s, 2) for s, _ in top[:5]]} scene_toks={list(scene_toks)[:12]}",
            flush=True,
        )

        # se todo mundo score ~0, aleatorio entre recentes
        best_score = top[0][0] if top else 0
        if best_score < 0.5:
            pick = random.choice(rows[: min(40, len(rows))])
            chosen = dict(pick)
            print("[ALBUM] score baixo -> aleatoria recente", flush=True)
        else:
            # entre as top com score perto do melhor
            pool = [d for s, d in top if s >= best_score - 0.8]
            chosen = random.choice(pool) if pool else top[0][1]

            # LLM free opcional so se empatar varias
            if (
                self.use_llm_match
                and self.llm
                and len(pool) >= 3
                and best_score >= 1.0
            ):
                try:
                    llm_pick = await self._llm_choose(scene, pool[:5])
                    if llm_pick is not None and 0 <= llm_pick < len(pool):
                        chosen = pool[llm_pick]
                        print(f"[ALBUM] LLM escolheu index={llm_pick}", flush=True)
                except Exception as e:
                    print(f"[ALBUM] LLM match skip: {e}", flush=True)

        fid = chosen.get("file_id")
        if fid:
            self._recent_file_ids.append(fid)
            self._recent_file_ids = self._recent_file_ids[-12:]

        print(
            f"[ALBUM] escolhida id={chosen.get('id')} "
            f"file={str(fid)[:24]}... cap={str(chosen.get('caption') or '')[:40]!r}",
            flush=True,
        )
        return chosen

    async def _llm_choose(self, scene: str, pool: list[dict]) -> int | None:
        lines = []
        for i, p in enumerate(pool, 1):
            cap = (p.get("caption") or p.get("tags") or "sem descricao")[:80]
            lines.append(f"{i}) {cap}")
        prompt = (
            "Voce escolhe a foto de um album que melhor combina com a cena.\n"
            f"Cena: {scene[:300]}\n"
            "Opcoes:\n"
            + "\n".join(lines)
            + "\nResponda SO com o numero (1-{})".format(len(pool))
        )
        # interface do router: generate(system, messages)
        text = await self.llm.generate(
            "Responda apenas com um numero inteiro.",
            [{"role": "user", "content": prompt}],
        )
        if not text:
            return None
        m = re.search(r"([1-9][0-9]*)", str(text))
        if not m:
            return None
        n = int(m.group(1)) - 1
        if 0 <= n < len(pool):
            return n
        return None
