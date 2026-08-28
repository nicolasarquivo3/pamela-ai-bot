"""
Memoria de longo prazo: fatos estaveis do casal / personagem / usuario.

Tipos: fact, preference, boundary, milestone, place, person, kink_soft
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text


class LongTermMemoryService:
    def __init__(self, session, max_in_context: int = 20):
        self.session = session
        self.max_in_context = max_in_context
        self._ready = False

    async def ensure_table(self):
        if self._ready:
            return
        await self.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    character_id BIGINT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    key TEXT,
                    content TEXT NOT NULL,
                    importance SMALLINT NOT NULL DEFAULT 5,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (user_id, character_id, key)
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_user
                ON long_term_memories (user_id, character_id, active, importance DESC)
                """
            )
        )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
        self._ready = True
        print("[LTM] tabela long_term_memories OK", flush=True)

    async def seed_defaults(self, user_id: int, character_id: int):
        """Fatos base do casal (idempotente por key)."""
        await self.ensure_table()
        defaults = [
            ("fact", "ritmo_meses", "A historia do casal avanca em MESES de vida real. Nao comprimir hotwife em poucas mensagens. Maioria dos dias e namoro normal + visual.", 10),

            ("fact", "virgindade", "Pâmela perdeu a virgindade com ele; ele é o único até hoje no tempo da história.", 10),
            ("fact", "idades", "Pâmela tem 20 anos; ele cerca de 30.", 8),
            ("fact", "chamado", "Ela o chama de amor.", 7),
            ("fact", "rotina", "Ela estuda (faculdade à noite), trabalha e dorme algumas noites com ele.", 8),
            ("preference", "musica", "Ela adora sertanejo e pagode e dançar; ele não dança e não curte dançar.", 7),
            ("preference", "saidas", "Gostam de balada (foco), barzinho e praia às vezes.", 7),
            ("kink_soft", "exib", "Eles brincam com roupa micro, exibicionismo visual e contar na cama; evolução hotwife é lenta e por escolha dela.", 9),
            ("boundary", "humilhacao", "Sem humilhar ele; provocação de ciúme só bobinha de namorados; tudo é pelo casal.", 10),
            ("boundary", "sentimento_outro", "Outros homens, se um dia existirem no RP, são só físicos — sem namoro/sentimento.", 10),
            ("milestone", "danca_colada", "Já rolou: balada, ela bêbada, ele no banheiro, ela dançou sertanejo colada com um cara; ciúme na hora, conversa e tesão depois na cama.", 9),
            (
                "fact",
                "cena_atual",
                "NARRATIVA ATUAL (recomeço): os dois estão se arrumando para ir à balada como de costume. "
                "Ela se arruma bem gostosa (roupa micro/sensual no estilo que o casal curte); enquanto isso "
                "bebem e conversam como namorados — clima da noite, o que esperar da balada, ciúmes leves, "
                "provocação, carinho. Ainda NÃO chegaram na balada; o momento é o 'antes' em casa (ou onde "
                "se preparam). Sem pular para beijo/sexo com outros. Foco: arrumação, bebida, papo de casal e balada.",
                10,
            ),
            (
                "fact",
                "cena_atual_fase",
                "Fase da noite: PREPARAÇÃO para balada (se arrumando + bebendo + conversando). "
                "Próximo passo natural: sair / chegar na balada / dançar / eventuais surpresas leves.",
                9,
            ),
            ("boundary", "nao_esta_hotwife_ainda", "No recomeco: so visual/fantasia; NAO narrar sexo com outros como fato cedo.", 10),

        ]
        for kind, key, content, imp in defaults:
            await self.session.execute(
                text(
                    """
                    INSERT INTO long_term_memories
                        (user_id, character_id, kind, key, content, importance)
                    SELECT :u, :c, :k, :key, :content, :imp
                    WHERE NOT EXISTS (
                        SELECT 1 FROM long_term_memories
                        WHERE user_id=:u AND character_id=:c AND key=:key AND active
                    )
                    """
                ),
                {
                    "u": user_id,
                    "c": character_id,
                    "k": kind,
                    "key": key,
                    "content": content,
                    "imp": imp,
                },
            )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()

    async def upsert(
        self,
        user_id: int,
        character_id: int,
        content: str,
        kind: str = "fact",
        key: str | None = None,
        importance: int = 6,
    ):
        await self.ensure_table()
        content = (content or "").strip()
        if len(content) < 8:
            return
        key = (key or re.sub(r"\W+", "_", content[:40].lower())).strip("_")[:80]
        await self.session.execute(
            text(
                """
                INSERT INTO long_term_memories
                    (user_id, character_id, kind, key, content, importance, updated_at)
                VALUES (:u, :c, :k, :key, :content, :imp, NOW())
                ON CONFLICT (user_id, character_id, key) DO UPDATE SET
                    content = EXCLUDED.content,
                    importance = GREATEST(long_term_memories.importance, EXCLUDED.importance),
                    kind = EXCLUDED.kind,
                    active = TRUE,
                    updated_at = NOW()
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "k": kind,
                "key": key,
                "content": content[:2000],
                "imp": max(1, min(10, importance)),
            },
        )
        # se key exists, update content
        await self.session.execute(
            text(
                """
                UPDATE long_term_memories SET content=:content, importance=GREATEST(importance,:imp),
                    updated_at=NOW(), active=TRUE
                WHERE user_id=:u AND character_id=:c AND key=:key
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "key": key,
                "content": content[:2000],
                "imp": importance,
            },
        )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()

    async def set_current_scene(
        self,
        user_id: int,
        character_id: int,
        content: str,
        key: str = "cena_atual",
    ):
        await self.upsert(
            user_id,
            character_id,
            content,
            kind="fact",
            key=key,
            importance=10,
        )

    async def maybe_extract(

        self,
        user_id: int,
        character_id: int,
        user_text: str,
        reply_text: str,
    ):
        """Heuristica leve: grava marcos sem LLM extra."""
        blob = f"{user_text or ''} {reply_text or ''}".lower()
        await self.seed_defaults(user_id, character_id)

        rules = [
            (r"\bfacul|faculdade|aula\b", "place_facul", "place", "Faculdade à noite faz parte da rotina dela.", 6),
            (r"\btrabalh", "work", "fact", "Ela trabalha (além de estudar).", 6),
            (r"\bpraia\b", "praia", "place", "Já foram / falam em ir à praia juntos.", 5),
            (r"\bbarzinho\b|\bbar\b", "bar", "place", "Gostam de barzinho.", 5),
            (r"\bsem calcinha\b", "sem_calcinha", "kink_soft", "Já brincaram com ela sair sem calcinha no clima.", 7),
            (r"\bsem suti[aã]\b", "sem_sutia", "kink_soft", "Já brincaram com ela sem sutiã no clima.", 6),
            (r"\bci[uú]me", "ciume_talk", "milestone", "Ciúme dele já foi assunto entre os dois (provocação leve, sem humilhação).", 7),
            (r"\bamigas?\b", "amigas", "person", "Ela tem amigas com quem sai (balada/facul).", 5),
            (r"\bdormi|dormir|pernoite|passar a noite\b", "dormir_junto", "fact", "Eles dormem juntos em algumas noites.", 6),
        ]
        for pat, key, kind, content, imp in rules:
            if re.search(pat, blob, re.I):
                await self.upsert(user_id, character_id, content, kind=kind, key=key, importance=imp)

        # marco de evento se reply/user menciona danca colada
        if re.search(r"dan[cç]a\s+colad|colad[oa].*cara|cara.*colad", blob, re.I):
            await self.upsert(
                user_id,
                character_id,
                "Marco: dança colada com um cara na balada (ciúme + conversa na cama depois).",
                kind="milestone",
                key="danca_colada",
                importance=9,
            )

    async def recall(
        self,
        user_id: int,
        character_id: int,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        await self.ensure_table()
        await self.seed_defaults(user_id, character_id)
        lim = limit or self.max_in_context
        r = await self.session.execute(
            text(
                """
                SELECT id, kind, key, content, importance
                FROM long_term_memories
                WHERE user_id=:u AND character_id=:c AND active
                ORDER BY importance DESC, updated_at DESC
                LIMIT :lim
                """
            ),
            {"u": user_id, "c": character_id, "lim": lim},
        )
        rows = [dict(x) for x in r.mappings().all()]
        if query:
            q = query.lower()
            toks = [t for t in re.split(r"\W+", q) if len(t) > 3]

            def score(row):
                c = (row.get("content") or "").lower()
                s = int(row.get("importance") or 0)
                for t in toks:
                    if t in c:
                        s += 2
                return s

            rows.sort(key=score, reverse=True)
        return rows[:lim]

    def format_for_prompt(self, rows: list[dict]) -> str:
        if not rows:
            return "(sem memorias de longo prazo ainda)"
        lines = []
        for r in rows:
            lines.append(
                f"- [{r.get('kind')}|{r.get('key')}] {r.get('content')}"
            )
        return "\n".join(lines)
