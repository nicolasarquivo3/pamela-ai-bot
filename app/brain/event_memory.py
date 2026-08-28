"""
Memoria de eventos / noites / momentos nao triviais.

- Detecta quando a conversa fala de balada, noite, casa, provocacao, briga, etc.
- Gera um resumo curto (PT) com Gemini (se disponivel) ou regra simples.
- Guarda no Postgres (tabela event_memories).
- O contexto do bot puxa os N eventos mais relevantes.

100% gratis (usa as mesmas keys Gemini).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text


# palavras que sugerem evento emocional / nao trivial
EVENT_CUES = re.compile(
    r"(?i)\b("
    r"balada|festa|night|clube|bar|sa[ií]mos|saida|"
    r"noite|madrugada|ontem|hoje\s+[àa]\s+noite|"
    r"em\s+casa|fomos\s+pra\s+casa|depois\s+(que|da|do)|"
    r"provoc|excit|beijo|sexo|fizemos|aconteceu|"
    r"briga|discuss[aã]o|ci[uú]me|chor|triste|feliz|"
    r"te\s+amei|eu\s+te\s+amo|saudade|nosso\s+momento|"
    r"viagem|hotel|praia|shopping|cinema|jantar|"
    r"primeira\s+vez|n[aã]o\s+esque[cç]o|lembra"
    r")\b"
)

TRIVIAL = re.compile(
    r"(?i)^(oi|ola|ol[aá]|bom\s+dia|boa\s+tarde|boa\s+noite|kkk|haha|ok|blz|sim|n[aã]o|❤️|🥰|😘)+\s*$"
)


class EventMemoryService:
    def __init__(
        self,
        session,
        llm=None,
        max_events_in_context: int = 8,
        session_factory=None,
    ):
        self.session = session
        self.llm = llm
        self.max_events_in_context = max_events_in_context
        self.session_factory = session_factory
        self._table_ready = False

    def set_llm(self, llm):
        self.llm = llm

    async def ensure_table(self):
        if self._table_ready:
            return
        await self.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS event_memories (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    character_id BIGINT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL,
                    keywords TEXT NOT NULL DEFAULT '',
                    emotion TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_hint TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_event_mem_user
                ON event_memories (user_id, character_id, updated_at DESC)
                """
            )
        )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
        self._table_ready = True
        print("[EVENT-MEM] tabela event_memories OK", flush=True)

    def is_eventish(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < 12:
            return False
        if TRIVIAL.match(t):
            return False
        return bool(EVENT_CUES.search(t))

    async def maybe_capture(
        self,
        user_id: int,
        character_id: int,
        user_text: str,
        reply_text: str = "",
        recent_lines: list[str] | None = None,
    ) -> dict | None:
        """
        Se a fala do usuario (ou o par) parece evento, cria/atualiza resumo.
        Nao bloqueia a conversa se falhar.
        """
        try:
            await self.ensure_table()
            blob = f"{user_text or ''}\n{reply_text or ''}"
            if not self.is_eventish(user_text) and not self.is_eventish(blob):
                return None

            # evita spam: se ja tem evento quase igual nas ultimas 2h, so atualiza
            r = await self.session.execute(
                text(
                    """
                    SELECT id, summary FROM event_memories
                    WHERE user_id = :u AND character_id = :c
                      AND updated_at > NOW() - INTERVAL '3 hours'
                    ORDER BY updated_at DESC
                    LIMIT 3
                    """
                ),
                {"u": user_id, "c": character_id},
            )
            recent = list(r.fetchall())

            context_snip = ""
            if recent_lines:
                context_snip = "\n".join(recent_lines[-12:])

            summary = await self._build_summary(
                user_text=user_text,
                reply_text=reply_text,
                context_snip=context_snip,
            )
            if not summary:
                return None

            title = self._title_from(summary, user_text)
            keywords = self._keywords(user_text + " " + reply_text)
            emotion = self._emotion_guess(user_text + " " + reply_text)
            importance = self._importance(user_text, reply_text)

            # merge com evento recente se keywords se sobrepoem
            merged_id = None
            for row in recent:
                old = (row[1] or "").lower()
                if any(k in old for k in keywords.split()[:4] if len(k) > 3):
                    merged_id = row[0]
                    break

            if merged_id:
                await self.session.execute(
                    text(
                        """
                        UPDATE event_memories
                        SET summary = :s, title = :t, keywords = :k,
                            emotion = :e, importance = GREATEST(importance, :i),
                            updated_at = NOW(), source_hint = :h
                        WHERE id = :id
                        """
                    ),
                    {
                        "s": summary[:1200],
                        "t": title[:120],
                        "k": keywords[:300],
                        "e": emotion[:80],
                        "i": importance,
                        "h": (user_text or "")[:200],
                        "id": merged_id,
                    },
                )
                try:
                    await self.session.commit()
                except Exception:
                    await self.session.rollback()
                print(f"[EVENT-MEM] update id={merged_id} title={title!r}", flush=True)
                return {"id": merged_id, "title": title, "summary": summary, "updated": True}

            ins = await self.session.execute(
                text(
                    """
                    INSERT INTO event_memories
                    (user_id, character_id, title, summary, keywords, emotion, importance, source_hint)
                    VALUES (:u, :c, :t, :s, :k, :e, :i, :h)
                    RETURNING id
                    """
                ),
                {
                    "u": user_id,
                    "c": character_id,
                    "t": title[:120],
                    "s": summary[:1200],
                    "k": keywords[:300],
                    "e": emotion[:80],
                    "i": importance,
                    "h": (user_text or "")[:200],
                },
            )
            new_id = ins.scalar()
            try:
                await self.session.commit()
            except Exception:
                await self.session.rollback()
            print(f"[EVENT-MEM] new id={new_id} title={title!r}", flush=True)
            return {"id": new_id, "title": title, "summary": summary, "updated": False}
        except Exception as e:
            print(f"[EVENT-MEM] capture fail: {e}", flush=True)
            try:
                await self.session.rollback()
            except Exception:
                pass
            return None

    async def _build_summary(
        self, user_text: str, reply_text: str, context_snip: str
    ) -> str:
        # tenta LLM
        if self.llm is not None:
            try:
                system = (
                    "Voce resume momentos de um roleplay romantico/adulto ficcional. "
                    "Escreva em portugues brasileiro, 2 a 4 frases, no passado, "
                    "fatos emocionais e o que aconteceu (lugar, clima, o que fizeram). "
                    "Sem meta-comentario, sem ingles, sem [foto]. "
                    "Se for sensual, resuma o clima sem pornografia grafica."
                )
                user = (
                    f"Trechos recentes:\n{context_snip[:1500]}\n\n"
                    f"Usuario agora: {user_text}\n"
                    f"Personagem: {reply_text[:400]}\n\n"
                    "Resumo do EVENTO/MOMENTO (titulo implicito no texto):"
                )
                # LLMRouter / Gemini interface
                if hasattr(self.llm, "generate"):
                    out = await self.llm.generate(
                        system,
                        [
                            {"role": "user", "content": user},
                        ],
                    )
                    if out and len(out.strip()) > 20:
                        # se veio CoT, pega so se for PT
                        if not re.search(r"(?i)okay,? let's see", out):
                            return out.strip()[:1200]
            except Exception as e:
                print(f"[EVENT-MEM] llm summary fail: {e}", flush=True)

        # fallback deterministico
        place = "um momento juntos"
        if re.search(r"(?i)balada|festa|clube", user_text):
            place = "a balada/festa"
        elif re.search(r"(?i)casa", user_text):
            place = "em casa"
        elif re.search(r"(?i)noite", user_text):
            place = "aquela noite"
        return (
            f"Momento marcado {place}: o usuario relembrou/descreveu "
            f"\"{(user_text or '')[:180]}\". "
            f"Clima emocional envolvido; a personagem deve tratar como memoria compartilhada."
        )[:800]

    def _title_from(self, summary: str, user_text: str) -> str:
        if re.search(r"(?i)balada|festa", user_text + summary):
            return "Noite na balada/festa"
        if re.search(r"(?i)casa|depois", user_text + summary):
            return "Depois em casa"
        if re.search(r"(?i)briga|discuss", user_text + summary):
            return "Discussao/briga"
        if re.search(r"(?i)viagem|hotel|praia", user_text + summary):
            return "Viagem/passeio"
        first = (summary or user_text or "Momento")[:60]
        return first.split(".")[0][:80] or "Momento especial"

    def _keywords(self, text: str) -> str:
        words = re.findall(r"[a-zA-ZÀ-ÿ]{4,}", (text or "").lower())
        stop = {
            "para", "como", "quando", "onde", "isso", "aqui", "voce", "você",
            "ela", "ele", "muito", "minha", "meu", "nossa", "depois", "antes",
            "fazer", "feito", "lembra", "lembrar",
        }
        out = []
        for w in words:
            if w in stop:
                continue
            if w not in out:
                out.append(w)
            if len(out) >= 12:
                break
        return " ".join(out)

    def _emotion_guess(self, text: str) -> str:
        t = (text or "").lower()
        if re.search(r"ci[uú]me|briga|raiva", t):
            return "tensao"
        if re.search(r"triste|chor|saudade", t):
            return "melancolia"
        if re.search(r"provoc|excit|gostos|sexy|fogo", t):
            return "desejo"
        if re.search(r"amor|amei|carinh", t):
            return "carinho"
        if re.search(r"balada|festa|rir|kkk", t):
            return "euforia"
        return "intenso"

    def _importance(self, user_text: str, reply_text: str) -> float:
        t = f"{user_text} {reply_text}".lower()
        score = 0.5
        if re.search(r"lembra|nunca|sempre|primeira", t):
            score += 0.2
        if re.search(r"balada|casa|noite|sexo|provoc", t):
            score += 0.2
        if re.search(r"briga|chor|amo", t):
            score += 0.15
        return min(1.0, score)

    async def recall_for_context(
        self,
        user_id: int,
        character_id: int,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        await self.ensure_table()
        lim = limit or self.max_events_in_context
        # busca simples por keyword overlap + recentes importantes
        q = (query or "").lower()
        r = await self.session.execute(
            text(
                """
                SELECT id, title, summary, keywords, emotion, importance, updated_at
                FROM event_memories
                WHERE user_id = :u AND character_id = :c
                ORDER BY importance DESC, updated_at DESC
                LIMIT 40
                """
            ),
            {"u": user_id, "c": character_id},
        )
        rows = list(r.fetchall())
        scored = []
        for row in rows:
            title, summary, kw = row[1] or "", row[2] or "", row[3] or ""
            sc = float(row[5] or 0.5)
            if q:
                for tok in re.findall(r"[a-zA-ZÀ-ÿ]{3,}", q):
                    if tok in (title + " " + summary + " " + kw).lower():
                        sc += 0.35
            scored.append((sc, row))
        scored.sort(key=lambda x: -x[0])
        out = []
        for sc, row in scored[:lim]:
            out.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "keywords": row[3],
                    "emotion": row[4],
                    "importance": row[5],
                    "score": sc,
                }
            )
        return out

    def format_for_prompt(self, events: list[dict]) -> str:
        if not events:
            return "(nenhum evento marcado ainda)"
        lines = []
        for e in events:
            lines.append(
                f"- [{e.get('title')}] ({e.get('emotion') or 'clima'}): {e.get('summary')}"
            )
        return "\n".join(lines)
