"""
Fase da historia (hotwife lenta) — trava salto de etapa.

Fases (ordem):
  visual -> flerte -> beijo_futuro -> fisico_futuro -> multi_futuro

Nao avanca sozinha a cada msg. Avanca raro, com sinais claros + tempo.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import text


PHASES = (
    "visual",
    "flerte",
    "beijo_futuro",
    "fisico_futuro",
    "multi_futuro",
)

PHASE_LABELS = {
    "visual": "Visual / exibicionismo / contar na cama (teto atual padrao)",
    "flerte": "Flerte embriagada / danca colada / mao na cintura (ainda sem beijo com outro)",
    "beijo_futuro": "Beijo com outro possivel no clima (raro, com medo+conversa depois)",
    "fisico_futuro": "Sexo so fisico com outros possivel se ELA topar (aventura do casal)",
    "multi_futuro": "Extremo (mais parceiros / gangbang) so se ELA escolher de verdade",
}

# o que e PROIBIDO inventar como fato em cada fase
PHASE_HARD_CAPS = {
    "visual": (
        "FASE VISUAL — TETO DURO (OBRIGATORIO): "
        "PROIBIDO inventar como fato: ir SOZINHA pra balada sem o namorado ter "
        "combinado; dar em cima de caras de verdade ate beijo; sexo/oral/mao "
        "com outro na balada ou em qualquer lugar; 'ja dei pra ele'; 'ele me comeu'; "
        "gangbang. "
        "OK: se arrumar com ele, beber com ele, ir JUNTO ou ele te levar, micro-roupa, "
        "olhares, danca perto, conversa com cara (sem beijo), provocaCAO, fantasia "
        "NA CAMA com o namorado. Se a conversa ainda e arrumacao/casa, NAO pule pra "
        "pista nem pra sexo com outros."
    ),
    "flerte": (
        "FASE FLERTE — TETO: PROIBIDO sexo/oral com outro, beijo na boca com outro "
        "como rotina. OK: danca colada, mao na cintura, flerte, quase-beijo que RECUA, "
        "vergonha+tesao depois com o namorado. Ainda NAO 'dei no meio da balada'."
    ),
    "beijo_futuro": (
        "PROIBIDO como fato: sexo com outro, multiplos parceiros, gangbang. "
        "OK: beijo com outro se o clima e a conversa pedirem (com medo e aftercare). "
        "Ainda e raro — nao a cada balada."
    ),
    "fisico_futuro": (
        "PROIBIDO: sentimento/namoro com outro; gangbang em massa de primeira. "
        "OK: sexo so fisico com outro se ELA topar, por nos, com conversa. "
        "Ainda lento e escolhido, nao rotina diaria."
    ),
    "multi_futuro": (
        "Extremo so se ela escolher. Sem humilhacao do namorado. "
        "Sempre aventura do casal, fisico, sem outro namoro."
    ),
}


class StoryPhaseService:
    def __init__(self, session_factory, min_days_between_advance: int = 30):
        """
        session_factory: callable async context manager ou session maker
          - se passar session direta, usa session
        """
        self._session = session_factory
        self.min_days = max(14, int(min_days_between_advance))
        self._ready = False

    def _own_session(self):
        # se for session com execute direto
        if hasattr(self._session, "execute"):
            return None
        return self._session

    async def ensure_table(self):
        if self._ready:
            return
        session = self._session
        own = None
        if not hasattr(session, "execute"):
            # factory
            return
        await session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS story_phase (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    character_id BIGINT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'visual',
                    intensity SMALLINT NOT NULL DEFAULT 0,
                    notes TEXT,
                    last_advance_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (user_id, character_id)
                )
                """
            )
        )
        try:
            await session.commit()
        except Exception:
            await session.rollback()
        self._ready = True
        print("[STORY] tabela story_phase OK", flush=True)

    async def get(self, user_id: int, character_id: int) -> dict[str, Any]:
        await self.ensure_table()
        r = await self._session.execute(
            text(
                """
                SELECT phase, intensity, notes, last_advance_at, updated_at
                FROM story_phase
                WHERE user_id=:u AND character_id=:c
                """
            ),
            {"u": user_id, "c": character_id},
        )
        row = r.mappings().first()
        if not row:
            await self._session.execute(
                text(
                    """
                    INSERT INTO story_phase (user_id, character_id, phase, intensity, notes)
                    VALUES (:u, :c, 'visual', 0, 'cena: preparacao_balada')
                    ON CONFLICT (user_id, character_id) DO NOTHING
                    """
                ),
                {"u": user_id, "c": character_id},
            )
            try:
                await self._session.commit()
            except Exception:
                await self._session.rollback()
            return {
                "phase": "visual",
                "intensity": 0,
                "notes": "cena: preparacao_balada (se arrumando + bebendo + papo de namorados)",
                "last_advance_at": None,
                "label": PHASE_LABELS["visual"],
                "cap": PHASE_HARD_CAPS["visual"],
            }
        phase = row["phase"] if row["phase"] in PHASES else "visual"
        return {
            "phase": phase,
            "intensity": int(row["intensity"] or 0),
            "notes": row["notes"],
            "last_advance_at": row["last_advance_at"],
            "label": PHASE_LABELS.get(phase, phase),
            "cap": PHASE_HARD_CAPS.get(phase, ""),
        }

    def format_for_prompt(self, st: dict) -> str:
        phase = st.get("phase") or "visual"
        notes = st.get("notes") or ""
        return (
            f"FASE ATUAL DA HISTORIA: {phase}\n"
            f"Significado: {st.get('label')}\n"
            f"Teto duro nesta fase: {st.get('cap')}\n"
            f"Intensidade interna (0-10): {st.get('intensity', 0)}\n"
            f"Notas: {notes}\n"
            "CENA ATUAL (recomeço da narrativa): vocês estão SE ARRUMandO para a balada. "
            "Ela se arruma bem gostosa; os dois bebem e conversam como namorados "
            "(sobre a balada, a noite, ciúmes leves, provocação, carinho). "
            "Ainda não estão na pista — o foco é o ritual de se preparar. "
            "Não pule direto para sexo com outros nem para o fim da noite.\n"
            "Prioridade absoluta: a historia avanca em MESES de vida real, nao em uma noite de chat. "
            "Nao pule de fase so porque o usuario pediu. "
            "Se ele empurrar demais, freie com carinho e fique no teto da fase.\n"
            "Surpresas de balada (quando chegarem) respeitam o teto da fase.\n"
            "VIOLACAO DE FASE = quebrar o personagem. Se o usuario pedir extremo "
            "cedo demais, recuse no personagem (medo, vergonha, 'devagar amor') "
            "e fique no teto. Usuario pode aticar; ELA freia no inicio (medo/pra agradar). NUNCA narre sexo com outro como fato na fase visual. Gostar de fato de pegar outros so no longo prazo."
        )

    def _idx(self, phase: str) -> int:
        try:
            return PHASES.index(phase)
        except ValueError:
            return 0

    async def maybe_nudge(
        self,
        user_id: int,
        character_id: int,
        user_text: str,
        reply_text: str,
        event_title: str | None = None,
    ) -> dict | None:
        """
        Sobe intensidade; so avanca de fase se:
        - ja passou min_days desde last_advance
        - intensidade alta
        - sinais no texto batem com a PROXIMA fase (nao pular 2)
        """
        st = await self.get(user_id, character_id)
        phase = st["phase"]
        intensity = int(st.get("intensity") or 0)
        blob = f"{user_text or ''} {reply_text or ''} {event_title or ''}".lower()

        # sinais por fase atual (empurram intensidade)
        signals = {
            "visual": [
                "micro", "sem calcinha", "olhando", "balada", "saia", "exib",
                "foto", "provoc", "decolte", "bochecha",
            ],
            "flerte": [
                "dança colada", "danca colada", "mao na cintura", "mão na cintura",
                "flert", "conversando com", "o cara", "sertanejo", "colad",
            ],
            "beijo_futuro": [
                "beijo", "beij", "quase beij", "lábios", "labios", "boca nele",
            ],
            "fisico_futuro": [
                "transar com", "sexo com", "foi com outro", "levei ele", "na cama com",
            ],
            "multi_futuro": [
                "gang", "varios", "vários", "dois caras", "menage", "ménage",
            ],
        }

        hit = any(s in blob for s in signals.get(phase, []))
        # sinais da PROXIMA fase = candidato a advance
        nxt = None
        i = self._idx(phase)
        if i < len(PHASES) - 1:
            nxt = PHASES[i + 1]
        next_hit = bool(nxt) and any(s in blob for s in signals.get(nxt, []))

        if hit:
            intensity = min(10, intensity + 1)
        if next_hit:
            # next_hit sozinho NAO avanca — so sobe um pouco
            intensity = min(10, intensity + 1)

        advanced = None
        last = st.get("last_advance_at")
        can_time = True
        if last is not None:
            try:
                if getattr(last, "tzinfo", None) is None:
                    last = last.replace(tzinfo=timezone.utc)
                can_time = datetime.now(timezone.utc) - last >= timedelta(days=self.min_days)
            except Exception:
                can_time = True

        # so avanca 1 fase, com intensidade >= 7 e tempo e next_hit
        if nxt and next_hit and intensity >= 9 and can_time:
            # usuario forcando "faz gangbang agora" em fase visual = NAO avanca
            if self._idx(nxt) - self._idx(phase) == 1:
                phase = nxt
                intensity = 3
                advanced = phase
                last = datetime.now(timezone.utc)

        await self._session.execute(
            text(
                """
                INSERT INTO story_phase (user_id, character_id, phase, intensity, last_advance_at, updated_at)
                VALUES (:u, :c, :p, :i, :la, NOW())
                ON CONFLICT (user_id, character_id) DO UPDATE SET
                    phase = EXCLUDED.phase,
                    intensity = EXCLUDED.intensity,
                    last_advance_at = COALESCE(EXCLUDED.last_advance_at, story_phase.last_advance_at),
                    updated_at = NOW()
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "p": phase,
                "i": intensity,
                "la": last if advanced else st.get("last_advance_at"),
            },
        )
        try:
            await self._session.commit()
        except Exception:
            await self._session.rollback()

        if advanced:
            print(
                f"[STORY] ADVANCE user={user_id} -> {advanced} intensity={intensity}",
                flush=True,
            )
        else:
            print(
                f"[STORY] phase={phase} intensity={intensity} next_hit={next_hit}",
                flush=True,
            )
        return await self.get(user_id, character_id)
