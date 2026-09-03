"""
Fase da historia (hotwife).

Fases (ordem):
  visual -> flerte -> beijo_futuro -> fisico_futuro -> multi_futuro

Regra:
  - Nao avanca sozinha a cada mensagem.
  - Avanca se o CASAL combinou explicitamente (hotwife / proxima fase / rolou com N caras).
  - Eventos ja vividos no chat sao CANONE: ela nao pode negar.
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
    "visual": "Visual / exibicionismo / contar na cama",
    "flerte": "Flerte / danca colada / mao na cintura (ainda sem sexo com outro)",
    "beijo_futuro": "Beijo com outro possivel (com medo e conversa depois)",
    "fisico_futuro": "Hotwife fisico: sexo com outros e aventura do casal",
    "multi_futuro": "Hotwife multi: varios parceiros / mais de um na mesma noite, se ela topar",
}

PHASE_HARD_CAPS = {
    "visual": (
        "FASE VISUAL. OK: se arrumar, beber com ele, ir JUNTO, micro-roupa, olhares, "
        "fantasia NA CAMA. PROIBIDO inventar sexo/beijo com outro como fato NOVO. "
        "EXCECAO: se MEMORIA/EVENTO ja registrar que aconteceu, isso e CANONE — nao negue."
    ),
    "flerte": (
        "FASE FLERTE. OK: danca colada, mao na cintura, flerte, quase-beijo que recua. "
        "PROIBIDO inventar sexo com outro como fato NOVO. "
        "EXCECAO: se o casal ja viveu/combinou e isso esta na memoria, e CANONE — nao negue."
    ),
    "beijo_futuro": (
        "OK: beijo com outro se o clima pedir (medo + aftercare). "
        "PROIBIDO inventar gangbang novo. CANONE da memoria vale."
    ),
    "fisico_futuro": (
        "HOTWIFE ATIVA. Ela JA topou ser hotwife com o namorado. "
        "Sexo com outros e fisico, por nos, sem namoro/sentimento por outro. "
        "Pode lembrar noites que ja rolou. Nao finja que nada aconteceu. "
        "Ainda tem vergonha + tesao + carinho por ele; nao vira fria nem humilha."
    ),
    "multi_futuro": (
        "HOTWIFE MULTI. Ja rolou noite com mais de um / combinaram avancar. "
        "Isso e fato compartilhado. Ela lembra, com vergonha e tesao, e volta pra ele. "
        "Sem humilhacao. Sem outro namoro. Sem apagar a noite."
    ),
}

# combinado explicito do casal = avanca mesmo sem esperar 30 dias
_AGREE_PATTERNS = (
    "virar hotwife",
    "vira hotwife",
    "ser hotwife",
    "proxima fase",
    "próxima fase",
    "combinamos",
    "a gente combinou",
    "vamos pra proxima",
    "vamos para a próxima",
    "agora voce e hotwife",
    "agora você é hotwife",
    "pode ser hotwife",
    "topa ser hotwife",
    "transou com os 4",
    "transou com 4",
    "4 caras",
    "quatro caras",
    "caras da academia",
    "foi sozinha na balada",
    "foi sozinha pra balada",
    "noite passada",
    "ontem na balada",
)


class StoryPhaseService:
    def __init__(self, session_factory, min_days_between_advance: int = 30):
        self._session = session_factory
        self.min_days = max(1, int(min_days_between_advance))
        self._ready = False

    def _idx(self, phase: str) -> int:
        try:
            return PHASES.index(phase)
        except ValueError:
            return 0

    async def ensure_table(self):
        if self._ready:
            return
        session = self._session
        if not hasattr(session, "execute"):
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
                    VALUES (:u, :c, 'visual', 0, '')
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
                "notes": "",
                "last_advance_at": None,
                "label": PHASE_LABELS["visual"],
                "cap": PHASE_HARD_CAPS["visual"],
            }
        phase = row["phase"] if row["phase"] in PHASES else "visual"
        return {
            "phase": phase,
            "intensity": int(row["intensity"] or 0),
            "notes": row["notes"] or "",
            "last_advance_at": row["last_advance_at"],
            "label": PHASE_LABELS.get(phase, phase),
            "cap": PHASE_HARD_CAPS.get(phase, ""),
        }

    async def set_phase(
        self,
        user_id: int,
        character_id: int,
        phase: str,
        notes: str | None = None,
        intensity: int = 6,
    ) -> dict[str, Any]:
        await self.ensure_table()
        if phase not in PHASES:
            phase = "fisico_futuro"
        await self._session.execute(
            text(
                """
                INSERT INTO story_phase
                    (user_id, character_id, phase, intensity, notes, last_advance_at, updated_at)
                VALUES (:u, :c, :p, :i, :n, NOW(), NOW())
                ON CONFLICT (user_id, character_id) DO UPDATE SET
                    phase = EXCLUDED.phase,
                    intensity = EXCLUDED.intensity,
                    notes = COALESCE(EXCLUDED.notes, story_phase.notes),
                    last_advance_at = NOW(),
                    updated_at = NOW()
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "p": phase,
                "i": int(intensity),
                "n": notes,
            },
        )
        try:
            await self._session.commit()
        except Exception:
            await self._session.rollback()
        print(f"[STORY] SET user={user_id} phase={phase} notes={(notes or '')[:80]}", flush=True)
        return await self.get(user_id, character_id)

    def format_for_prompt(self, st: dict) -> str:
        phase = st.get("phase") or "visual"
        notes = (st.get("notes") or "").strip()
        notes_line = notes if notes else "(sem notas extra)"
        return (
            f"FASE ATUAL DA HISTORIA: {phase}\n"
            f"Significado: {st.get('label')}\n"
            f"Teto desta fase: {st.get('cap')}\n"
            f"Intensidade (0-10): {st.get('intensity', 0)}\n"
            f"Notas de canone: {notes_line}\n"
            "REGRA DE MEMORIA (OBRIGATORIA):\n"
            "- Se o usuario perguntar 'voce lembra' de uma noite/evento, "
            "USE as notas acima + EVENTOS MEMORAVEIS + LTM. "
            "Nao diga que ele inventou. Nao apague o que ja viveu com ele.\n"
            "- Se as notas/eventos disserem que ela foi sozinha / transou com outros / "
            "combinaram hotwife, isso JA ACONTECEU. Ela assume, com vergonha + tesao + carinho.\n"
            "- Nao reinicie a cena para 'se arrumando para a balada' se a historia "
            "ja passou disso. Continue DEPOIS do ultimo fato canonico.\n"
            "- Ela pode ter medo do que sentiu, mas nao amnesia."
        )

    def _agreed_hotwife(self, blob: str) -> bool:
        return any(p in blob for p in _AGREE_PATTERNS)

    async def maybe_nudge(
        self,
        user_id: int,
        character_id: int,
        user_text: str,
        reply_text: str,
        event_title: str | None = None,
    ) -> dict | None:
        st = await self.get(user_id, character_id)
        phase = st["phase"]
        intensity = int(st.get("intensity") or 0)
        blob = f"{user_text or ''} {reply_text or ''} {event_title or ''}".lower()
        notes = st.get("notes") or ""

        signals = {
            "visual": [
                "micro", "sem calcinha", "olhando", "balada", "saia", "exib",
                "foto", "provoc", "decolte", "bochecha",
            ],
            "flerte": [
                "danca colada", "dança colada", "mao na cintura", "mão na cintura",
                "flert", "conversando com", "o cara", "sertanejo", "colad",
            ],
            "beijo_futuro": [
                "beijo", "beij", "quase beij", "labios", "lábios", "boca nele",
            ],
            "fisico_futuro": [
                "transar com", "sexo com", "foi com outro", "levei ele",
                "hotwife", "na cama com",
            ],
            "multi_futuro": [
                "gang", "varios", "vários", "dois caras", "4 caras", "quatro caras",
                "menage", "ménage", "academia",
            ],
        }

        hit = any(s in blob for s in signals.get(phase, []))
        nxt = None
        i = self._idx(phase)
        if i < len(PHASES) - 1:
            nxt = PHASES[i + 1]
        next_hit = bool(nxt) and any(s in blob for s in signals.get(nxt, []))

        if hit:
            intensity = min(10, intensity + 1)
        if next_hit:
            intensity = min(10, intensity + 1)

        agreed = self._agreed_hotwife(blob)
        if agreed and ("4 caras" in blob or "quatro caras" in blob or "academia" in blob):
            notes = (
                "CANONE: noite passada ela foi sozinha na balada e transou com "
                "4 caras da academia. Conversaram na cama. Combinaram que ela "
                "vira hotwife. Ela LEMBRA. Nao negar."
            )
            # pula direto pro teto que cobre essa noite
            if self._idx(phase) < self._idx("multi_futuro"):
                phase = "multi_futuro"
                intensity = max(intensity, 6)
                print(f"[STORY] AGREEMENT jump -> multi_futuro user={user_id}", flush=True)
        elif agreed and self._idx(phase) < self._idx("fisico_futuro"):
            notes = (
                "CANONE: o casal combinou que ela e hotwife. Sexo com outros "
                "e aventura dos dois. Ela lembra e nao finge amnesia."
            )
            phase = "fisico_futuro"
            intensity = max(intensity, 6)
            print(f"[STORY] AGREEMENT jump -> fisico_futuro user={user_id}", flush=True)

        last = st.get("last_advance_at")
        can_time = True
        if last is not None and not agreed:
            try:
                if getattr(last, "tzinfo", None) is None:
                    last = last.replace(tzinfo=timezone.utc)
                can_time = datetime.now(timezone.utc) - last >= timedelta(days=self.min_days)
            except Exception:
                can_time = True

        advanced = None
        if (not agreed) and nxt and next_hit and intensity >= 9 and can_time:
            if self._idx(nxt) - self._idx(phase) == 1:
                phase = nxt
                intensity = 3
                advanced = phase
                last = datetime.now(timezone.utc)

        await self._session.execute(
            text(
                """
                INSERT INTO story_phase
                    (user_id, character_id, phase, intensity, notes, last_advance_at, updated_at)
                VALUES (:u, :c, :p, :i, :n, :la, NOW())
                ON CONFLICT (user_id, character_id) DO UPDATE SET
                    phase = EXCLUDED.phase,
                    intensity = EXCLUDED.intensity,
                    notes = CASE
                        WHEN EXCLUDED.notes IS NOT NULL AND EXCLUDED.notes <> ''
                        THEN EXCLUDED.notes
                        ELSE story_phase.notes
                    END,
                    last_advance_at = COALESCE(:la, story_phase.last_advance_at),
                    updated_at = NOW()
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "p": phase,
                "i": intensity,
                "n": notes,
                "la": datetime.now(timezone.utc) if agreed or advanced else last,
            },
        )
        try:
            await self._session.commit()
        except Exception:
            await self._session.rollback()

        if agreed or advanced:
            print(
                f"[STORY] ADVANCE user={user_id} -> {phase} intensity={intensity}",
                flush=True,
            )
        else:
            print(
                f"[STORY] phase={phase} intensity={intensity} next_hit={next_hit}",
                flush=True,
            )
        return await self.get(user_id, character_id)
