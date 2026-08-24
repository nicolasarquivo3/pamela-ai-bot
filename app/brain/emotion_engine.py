from datetime import datetime, timezone
from sqlalchemy import select
from app.database.models import EmotionState

def clamp(x):
    return max(0.0, min(1.0, float(x)))

class EmotionEngine:
    def __init__(self, session):
        self.session = session

    async def get(self, user_id, character_id):
        result = await self.session.execute(
            select(EmotionState).where(EmotionState.user_id == user_id)
        )
        state = result.scalar_one_or_none()
        if not state:
            state = EmotionState(user_id=user_id, character_id=character_id)
            self.session.add(state)
            await self.session.flush()
        return state

    async def update_from_message(self, user_id, character_id, text):
        state = await self.get(user_id, character_id)
        t = text.lower()

        # Lightweight affect detection. It does not diagnose the user.
        positive = any(x in t for x in (
            "obrigado", "obrigada", "amo", "gosto", "adorei", "legal",
            "feliz", "saudade", "carinho", "linda", "lindaaa", "perfeita",
            "bom dia", "boa noite", "❤️", "❤", "😍", "🥰"
        ))
        negative = any(x in t for x in (
            "bravo", "irritado", "irritada", "raiva", "odeio", "péssimo",
            "pessimo", "chato", "chateado", "decepcionado", "não gostei",
            "nao gostei"
        ))
        question = "?" in text
        personal = any(x in t for x in (
            "você", "voce", "nós", "nos", "nossa", "nosso", "meu", "minha"
        ))

        if positive:
            state.valence = clamp(state.valence + 0.08)
            state.affection = clamp(state.affection + 0.04)
            state.trust = clamp(state.trust + 0.025)
            state.frustration = clamp(state.frustration - 0.04)
            state.last_trigger = "positive_interaction"
        elif negative:
            state.valence = clamp(state.valence - 0.08)
            state.frustration = clamp(state.frustration + 0.08)
            state.trust = clamp(state.trust - 0.02)
            state.last_trigger = "negative_interaction"
        else:
            state.valence = clamp(state.valence * 0.98 + 0.02)
            state.frustration = clamp(state.frustration * 0.96)
            state.last_trigger = "neutral_interaction"

        if question:
            state.curiosity = clamp(state.curiosity + 0.04)
        if personal:
            state.affection = clamp(state.affection + 0.015)

        state.arousal = clamp(0.20 + abs(state.valence - 0.5) * 0.45)
        state.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return state

    def style(self, state):
        if state.frustration > 0.65:
            return "mais calma, curta e cuidadosa"
        if state.valence > 0.70 and state.affection > 0.60:
            return "calorosa, brincalhona e carinhosa"
        if state.valence < 0.30:
            return "mais reservada, gentil e acolhedora"
        if state.curiosity > 0.70:
            return "curiosa e envolvente"
        return "natural, leve e calorosa"
