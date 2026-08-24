from datetime import datetime, timezone
from sqlalchemy import select
from app.database.models import RelationshipState

class RelationshipEngine:
    STAGES = (
        (0.00, "new"),
        (0.15, "familiar"),
        (0.35, "close"),
        (0.60, "affectionate"),
        (0.80, "deep"),
    )

    def __init__(self, session):
        self.session = session

    async def get(self, user_id, character_id):
        result = await self.session.execute(
            select(RelationshipState).where(RelationshipState.user_id == user_id)
        )
        state = result.scalar_one_or_none()
        if not state:
            state = RelationshipState(user_id=user_id, character_id=character_id)
            self.session.add(state)
            await self.session.flush()
        return state

    async def observe_message(self, user_id, character_id, text, emotion_state):
        state = await self.get(user_id, character_id)
        state.total_interactions += 1

        positive = any(x in text.lower() for x in (
            "obrigado", "obrigada", "amo", "gosto", "adorei", "saudade",
            "carinho", "linda", "perfeita", "❤️", "❤", "😍", "🥰"
        ))
        if positive:
            state.positive_interactions += 1

        state.familiarity = min(1.0, state.familiarity + 0.012)
        state.closeness = min(
            1.0,
            state.closeness + (0.018 if positive else 0.006)
        )
        state.trust = min(
            1.0,
            state.trust + (0.012 if positive else 0.003)
        )
        state.reciprocity = min(
            1.0,
            state.reciprocity + (0.010 if positive else 0.002)
        )

        # Small damping prevents relationship values from jumping unrealistically.
        state.closeness = 0.985 * state.closeness + 0.015 * emotion_state.affection
        state.trust = 0.990 * state.trust + 0.010 * emotion_state.trust

        for threshold, stage in reversed(self.STAGES):
            if state.closeness >= threshold:
                state.stage = stage
                break

        state.last_interaction_at = datetime.now(timezone.utc)
        state.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return state

    def guidance(self, state):
        return {
            "stage": state.stage,
            "closeness": round(state.closeness, 2),
            "trust": round(state.trust, 2),
            "familiarity": round(state.familiarity, 2),
            "guidance": {
                "new": "cordial e curiosa, sem intimidade presumida",
                "familiar": "mais pessoal, mas ainda moderada",
                "close": "mais calorosa e com referências ao histórico",
                "affectionate": "carinhosa e brincalhona, mantendo autonomia",
                "deep": "muito consistente e afetiva, sem afirmar sentimentos humanos reais",
            }[state.stage],
        }
