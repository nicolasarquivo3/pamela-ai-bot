from datetime import date, datetime, timedelta, timezone
from sqlalchemy import select
from app.database.models import AutonomyState, User
from app.brain.context_manager import ContextManager
from app.brain.memory_manager import MemoryManager
from app.brain.semantic_memory import SemanticMemoryManager
from app.brain.emotion_engine import EmotionEngine
from app.brain.relationship_engine import RelationshipEngine
from app.repositories import UserRepository, CharacterRepository

class AutonomyService:
    def __init__(
        self,
        session_factory,
        telegram_bot,
        llm,
        memory_manager_factory,
        min_interval_minutes=90,
        max_daily_messages=3,
    ):
        self.session_factory = session_factory
        self.telegram_bot = telegram_bot
        self.llm = llm
        self.memory_manager_factory = memory_manager_factory
        self.min_interval_minutes = int(min_interval_minutes)
        self.max_daily_messages = int(max_daily_messages)

    async def tick(self):
        # A short-lived DB session keeps scheduled invocations isolated.
        async with self.session_factory() as session:
            users = await UserRepository(session).active_users()
            sent = 0
            waited = 0
            for user in users:
                try:
                    result = await self._process_user(session, user)
                    await session.commit()
                    if result["action"] == "message":
                        sent += 1
                    else:
                        waited += 1
                except Exception:
                    # One user's failure must not roll back work already
                    # committed for previous users in this tick.
                    await session.rollback()
                    waited += 1
            return {"sent": sent, "waited": waited}

    async def _process_user(self, session, user):
        character_id = user.character_id or 1
        state = await self._get_state(session, user.id, character_id)

        now = datetime.now(timezone.utc)

        # Lock the row for this transaction to avoid two overlapping ticks
        # both deciding to send.
        locked = await session.execute(
            select(AutonomyState)
            .where(AutonomyState.id == state.id)
            .with_for_update()
        )
        state = locked.scalar_one()
        self._reset_daily_counter(state, now)

        memory_manager, semantic_manager, context_manager = self.memory_manager_factory(
            session
        )
        context = await context_manager.build(
            user.id, character_id, query=None, semantic_manager=semantic_manager
        )

        from app.brain.decision_engine import DecisionEngine
        decision = DecisionEngine(
            self.min_interval_minutes, self.max_daily_messages
        ).decide(context, state, now)

        state.last_decision_at = now
        state.updated_at = now

        if decision["action"] != "message":
            return decision

        text = await self._compose_message(context, decision)
        if not text:
            return {"action": "wait", "reason": "llm_unavailable"}

        await self.telegram_bot.send_message(user.telegram_id, text)

        # Persist outbound messages in the same DB transaction.
        await context_manager.record(
            user.id,
            character_id,
            "assistant",
            text,
            metadata={"autonomous": True, "reason": decision.get("reason")},
        )
        state.last_outbound_at = now
        state.daily_messages += 1
        state.updated_at = now
        await session.flush()

        return {"action": "message", "reason": decision.get("reason")}

    async def _compose_message(self, context, decision):
        if not self.llm or not await self.llm.available():
            return None
        prompt = (
            "Você vai escrever UMA mensagem curta e natural para iniciar uma "
            "conversa espontânea com o usuário. Use somente informações presentes "
            "no contexto. A mensagem deve ter um motivo contextual real, nunca "
            "inventar eventos externos, e não deve cobrar resposta nem manipular "
            "emocionalmente. Não diga que você fez coisas fora do chat. "
            "Se houver uma memória relevante, use-a de modo natural. "
            f"Motivo interno: {decision.get('reason')}. "
            "Responda apenas com a mensagem, em pt-BR."
        )
        system = self._system_prompt(context)
        messages = context["messages"][-12:] + [{"role": "user", "content": prompt}]
        return await self.llm.generate(system, messages)

    def _system_prompt(self, context):
        character = context["character"]
        return (
            f"Você é {character.get('name', 'Lia')}, uma personagem virtual adulta. "
            f"Personalidade: {character.get('personality', {})}. "
            f"Memórias: {context.get('memories', [])}. "
            f"Memórias semânticas: {context.get('semantic_memories', [])}. "
            f"Estado emocional: {context.get('emotion')}. "
            f"Relacionamento: {context.get('relationship')}. "
            "Nunca alegue ser uma pessoa humana real, consciência real ou vida "
            "fora do sistema. Não invente fatos."
        )

    async def _get_state(self, session, user_id, character_id):
        result = await session.execute(
            select(AutonomyState).where(
                AutonomyState.user_id == user_id,
                AutonomyState.character_id == character_id,
            )
        )
        state = result.scalar_one_or_none()
        if not state:
            state = AutonomyState(
                user_id=user_id,
                character_id=character_id,
                enabled=True,
                daily_messages=0,
            )
            session.add(state)
            await session.flush()
        return state

    @staticmethod
    def _reset_daily_counter(state, now):
        if state.daily_date != now.date():
            state.daily_date = now.date()
            state.daily_messages = 0
