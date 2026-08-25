from datetime import date, datetime, timedelta, timezone
import random
from sqlalchemy import select
from app.database.models import AutonomyState, User
from app.repositories import UserRepository, CharacterRepository
from app.images.models import ImageRequest


class AutonomyService:
    """
    Autonomia proativa.
    Agora pode mandar texto OU foto contextual (com face swap).
    Tudo 100% free usando o ImageService + FaceSwap que já existem.
    """

    def __init__(
        self,
        session_factory,
        telegram_bot,
        llm,
        memory_manager_factory,
        image_service=None,          # <-- novo
        min_interval_minutes=90,
        max_daily_messages=3,
        photo_chance=0.45,           # 45% das mensagens proativas são fotos
    ):
        self.session_factory = session_factory
        self.telegram_bot = telegram_bot
        self.llm = llm
        self.memory_manager_factory = memory_manager_factory
        self.image_service = image_service
        self.min_interval_minutes = int(min_interval_minutes)
        self.max_daily_messages = int(max_daily_messages)
        self.photo_chance = float(photo_chance)

    async def tick(self):
        async with self.session_factory() as session:
            users = await UserRepository(session).active_users()
            sent = 0
            waited = 0
            for user in users:
                try:
                    result = await self._process_user(session, user)
                    await session.commit()
                    if result.get("action") in ("message", "photo"):
                        sent += 1
                    else:
                        waited += 1
                except Exception as e:
                    print(f"[Autonomy] erro user {user.id}: {e}", flush=True)
                    await session.rollback()
                    waited += 1
            return {"sent": sent, "waited": waited}

    async def _process_user(self, session, user):
        character_id = user.character_id or 1
        state = await self._get_state(session, user.id, character_id)

        now = datetime.now(timezone.utc)

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

        # Decide se manda foto ou só texto
        want_photo = (
            self.image_service is not None
            and random.random() < self.photo_chance
        )

        if want_photo:
            result = await self._send_photo_message(
                session, user, character_id, context, decision, context_manager, state, now
            )
            if result.get("action") == "photo":
                return result
            # se falhou a foto, cai para texto

        # ---- TEXTO (fallback ou escolha) ----
        text = await self._compose_message(context, decision)
        if not text:
            return {"action": "wait", "reason": "llm_unavailable"}

        try:
            await self.telegram_bot.send_message(user.telegram_id, text)
        except Exception as e:
            print(f"[Autonomy] falha send_message: {e}", flush=True)
            return {"action": "wait", "reason": "telegram_error"}

        await context_manager.record(
            user.id,
            character_id,
            "assistant",
            text,
            metadata={"autonomous": True, "reason": decision.get("reason"), "type": "text"},
        )
        state.last_outbound_at = now
        state.daily_messages += 1
        state.updated_at = now
        await session.flush()

        return {"action": "message", "reason": decision.get("reason")}

    async def _send_photo_message(
        self, session, user, character_id, context, decision, context_manager, state, now
    ):
        """Gera cena contextual + caption, chama ImageService (já faz face swap) e envia."""
        try:
            composed = await self._compose_photo_content(context, decision)
            if not composed:
                return {"action": "wait", "reason": "llm_photo_failed"}

            caption = composed["caption"]
            scene = composed["scene"]

            # Gera a imagem (o ImageService já aplica face swap com a referência)
            image_result = await self.image_service.generate(
                ImageRequest(
                    user_id=user.id,
                    character_id=character_id,
                    scene=scene,
                )
            )

            if not image_result.success:
                print(f"[Autonomy] image fail: {image_result.error}", flush=True)
                return {"action": "wait", "reason": "image_generation_failed"}

            # Envia a foto
            if image_result.image_url:
                await self.telegram_bot.send_photo(
                    chat_id=user.telegram_id,
                    photo=image_result.image_url,
                    caption=caption,
                )
            elif image_result.image_bytes:
                from aiogram.types import BufferedInputFile
                photo = BufferedInputFile(
                    image_result.image_bytes,
                    filename="pamela_autonomous.jpg",
                )
                await self.telegram_bot.send_photo(
                    chat_id=user.telegram_id,
                    photo=photo,
                    caption=caption,
                )
            else:
                return {"action": "wait", "reason": "no_image_data"}

            # Persiste no histórico
            await context_manager.record(
                user.id,
                character_id,
                "assistant",
                f"[foto enviada] {caption}",
                metadata={
                    "autonomous": True,
                    "reason": decision.get("reason"),
                    "type": "photo",
                    "scene": scene,
                    "face_swapped": getattr(image_result, "face_swapped", True),
                },
            )
            state.last_outbound_at = now
            state.daily_messages += 1
            state.updated_at = now
            await session.flush()

            return {"action": "photo", "reason": decision.get("reason")}

        except Exception as e:
            print(f"[Autonomy] photo error: {e}", flush=True)
            return {"action": "wait", "reason": f"photo_exception:{e}"}

    async def _compose_photo_content(self, context, decision):
        """LLM retorna caption + scene em formato simples."""
        if not self.llm or not await self.llm.available():
            return None

        prompt = (
            "Você vai criar UMA mensagem espontânea com FOTO.\n"
            "Responda EXATAMENTE neste formato (sem mais nada):\n"
            "CAPTION: <mensagem curta e natural em pt-BR, 1-2 frases, como se estivesse mandando a foto>\n"
            "SCENE: <descrição visual curta da cena da foto, em inglês, photorealistic, ex: sitting on a couch smiling softly in soft evening light, wearing casual home clothes>\n\n"
            "Regras:\n"
            "- Use só informações do contexto (memórias, relacionamento, emoção).\n"
            "- Nunca invente eventos externos.\n"
            "- Não cobre resposta.\n"
            "- A SCENE deve descrever a personagem em uma situação natural e contextual.\n"
            f"Motivo interno: {decision.get('reason')}.\n"
            "Responda só com as duas linhas CAPTION: e SCENE:."
        )
        system = self._system_prompt(context)
        messages = context["messages"][-12:] + [{"role": "user", "content": prompt}]
        raw = await self.llm.generate(system, messages)
        if not raw:
            return None

        caption = ""
        scene = ""
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("CAPTION:"):
                caption = line[8:].strip()
            elif line.upper().startswith("SCENE:"):
                scene = line[6:].strip()

        if not caption or not scene:
            # fallback simples
            caption = raw.strip()[:200] if raw else "Oi... pensei em você ❤️"
            scene = "soft portrait of the woman looking gently at the camera, warm natural light, casual clothes"

        return {"caption": caption, "scene": scene}

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
            f"Você é {character.get('name', 'Pamela')}, uma personagem virtual adulta. "
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
