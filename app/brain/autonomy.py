"""
Autonomia proativa: texto e/ou foto.
Precisa de tick periodico (loop no main OU cron externo).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import random

from sqlalchemy import select

from app.database.models import AutonomyState, User
from app.repositories import UserRepository, CharacterRepository
from app.images.models import ImageRequest

try:
    from app.images.outfit import build_image_scene, get_current_outfit
except Exception:
    build_image_scene = None  # type: ignore
    get_current_outfit = None  # type: ignore


class AutonomyService:
    def __init__(
        self,
        session_factory,
        telegram_bot,
        llm,
        memory_manager_factory,
        image_service=None,
        min_interval_minutes=30,
        max_daily_messages=12,
        photo_chance=0.55,
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
        print(
            f"[Autonomy] tick start interval={self.min_interval_minutes}m "
            f"max_daily={self.max_daily_messages}",
            flush=True,
        )
        async with self.session_factory() as session:
            try:
                users = await UserRepository(session).active_users()
            except AttributeError:
                result = await session.execute(
                    select(User).where(User.active.is_(True))
                )
                users = list(result.scalars().all())
                print(
                    f"[Autonomy] fallback active_users={len(users)}",
                    flush=True,
                )

            sent = 0
            waited = 0
            for user in users:
                try:
                    result = await self._process_user(session, user)
                    await session.commit()
                    action = (result or {}).get("action")
                    reason = (result or {}).get("reason")
                    print(
                        f"[Autonomy] user={user.id} tg={user.telegram_id} "
                        f"action={action} reason={reason}",
                        flush=True,
                    )
                    if action in ("message", "photo"):
                        sent += 1
                    else:
                        waited += 1
                except Exception as e:
                    print(f"[Autonomy] erro user {user.id}: {e}", flush=True)
                    await session.rollback()
                    waited += 1

            out = {"sent": sent, "waited": waited, "users": len(users)}
            print(f"[Autonomy] tick done {out}", flush=True)
            return out

    async def _process_user(self, session, user):
        character_id = getattr(user, "character_id", None) or 1
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

        if decision.get("action") != "message":
            return decision

        want_photo = self.image_service is not None and random.random() < self.photo_chance

        if want_photo:
            result = await self._send_photo_message(
                session,
                user,
                character_id,
                context,
                decision,
                context_manager,
                state,
                now,
            )
            if result.get("action") == "photo":
                return result

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
            metadata={
                "autonomous": True,
                "reason": decision.get("reason"),
                "type": "text",
            },
        )
        state.last_outbound_at = now
        state.daily_messages = int(state.daily_messages or 0) + 1
        state.updated_at = now
        await session.flush()
        return {"action": "message", "reason": decision.get("reason")}

    async def _send_photo_message(
        self,
        session,
        user,
        character_id,
        context,
        decision,
        context_manager,
        state,
        now,
    ):
        if build_image_scene:
            scene = build_image_scene(
                "me manda uma foto pensando em voce",
                user_id=user.id,
                character_id=character_id,
            )
        else:
            outfit = "micro mini dress high heels"
            if get_current_outfit:
                outfit = get_current_outfit(user.id, character_id) or outfit
            scene = f"OUTFIT: {outfit} | PEDIDO: selfie espontanea pensando no usuario"

        caption = None
        try:
            raw = await self._compose_photo_caption(context, decision)
            if raw:
                caption, scene2 = self._parse_caption_scene(raw)
                if scene2 and "OUTFIT:" not in scene2:
                    scene = f"{scene} | acao: {scene2[:120]}"
        except Exception as e:
            print(f"[Autonomy] caption llm fail: {e}", flush=True)

        if not caption:
            caption = "Pensei em você agora... ❤️"

        try:
            img = await self.image_service.generate(
                ImageRequest(
                    user_id=user.id,
                    character_id=character_id,
                    scene=scene,
                )
            )
        except Exception as e:
            print(f"[Autonomy] image generate fail: {e}", flush=True)
            return {"action": "wait", "reason": "image_error"}

        if not img or not img.success:
            print(
                f"[Autonomy] image fail: {getattr(img, 'error', None)}",
                flush=True,
            )
            return {"action": "wait", "reason": "image_failed"}

        try:
            from aiogram.types import BufferedInputFile

            if img.image_bytes and len(img.image_bytes) > 100:
                photo = BufferedInputFile(img.image_bytes, filename="pamela.jpg")
                await self.telegram_bot.send_photo(
                    user.telegram_id, photo, caption=caption[:900]
                )
            elif img.image_url:
                await self.telegram_bot.send_photo(
                    user.telegram_id, img.image_url, caption=caption[:900]
                )
            else:
                return {"action": "wait", "reason": "no_image_data"}
        except Exception as e:
            print(f"[Autonomy] send_photo fail: {e}", flush=True)
            return {"action": "wait", "reason": "telegram_photo_error"}

        await context_manager.record(
            user.id,
            character_id,
            "assistant",
            f"[foto] {caption}",
            metadata={
                "autonomous": True,
                "reason": decision.get("reason"),
                "type": "image",
                "provider": getattr(img, "provider", None),
            },
        )
        state.last_outbound_at = now
        state.daily_messages = int(state.daily_messages or 0) + 1
        state.updated_at = now
        await session.flush()
        return {"action": "photo", "reason": decision.get("reason")}

    async def _compose_photo_caption(self, context, decision):
        if not self.llm or not await self.llm.available():
            return None
        prompt = (
            "Escreva UMA legenda curta (1-2 frases) de uma selfie espontanea "
            "que a personagem mandaria agora no Telegram, em pt-BR, carinhosa. "
            f"Motivo: {decision.get('reason')}. "
            "Formato opcional:\nCAPTION: ...\nSCENE: short english visual cue\n"
            "Responda so o texto util."
        )
        system = self._system_prompt(context)
        messages = context.get("messages", [])[-8:] + [
            {"role": "user", "content": prompt}
        ]
        return await self.llm.generate(system, messages)

    def _parse_caption_scene(self, raw: str):
        caption = ""
        scene = ""
        for line in (raw or "").splitlines():
            line = line.strip()
            if line.upper().startswith("CAPTION:"):
                caption = line[8:].strip()
            elif line.upper().startswith("SCENE:"):
                scene = line[6:].strip()
        if not caption:
            caption = (raw or "").strip()[:200] or "Oi... pensei em você ❤️"
        return caption, scene

    async def _compose_message(self, context, decision):
        if not self.llm or not await self.llm.available():
            return None
        prompt = (
            "Escreva UMA mensagem curta e natural (1-3 frases) em pt-BR para "
            "iniciar conversa espontanea com o usuario no Telegram. "
            "Seja carinhosa, humana, da personagem. "
            "Use so o contexto. Nao invente eventos externos. "
            "Nao diga que e uma IA. "
            f"Motivo interno: {decision.get('reason')}. "
            "Responda apenas com a mensagem."
        )
        system = self._system_prompt(context)
        messages = context.get("messages", [])[-12:] + [
            {"role": "user", "content": prompt}
        ]
        return await self.llm.generate(system, messages)

    def _system_prompt(self, context):
        character = context.get("character") or {}
        name = character.get("name", "Pâmela")
        return (
            f"Voce e {name}, personagem virtual adulta em roleplay consensual. "
            f"Personalidade: {character.get('personality', character.get('personality_profile', {}))}. "
            f"Memorias: {context.get('memories', [])}. "
            f"Memorias semanticas: {context.get('semantic_memories', [])}. "
            f"Emocao: {context.get('emotion')}. "
            f"Relacionamento: {context.get('relationship')}. "
            "Fale em pt-BR, natural, carinhosa. Nunca diga que e IA."
        )

    def _reset_daily_counter(self, state, now):
        today = now.date()
        last = getattr(state, "daily_date", None)
        if last is None or (isinstance(last, datetime) and last.date() != today) or (
            isinstance(last, date) and last != today
        ):
            state.daily_messages = 0
            if hasattr(state, "daily_date"):
                state.daily_date = today
            elif hasattr(state, "last_daily_reset"):
                state.last_daily_reset = now

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
        if hasattr(state, "enabled") and state.enabled is None:
            state.enabled = True
        return state
