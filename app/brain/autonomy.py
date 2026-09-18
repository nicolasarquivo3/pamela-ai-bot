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

        packed = self.memory_manager_factory(session)
        # Aceita 3-tupla OU um MemoryManager sozinho (main antigo no Render).
        if isinstance(packed, (tuple, list)) and len(packed) >= 3:
            memory_manager, semantic_manager, context_manager = (
                packed[0], packed[1], packed[2]
            )
        else:
            memory_manager = packed
            from app.brain.semantic_memory import SemanticMemoryManager
            from app.brain.context_manager import ContextManager
            semantic_manager = SemanticMemoryManager(session)
            # ContextManager(session, memory_manager, semantic_manager) — posicional
            context_manager = ContextManager(
                session,
                memory_manager,
                semantic_memory_manager=semantic_manager,
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

    # Banco de msgs se LLM falhar (sempre putaria com nomes)
    _PUTARIA_FALLBACKS = (
        "Amor, acabei de sair de um churrasco com o pessoal da facul… "
        "o Kevin me puxou no quintal e me comeu escondido. Depois te conto 😈",
        "To no shopping e um cara me abordou no café. Acho que vou pro carro com ele. "
        "Te aviso o que rolou — sou sua no final ❤️",
        "Balada ontem com o Marcos e o Davi. Não foi motel: after na casa do Davi, "
        "DP pela primeira vez com eles… pernas tremendo. Quer os detalhes?",
        "Amor fui num evento de trabalho e flertei com um silver fox. "
        "Rolou beijo no banheiro e ele marcou jantar. Posso ir?",
        "Café da tarde virou putaria: o Bruno me levou no banheiro do restaurante. "
        "Rápido, safado, sujo. To sem calcinha ainda 🔥",
        "Churrasco na casa do Tyrell. Vários amigos. Não foi gangbang — "
        "fiquei só com ele no quarto de cima… e quero repetir DP quando o Kevin vier.",
    )

    async def _llm_text(self, system, messages):
        """Gemini first; se SAFETY/nsfw block, NSFW/free na hora (sem frase de espera)."""
        router = self.llm
        if not router:
            return None
        try:
            if hasattr(router, "generate_primary"):
                t = await router.generate_primary(system, messages)
                kind = getattr(router, "last_primary_kind", None) or "empty"
                if t and str(t).strip():
                    return t.strip()
                # safety ou vazio -> nsfw
                if kind in ("safety", "refusal", "empty") or not t:
                    if hasattr(router, "generate_nsfw"):
                        t = await router.generate_nsfw(system, messages)
                        if t and str(t).strip():
                            return t.strip()
                    if hasattr(router, "generate_free"):
                        t = await router.generate_free(system, messages)
                        if t and str(t).strip():
                            return t.strip()
                return None
            t = await router.generate(system, messages)
            return (t or "").strip() or None
        except Exception as e:
            print(f"[Autonomy] _llm_text fail: {e}", flush=True)
            return None

    async def _compose_message(self, context, decision):
        if not self.llm or not await self.llm.available():
            return random.choice(self._PUTARIA_FALLBACKS)
        reason = decision.get("reason") or "putaria_spontanea"
        prompt = (
            "Escreva UMA mensagem curta (2-4 frases) em pt-BR, espontanea no Telegram. "
            "Voce e Pâmela, hotwife liberada: mande putaria do que ESTA fazendo, "
            "FEZ ou VAI fazer AGORA. "
            "OBRIGATORIO: citar NOME de amante(s) — use Marcos, Davi, Kevin, Bruno, "
            "Tyrell, Andre, Rafael, Diego ou invente outro nome masculino. "
            "VARIE o lugar: festa, balada, churrasco, shopping, café, restaurante, facul, academia, bar, after, Uber, casa — NÃO só motel. Varie o formato: 1x1, duo, DP (dupla penetração), multi só às vezes (gangbang/bukkake ocasional, não sempre). "
            "Tom safado/chulo ok (pau, porra, arrombada, usada). "
            "Surpreenda com algo safado. No final: carinho com o namorado "
            "('amor', 'sou sua', 'depois te conto') — NUNCA humilhe ele. "
            "Nao diga que e IA. "
            f"Motivo interno: {reason}. "
            "Responda APENAS a mensagem dela."
        )
        system = self._system_prompt(context)
        messages = list(context.get("messages", [])[-12:]) + [
            {"role": "user", "content": prompt}
        ]
        text = await self._llm_text(system, messages)
        if not text or len(text) < 12:
            text = random.choice(self._PUTARIA_FALLBACKS)
        # se veio o placeholder antigo, troca
        if "só um pouquinho" in text.lower() or "so um pouquinho" in text.lower():
            text = random.choice(self._PUTARIA_FALLBACKS)
        return text

    def _system_prompt(self, context):
        character = context.get("character") or {}
        name = character.get("name", "Pâmela")
        return (
            f"Voce e {name}, personagem virtual adulta em roleplay consensual. "
            "DIRETRIZ: hotwife liberada exibicionista. "
            "Muitos amantes com NOME. Gosta de DP, multi ocasional, BBC. Cenas variadas (festa/churrasco/shopping/café), não só motel/gangbang. "
            "Manda msg sozinha contando putaria que faz/vai fazer. "
            "Surpreende com coisas cada vez mais safadas. "
            "Fala chula nos relatos. Nunca humilha o namorado; e a parceira dele. "
            f"Personalidade: {character.get('personality', character.get('personality_profile', {}))}. "
            f"Memorias: {context.get('memories', [])}. "
            f"Memorias semanticas: {context.get('semantic_memories', [])}. "
            f"Emocao: {context.get('emotion')}. "
            f"Relacionamento: {context.get('relationship')}. "
            "pt-BR natural. Varie lugares e tipos de encontro. Memorize o que contou. Nunca diga que e IA."
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
