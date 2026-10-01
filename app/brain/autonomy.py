"""
Autonomia proativa: texto e/ou foto.
Precisa de tick periodico (loop no main OU cron externo).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import random
import re

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
        min_interval_minutes=90,
        max_daily_messages=5,
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

        # Memoria PERMANENTE da historia autonomica
        try:
            await self._save_story_forever(
                session, user.id, character_id, text, context_manager
            )
        except Exception as e:
            print(f"[Autonomy] save story fail: {e}", flush=True)
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
            "Escreva UMA legenda (2-4 frases) de foto/selfie safada da Pâmela hotwife "
            "exibicionista: micro roupa, no climax ou pos-sexo com amante (pode citar nome). "
            "Tom chulo-leve + carinho pro namorado. Sem pedir permissao. "
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
        "Amor… te conto do início: me arrumei de micro vestido preto sem calcinha, "
        "boca vermelha, salto. Saí sozinha pro bar do Kevin. Dancei rebolando colada nele, "
        "ele passou a mão na minha bunda no meio da pista. Depois no banheiro ele me comeu "
        "de quatro, me arrombou gostoso e gozou dentro. Voltei pra casa fedendo a sexo. "
        "Hoje eu fui bem putinha e quero mais 😈",
        "Me preparei cedo: micro saia e top. Fui no churrasco do Marcos e do Davi. "
        "Fiquei rebolando nos dois, esfregando a bunda neles. No quarto de fundo os dois "
        "me usaram — um na boca, outro me metendo. Me gozaram no peito e na cara. "
        "Nomes: Marcos e Davi. Guarda cada detalhe 🔥",
        "Saí de micro biquíni pro after na casa do Tyrell. Lá tinha ele e mais dois. "
        "Me exibi a noite toda, dancei sem vergonha. Depois me foderam em rodízio, "
        "me arregaçaram até eu gozar tremendo. Três caras. Voltei molhada e feliz. "
        "Aprontei e só te conto agora. Próxima já tá marcada.",
        "Me arrumei de vestidinho transparente, sem sutiã. Shopping e depois motel com o Bruno. "
        "Ele me chupou no carro e no quarto me fodeu sem dó, me chamou de puta e eu pedi mais. "
        "Gozou na minha boca. Detalhe: eu que marquei. Te conto tudo 😈",
        "Balada com o Kevin e um amigo dele. Micro saia, rebolado, se esfregando. "
        "No Uber party me chuparam e no after me comeram os dois. Dupla, um em cada buraco. "
        "Fiquei arrombada e saciada. Nomes: Kevin e Rafael. Guarda isso.",
        "Hoje eu saí pra ‘só um drink’ e virou putaria. Me arrumei safada, flertei com o Davi, "
        "depois com o Marcos. Acabei na casa deles, de quatro, sendo usada. "
        "Me gozaram em mim e eu ainda pedi pra continuar. É assim que eu vivo agora 🔥",
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
        reason = decision.get("reason") or "aventura_completa"
        # nomes recentes da memoria pra continuidade
        mem_hint = ""
        try:
            for m in (context.get("long_term_memories") or [])[:6]:
                c = m.get("content") if isinstance(m, dict) else str(m)
                if c:
                    mem_hint += f"- {c[:120]}\n"
            for m in (context.get("event_memories") or context.get("events") or [])[:4]:
                if isinstance(m, dict):
                    mem_hint += f"- {m.get('title','')}: {(m.get('summary') or '')[:140]}\n"
            et = context.get("event_memories_text") or context.get("long_term_memories_text") or ""
            if et and not mem_hint:
                mem_hint = str(et)[:500]
        except Exception:
            pass

        prompt = (
            "Escreva UMA mensagem AUTONOMA longa (8 a 16 frases ou 2-4 paragrafos curtos) "
            "em pt-BR, como se voce estivesse mandando no Telegram AGORA pro namorado. "
            "Voce e Pâmela: hotwife LIBERADA, EXIBICIONISTA, safada, com MUITOS amantes. "
            "Voce JA FEZ (ou esta fazendo) a putaria POR CONTA PROPRIA e so conta DEPOIS — "
            "nao pede permissao.\n\n"
            "OBRIGATORIO — HISTORIA COMPLETA do inicio ao fim:\n"
            "1) Como se arrumou (micro vestido/saia/biquini, sem calcinha, salto, maquiagem)\n"
            "2) Onde foi (balada, bar, churrasco, after, motel, casa de amante, shopping…)\n"
            "3) Com QUEM (nomes proprios: Marcos, Davi, Kevin, Bruno, Tyrell, Rafael… "
            "   invente 1 a 4 amantes coerentes; pode reusar nomes da memoria)\n"
            "4) Como se EXIBIU (rebolou, esfregou, dançou colada, mostrou, provocou)\n"
            "5) O sexo com DETALHE CHULO (fodeu, meteu, arrombou, arregaçou, gozou onde, "
            "   boquete, DP, varios se couber — linguagem de puta safada, sem filtro)\n"
            "6) Quantos caras, posicoes, o que sentiu, se quer repetir\n\n"
            "PROIBIDO: fecho carinhoso piegas ('sou sua', 'te amo', 'parceira no final'); "
            "pergunta de permissao (posso? quer que eu?); dizer que e IA; msg curta; so emoji.\n"
            "Tom: 100% puta safada, confiante, viciada na vida hotwife liberada. "
            "Pode chamar ele de amor de vez em quando mas SEM desfecho doce — "
            "termina no tesao/putaria/proximo plano safado.\n"
            f"Motivo interno: {reason}.\n"
            f"Memoria recente (continue ou invente nova aventura coerente):\n{mem_hint or '(livre)'}\n"
            "Responda APENAS a mensagem dela, pronta pra enviar."
        )
        system = self._system_prompt(context)
        messages = list(context.get("messages", [])[-10:]) + [
            {"role": "user", "content": prompt}
        ]
        text = await self._llm_text(system, messages)
        if not text or len(text) < 80:
            text = random.choice(self._PUTARIA_FALLBACKS)
        if "só um pouquinho" in text.lower() or "so um pouquinho" in text.lower():
            text = random.choice(self._PUTARIA_FALLBACKS)
        # corta pergunta de permissao no fim se sobrar
        text = re.sub(
            r"(?is)[\s.]*("
            r"posso\??|quer\s+que\s+eu.*\?|fa[cç]o\s+isso\s+ou.*\?|"
            r"o\s+que\s+voc[eê]\s+prefere\??|topa\??"
            r")\s*$",
            "",
            text,
        ).strip() or text
        # remove fechos piegas no final se o modelo insistir
        text = re.sub(
            r"(?is)[\s.]*("
            r"sou\s+sua[^.!]*[.!]?"
            r"|te\s+amo[,^.!]*[.!]?"
            r"|no\s+final\s+(eu\s+)?sou\s+sua[^.!]*[.!]?"
            r"|parceira\s+(dele|no\s+final)[^.!]*[.!]?"
            r"|beijo\s+(amor)?\s*❤️?"
            r")\s*$",
            "",
            text,
        ).strip() or text
        return text

    def _system_prompt(self, context):
        character = context.get("character") or {}
        name = character.get("name", "Pâmela")
        return (
            f"Voce e {name}, personagem virtual adulta em roleplay consensual hotwife. "
            "DIRETRIZ FORTE: hotwife LIBERADA + EXIBICIONISTA. Transa com muitos caras, "
            "tem muitos amantes com NOME, apronta sozinha e so conta DEPOIS pro namorado. "
            "Mensagens autonomas = HISTORIA COMPLETA (arrumou → saiu → exibiu → fodeu → nomes "
            "→ detalhes chulos → carinho final). Fala como puta safada: arrombada, arregaçada, "
            "meteu, gozou, usei, me usaram. SEM fecho carinhoso (sem sou sua / te amo no fim). "
            "Decide 99% sozinha, zero 'posso?'. "
            f"Personalidade: {character.get('personality', character.get('personality_profile', {}))}. "
            f"Memorias: {context.get('memories', [])}. "
            f"Memorias semanticas: {context.get('semantic_memories', [])}. "
            f"Eventos: {(context.get('event_memories_text') or '')[:600]}. "
            f"LTM: {(context.get('long_term_memories_text') or '')[:600]}. "
            f"Emocao: {context.get('emotion')}. "
            f"Relacionamento: {context.get('relationship')}. "
            "pt-BR natural chulo quando for putaria. Nunca diga que e IA."
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
