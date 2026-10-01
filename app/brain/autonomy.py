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
        "Me arrumei de micro saia plissada e top de tule, sem calcinha. No espelho rebolei "
        "ensaiando pra noite. Fui num rooftop bar — dancei sozinha no meio da pista, "
        "saia subindo, olhares de todo mundo. O barista loiro (Lucas) veio por trás, "
        "me esfreguei nele três músicas, ele duro. Amasso no corredor, mão na minha bunda, "
        "eu rebolando no pau dele por cima da calça. No banheiro do fundo ele me comeu "
        "apoiada na pia, me arrombou e gozou nas minhas costas. Nova conquista.",
        "Saí de body transparente e casaco aberto pro after do Rodrigo (trabalha com eventos). "
        "Lá me exibi no sofá: pernas abertas, rebolado no colo de um, depois no de outro. "
        "Beijei os dois na boca, me esfreguei, deixei passarem a mão no peito. "
        "No quarto do fundo o Rodrigo me chupou e o amigo (Thiago) me fodeu de quatro. "
        "Dois caras, nomes novos, eu bem puta.",
        "Micro biquíni vermelho na piscina do prédio do Yuri. Fiquei boiando de peito pra fora, "
        "depois dancei na borda rebolando pra ele e pro primo. Me esfreguei molhada nos dois, "
        "amasso molhado, mão por dentro do biquíni. No vestiário o Yuri me meteu em pé e "
        "o primo gozou na minha boca. Exibição → amasso → foda.",
        "Fui de vestido colado no clube de strip feminino open night. Dancei no pole de brincadeira, "
        "micro vestido subindo, seios quase saindo. Um casal me chamou — ele (André) e ela olhando. "
        "Rebolei no colo dele na frente dela, esfreguei a bunda, beijo com língua. "
        "No carro ele me comeu no banco de trás enquanto ela filmava. Bem safada.",
        "Churrasco no sítio do Caio. Shortinho e top. Dancei funk rebolando em cima da mesa, "
        "os caras gritando. Desci e me esfreguei no Caio e no amigo preto dele (Jaden). "
        "Amasso a três, mãos em mim. No quarto de hóspedes me foderam em rodízio, "
        "DP no final. Me gozaram na cara. Dois nomes novos, putaria completa.",
        "Shopping de mini dress branco. No banheiro do cinema me exibi pro segurança (Pedro) "
        "que me seguia com o olhar — levantei o vestido, mostrei que tava sem calcinha, "
        "dancei colada nele no corredor de serviço. Ele me chupou ali e me fodeu rápido "
        "contra a parede. Amador, safado, eu pedindo mais.",
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
        # memoria = o que EVITAR repetir + contexto
        mem_hint = ""
        try:
            chunks = []
            for m in (context.get("long_term_memories") or [])[:8]:
                c = m.get("content") if isinstance(m, dict) else str(m)
                if c:
                    chunks.append(c[:160])
            for m in (context.get("event_memories") or context.get("events") or [])[:6]:
                if isinstance(m, dict):
                    chunks.append(
                        f"{m.get('title', '')}: {(m.get('summary') or '')[:180]}"
                    )
            et = context.get("event_memories_text") or ""
            if et:
                chunks.append(str(et)[:400])
            lt = context.get("long_term_memories_text") or ""
            if lt:
                chunks.append(str(lt)[:300])
            blob = " ".join(chunks)
            used = sorted(
                set(
                    re.findall(
                        r"\b([A-ZÁÉÍÓÚÂÊÔÃÕ][a-záéíóúâêôãõ]{2,12})\b",
                        blob,
                    )
                )
            )
            ban = {
                "Amor", "Pamela", "Pâmela", "Depois", "Quando", "Hoje", "Ontem",
                "Entao", "Então", "Mas", "Com", "Para", "Essa", "Esse", "Uma",
                "Ele", "Ela", "Voce", "Você", "Telegram", "Muito", "Minha",
            }
            used = [n for n in used if n not in ban][:20]
            if used:
                mem_hint += (
                    "NOMES JA USADOS (evite repetir): " + ", ".join(used) + "\n"
                )
            for c in chunks[:8]:
                mem_hint += f"- {c}\n"
        except Exception:
            pass

        prompt = (
            "Escreva UMA mensagem AUTONOMA LONGA (10 a 18 frases / 3-5 paragrafos curtos) "
            "em pt-BR, Telegram, voce e Pamela: hotwife LIBERADA, EXIBICIONISTA, puta safada. "
            "Voce JA aprontou (ou esta aprontando) sozinha e so conta DEPOIS. Sem pedir permissao.\n\n"
            "VARIEDADE OBRIGATORIA (anti-repeticao):\n"
            "- INVENTA nomes novos de amantes (nao repita sempre Marcos/Davi/Kevin/Bruno/Tyrell). "
            "Use nomes variados: Lucas, Rodrigo, Thiago, Yuri, Andre, Caio, Jaden, Pedro, "
            "Rafael, Diego, Enzo, Miguel, Igor, Kai, Omar, Noah, etc. Ou invente outro. "
            "Se a memoria listar nomes recentes, EVITE os mesmos.\n"
            "- Quantidade de parceiros: as vezes 1, as vezes 2, as vezes 3+. Variar. "
            "Nao fazer sempre o mesmo roteiro.\n"
            "- Lugares DIFERENTES: rooftop, after, sitio, piscina, cinema, balada, bar, "
            "Uber, casa de amante, motel, academia, festa, praia, banheiro de shopping, "
            "camarim, hotel, churrasco, facul after. NUNCA so motel.\n"
            "- Tipos de sexo variados: oral, vaginal, anal ocasional, DP, rodizio, quickie, "
            "sexo no carro, menage, creampie, facial, etc.\n\n"
            "EXIBICIONISMO ANTES DO SEXO (ENFASE FORTE — metade ou mais do texto):\n"
            "1) Look detalhado (micro vestido/saia/biquini/body, sem calcinha?, salto)\n"
            "2) Como se EXIBIU: rebolou, dancou colada, saia subindo, peito aparecendo, "
            "olhares, se esfregou, sentou no colo, grind, provocou\n"
            "3) Amassos / beijos / maos bobas ANTES de foder\n"
            "4) So DEPOIS o sexo com detalhe chulo (arrombou, meteu, gozou onde, quantos)\n"
            "5) Termina no tesao/putaria/proximo plano — SEM fecho carinhoso "
            "(proibido: sou sua, te amo, parceira no final)\n\n"
            "Tom 100% puta safada hotwife liberada. Zero 'posso?'. Zero IA.\n"
            f"Motivo interno: {reason}.\n"
            f"Memoria recente (NAO repetir nomes/lugares/roteiro; invente NOVO):\n"
            f"{mem_hint or '(livre — invente tudo novo)'}\n"
            "Responda APENAS a mensagem dela."
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
        text = re.sub(
            r"(?is)[\s.]*("
            r"posso\??|quer\s+que\s+eu.*\?|fa[cç]o\s+isso\s+ou.*\?|"
            r"o\s+que\s+voc[eê]\s+prefere\??|topa\??"
            r")\s*$",
            "",
            text,
        ).strip() or text
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
            f"Voce e {name}, personagem virtual adulta hotwife liberada. "
            "VARIEDADE: invente amantes com nomes diferentes a cada relato; "
            "nao repita o mesmo elenco. Lugares e tipos de sexo variam. "
            "EXIBICIONISMO forte ANTES do sexo: danca, rebolado, esfrega, amasso, look micro. "
            "Depois putaria chula. SEM fecho carinhoso. Apronta sozinha e so conta depois. "
            "Zero 'posso?'. "
            f"Personalidade: {character.get('personality', character.get('personality_profile', {}))}. "
            f"Memorias: {context.get('memories', [])}. "
            f"Memorias semanticas: {context.get('semantic_memories', [])}. "
            f"Eventos recentes (NAO copiar): {(context.get('event_memories_text') or '')[:500]}. "
            f"LTM: {(context.get('long_term_memories_text') or '')[:400]}. "
            f"Emocao: {context.get('emotion')}. "
            f"Relacionamento: {context.get('relationship')}. "
            "pt-BR chulo na putaria. Nunca diga que e IA."
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
