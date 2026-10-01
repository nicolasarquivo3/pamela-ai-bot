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


    # Pools grandes — o modelo RECEBE nomes/lugar sorteados (obriga variedade)
    _NAME_POOL = (
        "Lucas", "Rodrigo", "Thiago", "Yuri", "André", "Caio", "Jaden", "Pedro",
        "Enzo", "Miguel", "Igor", "Kai", "Omar", "Noah", "Samuel", "Heitor",
        "Ben", "Theo", "Vicente", "Davi", "Rafael", "Diego", "Felipe", "Gustavo",
        "Leandro", "Murilo", "Otávio", "Pablo", "Renato", "Sérgio", "Tales",
        "Ubirajara", "Vitor", "Wagner", "Xander", "Yan", "Zeca", "Alex",
        "Breno", "Cauã", "Danilo", "Elias", "Fabrício", "Gael", "Hugo",
        "Isaac", "Jonas", "Kauan", "Luan", "Maicon", "Natan", "Patrick",
        "Quincy", "Ruan", "Silas", "Túlio", "Ulisses", "Valter", "Will",
        "Caleb", "Dante", "Evan", "Finn", "Greg", "Hassan", "Ibrahim",
        "Jamal", "Kelvin", "Lorenzo", "Malik", "Nico", "Oscar", "Phoenix",
    )
    _PLACE_POOL = (
        "rooftop bar em Moema",
        "after na casa de um DJ em Pinheiros",
        "sítio no interior com piscina",
        "cinema do shopping — sala vazia",
        "academia 24h depois da meia-noite",
        "festa de aniversário de uma amiga",
        "praia noturna com barraca",
        "Uber Black no caminho do aeroporto",
        "motel fora da cidade (só desta vez)",
        "camarim de um show",
        "cobertura com vista",
        "churrasco na laje de um amigo",
        "bar escondido de speakeasy",
        "sauna mista reservada",
        "quarto de hotel business",
        "vaga do estacionamento do shopping",
        "varanda de um apto alugado no Airbnb",
        "pista de um festival eletrônico",
        "barco parado no marina",
        "vestiário masculino da academia",
        "festa de empresa no rooftop",
        "casa de praia em Búzios",
        "clube de swing open night",
        "pub irlandês no centro (sem ser 'balada do centro')",
        "lounge de um cassino",
        "trailer party numa rua fechada",
        "studio de gravação depois do ensaio",
        "jacuzzi de um spa",
        "campo de futebol society depois do jogo",
        "biblioteca da facul vazia à noite",
    )
    _SEX_POOL = (
        "quickie em pé",
        "oral demorado + gozada na boca",
        "de quatro até arrombar",
        "ele sentado e ela rebolando em cima",
        "sexo no banco de trás",
        "menage 2 caras sem DP",
        "DP com dois",
        "rodízio de 3",
        "só amasso e boquete (sem penetração completa)",
        "creampie",
        "facial",
        "no chuveiro",
        "com ele filmando",
        "anal leve ocasional",
        "edge até ela implorar",
    )
    _LOOK_POOL = (
        "micro vestido preto colado sem calcinha",
        "micro saia plissada + top de tule",
        "body transparente e casaco aberto",
        "micro biquíni fio dental",
        "shortinho jeans rasgado e top mínimo",
        "vestido branco transparente na luz",
        "macacão aberto nas costas",
        "saia de couro curta e meia arrastão",
    )
    # nomes/lugares que o modelo viciou — ban permanente no texto
    _HARD_BAN = (
        "balada do centro",
        "balada no centro",
        "baladinha do centro",
        "festa do centro",
    )
    _HARD_BAN_NAMES = (
        "Marcos", "Kevin", "Tyrell",  # forçar rotação; podem voltar só se seed sortear
    )

    def _pick_story_seed(self, banned_names=None, banned_places=None):
        """Sorteia elenco/lugar/sexo OBRIGATÓRIOS pra esta mensagem."""
        banned_names = set(n.lower() for n in (banned_names or []))
        banned_places = [p.lower() for p in (banned_places or [])]
        names = [n for n in self._NAME_POOL if n.lower() not in banned_names]
        if len(names) < 5:
            names = list(self._NAME_POOL)
        n_partners = random.choice([1, 1, 1, 2, 2, 3])
        chosen = random.sample(names, k=min(n_partners, len(names)))
        places = list(self._PLACE_POOL)
        # evita lugares banidos por substring
        places = [
            p for p in places
            if not any(b in p.lower() for b in banned_places)
            and "balada do centro" not in p.lower()
        ] or list(self._PLACE_POOL)
        place = random.choice(places)
        sex = random.choice(self._SEX_POOL)
        look = random.choice(self._LOOK_POOL)
        return {
            "names": chosen,
            "n": len(chosen),
            "place": place,
            "sex": sex,
            "look": look,
        }

    def _extract_recent_bans(self, context) -> tuple[list, list]:
        """Nomes e lugares recentes pra banir nesta rodada."""
        blob_parts = []
        for key in (
            "event_memories_text",
            "long_term_memories_text",
            "messages",
        ):
            v = context.get(key)
            if isinstance(v, str):
                blob_parts.append(v)
            elif isinstance(v, list):
                for item in v[-12:]:
                    if isinstance(item, dict):
                        blob_parts.append(str(item.get("content") or item.get("summary") or item.get("text") or ""))
                    else:
                        blob_parts.append(str(item))
        for m in (context.get("long_term_memories") or [])[:10]:
            if isinstance(m, dict):
                blob_parts.append(str(m.get("content") or ""))
        for m in (context.get("event_memories") or context.get("events") or [])[:8]:
            if isinstance(m, dict):
                blob_parts.append(f"{m.get('title','')} {m.get('summary','')}")
        blob = " ".join(blob_parts)
        # nomes do pool que ja apareceram
        found_names = []
        low = blob.lower()
        for n in self._NAME_POOL:
            if n.lower() in low:
                found_names.append(n)
        for n in self._HARD_BAN_NAMES:
            if n.lower() in low and n not in found_names:
                found_names.append(n)
        # lugares-frase
        found_places = []
        for phrase in self._HARD_BAN:
            if phrase in low:
                found_places.append(phrase)
        # qualquer "balada ... centro"
        if re.search(r"balada.{0,20}centro|centro.{0,20}balada", low):
            found_places.append("balada do centro")
        return found_names, found_places

    def _story_violates_seed(self, text: str, seed: dict, banned_places: list) -> str | None:
        """Retorna motivo se o texto violou variedade; None se ok."""
        if not text:
            return "vazio"
        low = text.lower()
        for b in list(self._HARD_BAN) + list(banned_places or []):
            if b and b.lower() in low:
                return f"ban_place:{b}"
        # pelo menos 1 nome do seed deve aparecer
        if not any(n.lower() in low for n in seed.get("names") or []):
            return "sem_nome_seed"
        # se usou nomes hard-ban que NAO estao no seed
        for n in self._HARD_BAN_NAMES:
            if n.lower() in low and n not in seed.get("names", []):
                return f"nome_proibido:{n}"
        return None

    async def _compose_message(self, context, decision):
        if not self.llm or not await self.llm.available():
            return random.choice(self._PUTARIA_FALLBACKS)
        reason = decision.get("reason") or "aventura_completa"

        banned_names, banned_places = self._extract_recent_bans(context)
        # sempre banir o cliche
        for b in self._HARD_BAN:
            if b not in banned_places:
                banned_places.append(b)
        # banir elenco viciado se ja usou muito (sempre banir se aparecer em memoria)
        for n in self._HARD_BAN_NAMES:
            if n not in banned_names:
                banned_names.append(n)

        seed = self._pick_story_seed(banned_names, banned_places)
        names_s = ", ".join(seed["names"])
        print(
            f"[Autonomy] seed names={names_s!r} place={seed['place']!r} "
            f"sex={seed['sex']!r} ban_names={banned_names[:8]}",
            flush=True,
        )

        # memoria so como LISTA DE PROIBICOES — nao colar historias inteiras (modelo copia)
        mem_hint = (
            f"PROIBIDO repetir nomes: {', '.join(banned_names[:15]) or 'nenhum'}\n"
            f"PROIBIDO lugares/cliches: {', '.join(banned_places[:10]) or 'nenhum'}\n"
            "PROIBIDO: 'balada do centro', 'balada no centro', copiar relato anterior.\n"
            "NAO recontar a mesma aventura da memoria — invente cena 100% nova."
        )

        prompt = (
            "Voce e Pamela no Telegram. Hotwife liberada, exibicionista, puta safada. "
            "Aprontou sozinha e so conta DEPOIS. Mensagem LONGA (10-18 frases).\n\n"
            "=== ROTEIRO OBRIGATORIO DESTA MSG (NAO IGNORE) ===\n"
            f"- LOOK: {seed['look']}\n"
            f"- LUGAR (use este ou muito parecido, NAO troque por balada do centro): {seed['place']}\n"
            f"- AMANTE(S) — use EXATAMENTE estes nomes: {names_s}\n"
            f"- QTD: {seed['n']} parceiro(s)\n"
            f"- TIPO DE SEXO: {seed['sex']}\n"
            f"- Motivo interno: {reason}\n\n"
            "ESTRUTURA:\n"
            "1) Look detalhado + se arrumando\n"
            "2) METADE do texto = EXIBICIONISMO: danca, rebolado, esfrega, olhares, "
            "saia subindo, peito, colo, grind, amasso, beijo, mao boba\n"
            "3) Depois sexo chulo com os nomes do roteiro (arrombou, meteu, gozou...)\n"
            "4) Fecha no tesao / quero mais — SEM 'sou sua' / te amo\n\n"
            f"{mem_hint}\n\n"
            "Responda APENAS a mensagem dela em pt-BR."
        )
        system = (
            f"Voce e {((context.get('character') or {}).get('name') or 'Pamela')}, "
            "hotwife liberada puta safada. "
            "OBEDECA o roteiro (nomes+lugar+sexo). "
            "Nunca escreva 'balada do centro'. "
            "Nunca use Marcos/Kevin/Tyrell salvo se estiverem no roteiro. "
            "Exibicionismo longo antes do sexo. Zero 'posso?'. Zero IA. "
            "SEM fecho carinhoso."
        )
        messages = [{"role": "user", "content": prompt}]
        # nao misturar historico longo — evita copiar 'balada do centro' das msgs antigas
        # (opcional: 2 msgs so)
        hist = list(context.get("messages") or [])[-2:]
        if hist:
            messages = hist + messages

        text = await self._llm_text(system, messages)
        # se violou, tenta 1x de novo com seed novo
        reason_v = self._story_violates_seed(text or "", seed, banned_places)
        if reason_v:
            print(f"[Autonomy] seed violate ({reason_v}) — retry", flush=True)
            seed = self._pick_story_seed(banned_names, banned_places)
            names_s = ", ".join(seed["names"])
            prompt2 = (
                "REESCREVA do zero. Roteiro OBRIGATORIO:\n"
                f"LOOK={seed['look']} | LUGAR={seed['place']} | "
                f"NOMES={names_s} | SEXO={seed['sex']}\n"
                "Proibido: balada do centro, Marcos, Kevin, Tyrell (se nao estiverem nos nomes). "
                "Exibicionismo longo + sexo chulo. pt-BR. So a mensagem."
            )
            text2 = await self._llm_text(
                system,
                [{"role": "user", "content": prompt2}],
            )
            if text2 and len(text2) > 80:
                text = text2
                reason_v = self._story_violates_seed(text, seed, banned_places)
                if reason_v:
                    print(f"[Autonomy] still violate {reason_v} — fallback seed text", flush=True)
                    text = self._fallback_from_seed(seed)

        if not text or len(text) < 80:
            text = self._fallback_from_seed(seed)
        if "só um pouquinho" in text.lower() or "so um pouquinho" in text.lower():
            text = self._fallback_from_seed(seed)

        # hard replace cliche se escapar
        text = re.sub(
            r"(?i)baladinha?\s+(do|no)\s+centro|festa\s+do\s+centro",
            seed["place"],
            text,
        )
        text = re.sub(
            r"(?is)[\s.]*("
            r"posso\??|quer\s+que\s+eu.*\?|"
            r"sou\s+sua[^.!]*[.!]?|te\s+amo[,^.!]*[.!]?"
            r")\s*$",
            "",
            text,
        ).strip() or text
        return text

    def _fallback_from_seed(self, seed: dict) -> str:
        names = seed.get("names") or ["um cara"]
        n0 = names[0]
        extra = ""
        if len(names) > 1:
            extra = f" Também estava o {names[1]}."
        return (
            f"Me arrumei de {seed.get('look', 'micro vestido')}. "
            f"Fui em {seed.get('place', 'um after')}. "
            f"Dancei rebolando, saia subindo, me esfreguei no {n0}, "
            f"amasso com língua, mão na minha bunda, ele duro.{extra} "
            f"Depois {seed.get('sex', 'ele me fodeu sem dó')}. "
            f"Me arrombou e gozou. Nomes: {', '.join(names)}. "
            f"Quero repetir com gente nova."
        )


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
