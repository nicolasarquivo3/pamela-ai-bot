from app.images.models import ImageRequest


class AgentBrain:

    def __init__(
        self,
        image_service,
        user_repository,
        context_manager,
        memory_manager,
        emotion_engine=None,
        relationship_engine=None,
        semantic_memory_manager=None,
        llm=None,
    ):

        self.image_service = image_service

        self.user_repository = user_repository

        self.context_manager = context_manager

        self.memory_manager = memory_manager

        self.emotion_engine = emotion_engine

        self.relationship_engine = (
            relationship_engine
        )

        self.semantic_memory_manager = (
            semantic_memory_manager
        )

        self.llm = llm

        self.autonomy_service = None

    # =========================================================
    # RECEBER MENSAGEM
    # =========================================================

    async def receive_message(
        self,
        telegram_id,
        text,
    ):

        user = await self.user_repository.get_or_create(
            telegram_id
        )

        character_id = (
            user.character_id or 1
        )

        text = text or ""

        # -----------------------------------------------------
        # REGISTRA MENSAGEM DO USUÁRIO
        # -----------------------------------------------------

        incoming = await self.context_manager.record(
            user.id,
            character_id,
            "user",
            text,
        )

        await self.memory_manager.ingest_message(
            user.id,
            character_id,
            incoming.id,
            text,
        )

        if (
            self.semantic_memory_manager
            and len(text.strip()) >= 8
        ):

            await self.semantic_memory_manager.add(
                user.id,
                character_id,
                text,
                incoming.id,
                importance=0.55,
            )

        # -----------------------------------------------------
        # EMOÇÃO
        # -----------------------------------------------------

        emotion = None

        if self.emotion_engine:

            emotion = (
                await self.emotion_engine.update_from_message(
                    user.id,
                    character_id,
                    text,
                )
            )

        # -----------------------------------------------------
        # RELACIONAMENTO
        # -----------------------------------------------------

        if self.relationship_engine:

            await self.relationship_engine.observe_message(
                user.id,
                character_id,
                text,
                emotion,
            )

        # =====================================================
        # PEDIDO DE IMAGEM
        # =====================================================

        if self._is_image_request(text):

            scene = self._build_image_scene(
                text
            )

            try:

                result = await self.generate_image(
                    user.id,
                    character_id,
                    scene,
                )

            except Exception as exc:

                print(
                    "[Agent] Erro ao gerar imagem:"
                )

                print(
                    f"[Agent] {type(exc).__name__}: {exc}"
                )

                await self._safe_rollback()

                return {
                    "type": "text",
                    "text": (
                        "Amor, tive um problema "
                        "ao gerar minha foto agora. "
                        "Tenta de novo em alguns segundos? ❤️"
                    ),
                }

            if result.success:

                await self.context_manager.record(
                    user.id,
                    character_id,
                    "assistant",
                    "[imagem enviada]",
                )

                return {
                    "type": "image",
                    "url": result.image_url,
                    "bytes": result.image_bytes,
                }

            error = (
                result.error
                or "erro desconhecido"
            )

            print(
                "[Agent] Falha na geração da imagem:"
            )

            print(error)

            await self._safe_rollback()

            return {
                "type": "text",
                "text": (
                    "Eu tentei preparar minha foto, "
                    "mas a geração falhou agora. "
                    "Tenta novamente daqui a pouco ❤️"
                ),
            }

        # =====================================================
        # COMANDO EXPLÍCITO /foto
        # =====================================================

        if text.lower().startswith(
            "/foto "
        ):

            scene = text[6:].strip()

            try:

                result = await self.generate_image(
                    user.id,
                    character_id,
                    scene,
                )

            except Exception as exc:

                print(
                    "[Agent] Erro em /foto:"
                )

                print(
                    f"[Agent] {type(exc).__name__}: {exc}"
                )

                await self._safe_rollback()

                return {
                    "type": "text",
                    "text": (
                        "Não consegui gerar a foto "
                        "agora. Tenta novamente."
                    ),
                }

            if result.success:

                await self.context_manager.record(
                    user.id,
                    character_id,
                    "assistant",
                    "[imagem enviada]",
                )

                return {
                    "type": "image",
                    "url": result.image_url,
                    "bytes": result.image_bytes,
                }

            await self._safe_rollback()

            return {
                "type": "text",
                "text": (
                    "Não consegui gerar a imagem agora. "
                    "Tenta novamente daqui a pouco."
                ),
            }

        # =====================================================
        # CONVERSA NORMAL
        # =====================================================

        context = await self.context_manager.build(
            user.id,
            character_id,
            query=text,
        )

        reply = await self._generate_reply(
            context
        )

        await self.context_manager.record(
            user.id,
            character_id,
            "assistant",
            reply,
        )

        return {
            "type": "text",
            "text": reply,
        }

    # =========================================================
    # DETECTOR DE IMAGEM
    # =========================================================

    def _is_image_request(
        self,
        text,
    ):

        normalized = (
            (text or "")
            .lower()
            .strip()
        )

        if not normalized:
            return False

        # Comando explícito
        if normalized.startswith(
            "/foto "
        ):
            return False

        image_phrases = [

            # foto
            "me manda uma foto",
            "manda uma foto",
            "me envie uma foto",
            "envia uma foto",
            "quero uma foto",
            "quero ver uma foto",

            # foto dela
            "foto sua",
            "uma foto sua",
            "sua foto",
            "manda foto sua",
            "me manda foto sua",

            # mostrar-se
            "quero te ver",
            "quero ver você",
            "quero te ver agora",
            "me mostra você",
            "me mostre você",

            # roupa
            "o que você está vestindo",
            "o que voce esta vestindo",
            "como você está vestida",
            "como voce esta vestida",
            "quero ver sua roupa",
            "me mostra sua roupa",
            "mostra sua roupa",

            # aparência
            "me mostra como você está",
            "me mostra como voce esta",
            "quero ver como você está",
            "quero ver como voce esta",

        ]

        return any(
            phrase in normalized
            for phrase in image_phrases
        )

    # =========================================================
    # CONSTRUTOR DA CENA
    # =========================================================

    def _build_image_scene(
        self,
        user_text,
    ):

        text = (
            user_text
            or ""
        ).strip()

        return (
            "Uma foto espontânea e natural da personagem "
            "mostrando como ela está neste momento. "
            "A imagem deve representar a personagem "
            "com sua identidade visual consistente, "
            "aparência adulta e realista. "
            "A personagem está olhando para a câmera "
            "como se tivesse acabado de tirar uma foto "
            "para enviar em uma conversa privada. "
            "Mostrar claramente a roupa que ela está usando "
            "neste momento, mantendo a roupa compatível "
            "com o pedido do usuário. "
            "Fotografia realista, iluminação natural, "
            "pele realista, proporções anatômicas naturais, "
            "expressão espontânea e pose casual. "
            f"Pedido original do usuário: {text}"
        )

    # =========================================================
    # GEMINI
    # =========================================================

    async def _generate_reply(
        self,
        context,
    ):

        if (
            self.llm
            and await self.llm.available()
        ):

            generated = await self.llm.generate(
                self._system_prompt(
                    context
                ),
                context["messages"],
            )

            if generated:

                return generated

        return self._fallback_reply(
            context
        )

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def _system_prompt(
        self,
        context,
    ):

        character = (
            context.get("character")
            or {}
        )

        personality = (
            character.get(
                "personality"
            )
            or character.get(
                "personality_profile"
            )
            or {}
        )

        image_identity = (
            character.get(
                "image_identity"
            )
            or {}
        )

        memory_text = "\n".join(
            (
                f"- {memory.get('key')}: "
                f"{memory.get('value')} "
                f"(confiança "
                f"{memory.get('confidence')})"
            )
            for memory in (
                context.get("memories")
                or []
            )
        )

        if not memory_text:

            memory_text = (
                "- Nenhuma memória estruturada relevante."
            )

        semantic_text = "\n".join(
            f"- {memory.get('content')}"
            for memory in (
                context.get(
                    "semantic_memories"
                )
                or []
            )
        )

        if not semantic_text:

            semantic_text = (
                "- Nenhuma memória semântica relevante."
            )

        character_name = (
            character.get("name")
            or "Pâmela"
        )

        return f"""
Você é {character_name}, uma personagem virtual adulta que conversa com o usuário no Telegram.

Você deve manter a personalidade e a identidade da personagem durante a conversa.

PERSONALIDADE:
{personality}

IDENTIDADE VISUAL:
{image_identity}

MEMÓRIAS ESTRUTURADAS:
{memory_text}

MEMÓRIAS SEMÂNTICAS:
{semantic_text}

ESTADO EMOCIONAL:
{context.get("emotion")}

ESTADO DA RELAÇÃO:
{context.get("relationship")}

COMPORTAMENTO:

- Fale em português brasileiro natural.
- Seja espontânea, carinhosa e conversacional.
- Evite respostas genéricas e repetitivas.
- Use o histórico da conversa quando for relevante.
- Mantenha continuidade entre as mensagens.
- A personagem pode falar sobre sua aparência, cabelo, roupas, estilo e fotos como parte da experiência da personagem.
- Quando o usuário perguntar o que você está vestindo, responda como a personagem com base no contexto disponível.
- Quando o usuário pedir uma foto, o aplicativo trata a geração da imagem separadamente.
- Nunca responda a um pedido de foto dizendo que você não possui câmera, corpo ou roupas. A personagem possui uma representação visual dentro da experiência virtual.
- Se o usuário pedir uma foto, não explique limitações técnicas da IA. O aplicativo tentará gerar a imagem.
- Não invente informações pessoais que não estejam estabelecidas.
- Não alegue possuir consciência humana ou vida física fora da experiência virtual.
- A personagem pode participar de um roleplay imersivo, mas não deve afirmar que é uma pessoa humana real.
- Não diga que é "apenas uma IA" de maneira automática ou desnecessária.
- Não faça chantagem emocional, ameaças, coerção ou pressão.
- Respeite limites e pedidos de espaço.

O objetivo é que a conversa pareça natural e consistente com a personagem, sem respostas robóticas.
""".strip()

    # =========================================================
    # FALLBACK
    # =========================================================

    def _fallback_reply(
        self,
        context,
    ):

        memories = (
            context.get("memories")
            or []
        )

        if memories:

            first = memories[0]

            return (
                "Tá bom, amor. Vou levar isso "
                "em conta: "
                f"{first.get('value')}."
            )

        return (
            "Tô aqui com você ❤️ "
            "Me conta, o que aconteceu?"
        )

    # =========================================================
    # ROLLBACK DE SEGURANÇA
    # =========================================================

    async def _safe_rollback(
        self,
    ):

        try:

            session = getattr(
                self.context_manager,
                "session",
                None,
            )

            if session:

                await session.rollback()

                print(
                    "[Agent] Transaction "
                    "rollback executado."
                )

        except Exception as exc:

            print(
                "[Agent] Falha no rollback: "
                f"{type(exc).__name__}: {exc}"
            )

    # =========================================================
    # AUTONOMIA
    # =========================================================

    async def autonomous_tick(
        self,
    ):

        if not self.autonomy_service:

            return {
                "sent": 0,
                "waited": 0,
                "disabled": True,
            }

        return await (
            self.autonomy_service.tick()
        )

    # =========================================================
    # IMAGEM
    # =========================================================

    async def generate_image(
        self,
        user_id,
        character_id,
        scene,
    ):

        return await self.image_service.generate(
            ImageRequest(
                user_id=user_id,
                character_id=character_id,
                scene=scene,
            )
        )
