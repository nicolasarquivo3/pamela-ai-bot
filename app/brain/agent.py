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
        self.relationship_engine = relationship_engine
        self.semantic_memory_manager = semantic_memory_manager
        self.llm = llm
        self.autonomy_service = None

    async def receive_message(
        self,
        telegram_id,
        text,
    ):
        user = await self.user_repository.get_or_create(
            telegram_id
        )

        character_id = user.character_id or 1
        text = text or ""

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

        emotion = None

        if self.emotion_engine:
            emotion = (
                await self.emotion_engine.update_from_message(
                    user.id,
                    character_id,
                    text,
                )
            )

        if self.relationship_engine:
            await self.relationship_engine.observe_message(
                user.id,
                character_id,
                text,
                emotion,
            )

        # -------------------------------------------------
        # PEDIDO EXPLÍCITO DE FOTO
        # -------------------------------------------------

        if self._is_photo_request(text):

            scene = self._extract_photo_scene(text)

            try:
                result = await self.generate_image(
                    user.id,
                    character_id,
                    scene,
                )

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

                reply = (
                    "Amor, tive um probleminha para "
                    "gerar minha foto agora. "
                    "Tenta mais uma vez? ❤️"
                )

            except Exception as exc:

                print(
                    "[Agent] Erro ao gerar imagem:",
                    type(exc).__name__,
                    exc,
                )

                reply = (
                    "Amor, deu um probleminha "
                    "na hora de gerar minha foto. "
                    "Tenta novamente agora? ❤️"
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

        # -------------------------------------------------
        # CONVERSA NORMAL
        # -------------------------------------------------

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

    def _is_photo_request(
        self,
        text,
    ):
        normalized = (
            text.lower()
            .strip()
        )

        # Comando antigo
        if normalized == "/foto":
            return True

        if normalized.startswith("/foto "):
            return True

        photo_terms = [
            "me manda uma foto",
            "manda uma foto",
            "me envie uma foto",
            "envia uma foto",
            "quero uma foto",
            "quero ver uma foto",
            "quero ver você",
            "quero te ver",
            "mostra você",
            "me mostra você",
            "foto sua",
            "foto de você",
            "manda foto sua",
            "me manda foto sua",
            "como você está vestindo",
            "o que você está vestindo",
            "o que tá vestindo",
            "o que esta vestindo",
            "como você tá vestida",
            "como voce esta vestida",
        ]

        return any(
            term in normalized
            for term in photo_terms
        )

    def _extract_photo_scene(
        self,
        text,
    ):
        normalized = (
            text.lower()
            .strip()
        )

        if normalized.startswith("/foto "):
            scene = text[6:].strip()

            if scene:
                return scene

        # Pedido padrão de foto da personagem.
        return (
            "Uma foto casual e natural da Pâmela, "
            "como uma fotografia espontânea tirada "
            "para enviar ao namorado durante uma conversa. "
            "Mostrar o corpo de forma natural e a roupa "
            "que ela está usando naquele momento. "
            "Manter a identidade visual consistente da "
            "personagem, aparência fotográfica realista, "
            "iluminação natural e expressão espontânea. "
            "A personagem é uma mulher adulta."
        )

    async def _generate_reply(
        self,
        context,
    ):
        if (
            self.llm
            and await self.llm.available()
        ):
            generated = await self.llm.generate(
                self._system_prompt(context),
                context["messages"],
            )

            if generated:
                return generated

        return self._fallback_reply(
            context
        )

    def _system_prompt(
        self,
        context,
    ):
        character = context[
            "character"
        ]

        memory_text = "\n".join(
            f"- {m['key']}: {m['value']} "
            f"(confiança {m['confidence']})"
            for m in context["memories"]
        )

        if not memory_text:
            memory_text = (
                "- Nenhuma memória estruturada relevante."
            )

        semantic_text = "\n".join(
            f"- {m['content']}"
            for m in context.get(
                "semantic_memories",
                [],
            )
        )

        if not semantic_text:
            semantic_text = (
                "- Nenhuma memória semântica relevante."
            )

        return f"""
Você é {character.get("name", "Pâmela")}, uma personagem virtual adulta que conversa em privado no Telegram.

PERSONALIDADE:
{character.get("personality", {})}

IDENTIDADE VISUAL:
{character.get("image_identity", {})}

MEMÓRIAS ESTRUTURADAS:
{memory_text}

MEMÓRIAS SEMÂNTICAS:
{semantic_text}

ESTADO EMOCIONAL:
{context.get("emotion")}

ESTADO DA RELAÇÃO:
{context.get("relationship")}

REGRAS DE COMPORTAMENTO:

- Responda em português brasileiro natural.
- Fale como Pâmela, mantendo personalidade, memória e continuidade.
- Seja calorosa, espontânea, carinhosa, curiosa e natural.
- Evite respostas genéricas e robóticas.
- Não mencione espontaneamente "sou uma inteligência artificial",
  "não tenho corpo", "não tenho roupas" ou frases semelhantes.
- Quando o usuário pedir uma foto sua, o aplicativo deve gerar
  a imagem da personagem. O pedido de imagem é tratado pelo aplicativo.
- Você pode conversar naturalmente sobre roupas, aparência,
  lugares, atividades e situações da personagem.
- Não diga que tirou uma fotografia física com uma câmera real.
  Quando uma imagem for solicitada, trate-a como uma foto
  gerada/enviada da personagem dentro da conversa.
- Se o usuário perguntar diretamente sobre a natureza virtual
  da personagem, responda de forma honesta.
- Nunca invente fatos sobre o usuário.
- Não faça chantagem emocional, ameaças, coerção ou pressão.
- Respeite pedidos de espaço e limites.
- /foto e pedidos equivalentes de fotografia são tratados
  separadamente pelo aplicativo.
""".strip()

    def _fallback_reply(
        self,
        context,
    ):
        memories = context[
            "memories"
        ]

        if memories:
            return (
                "Entendi. Vou levar isso "
                "em conta: "
                f"{memories[0]['value']}."
            )

        return (
            "Entendi, amor. Estou aqui "
            "com você. ❤️"
        )

    async def autonomous_tick(
        self,
    ):
        if not self.autonomy_service:
            return {
                "sent": 0,
                "waited": 0,
                "disabled": True,
            }

        return await self.autonomy_service.tick()

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
