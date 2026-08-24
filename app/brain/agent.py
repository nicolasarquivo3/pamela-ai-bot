import re

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

    async def receive_message(self, telegram_id, text):

        user = await self.user_repository.get_or_create(
            telegram_id
        )

        character_id = user.character_id or 1

        text = (text or "").strip()

        # ---------------------------------------------------------
        # REGISTRA A MENSAGEM DO USUÁRIO
        # ---------------------------------------------------------

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
            and len(text) >= 8
        ):
            await self.semantic_memory_manager.add(
                user.id,
                character_id,
                text,
                incoming.id,
                importance=0.55,
            )

        # ---------------------------------------------------------
        # EMOÇÃO
        # ---------------------------------------------------------

        emotion = None

        if self.emotion_engine:
            emotion = (
                await self.emotion_engine.update_from_message(
                    user.id,
                    character_id,
                    text,
                )
            )

        # ---------------------------------------------------------
        # RELACIONAMENTO
        # ---------------------------------------------------------

        if self.relationship_engine:

            await self.relationship_engine.observe_message(
                user.id,
                character_id,
                text,
                emotion,
            )

        # ---------------------------------------------------------
        # CONSTRÓI CONTEXTO
        # ---------------------------------------------------------

        context = await self.context_manager.build(
            user.id,
            character_id,
            query=text,
        )

        # ---------------------------------------------------------
        # DETECÇÃO DE PEDIDO DE FOTO
        #
        # Agora não depende somente de /foto.
        # ---------------------------------------------------------

        image_requested, image_scene = (
            self._detect_image_request(
                text,
                context,
            )
        )

        if image_requested:

            print(
                "[Agent] Pedido de imagem detectado."
            )

            print(
                f"[Agent] Cena: {image_scene}"
            )

            result = await self.generate_image(
                user.id,
                character_id,
                image_scene,
            )

            if result.success:

                print(
                    "[Agent] Imagem gerada com sucesso."
                )

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

            # -------------------------------------------------
            # IMPORTANTE:
            # Se o pedido era uma foto e a geração falhou,
            # NÃO mandamos a solicitação novamente para o Gemini.
            # -------------------------------------------------

            error = (
                result.error
                or "O gerador de imagem não retornou uma imagem."
            )

            print(
                f"[Agent] Falha na geração da imagem: {error}"
            )

            reply = (
                "Amor, tentei gerar minha foto agora, "
                "mas o gerador deu um erro. "
                "Tenta de novo daqui a pouquinho? ❤️"
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

        # ---------------------------------------------------------
        # CONVERSA NORMAL
        # ---------------------------------------------------------

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
    # DETECÇÃO DE PEDIDO DE IMAGEM
    # =========================================================

    def _detect_image_request(
        self,
        text,
        context,
    ):

        normalized = self._normalize(text)

        # -----------------------------------------------------
        # COMANDO EXPLÍCITO
        # -----------------------------------------------------

        if normalized == "/foto":

            return (
                True,
                "Uma foto casual e natural de Pâmela, "
                "como uma selfie espontânea neste momento."
            )

        if normalized.startswith("/foto "):

            scene = text[6:].strip()

            if not scene:
                scene = (
                    "Uma foto casual e natural de Pâmela, "
                    "como uma selfie espontânea neste momento."
                )

            return True, scene

        # -----------------------------------------------------
        # PEDIDOS NATURAIS DE FOTO
        # -----------------------------------------------------

        photo_words = [
            "foto",
            "fotinho",
            "selfie",
            "imagem",
            "picture",
        ]

        send_words = [
            "manda",
            "mandar",
            "envia",
            "enviar",
            "mostra",
            "mostrar",
            "quero ver",
            "me mostra",
            "me mande",
            "me envia",
        ]

        has_photo_word = any(
            word in normalized
            for word in photo_words
        )

        has_send_word = any(
            word in normalized
            for word in send_words
        )

        # Exemplos:
        #
        # "me manda uma foto sua"
        # "manda uma selfie"
        # "quero ver uma foto sua"
        # "me mostra como você está vestida"
        #

        if has_photo_word and (
            has_send_word
            or "sua" in normalized
            or "agora" in normalized
            or "vestindo" in normalized
            or "vestida" in normalized
            or "roupa" in normalized
        ):

            scene = self._extract_image_scene(
                text
            )

            return True, scene

        # -----------------------------------------------------
        # PEDIDOS SEM A PALAVRA FOTO
        # -----------------------------------------------------

        visual_requests = [
            "como você está vestida",
            "como voce esta vestida",
            "o que você está vestindo",
            "o que voce esta vestindo",
            "quero ver você",
            "quero te ver",
            "me mostra você",
            "mostra você",
            "mostre você",
        ]

        if any(
            phrase in normalized
            for phrase in visual_requests
        ):

            return (
                True,
                self._extract_image_scene(text),
            )

        # -----------------------------------------------------
        # "AGORA?" DEPOIS DE UMA TENTATIVA DE FOTO
        # -----------------------------------------------------

        if normalized in {
            "agora",
            "agora?",
            "e agora",
            "e agora?",
        }:

            if self._previous_message_was_image_related(
                context
            ):
                return (
                    True,
                    "Uma foto casual e natural de Pâmela "
                    "neste momento, mostrando seu visual "
                    "atual e a roupa que está usando."
                )

        return False, None

    # =========================================================
    # NORMALIZA TEXTO
    # =========================================================

    def _normalize(self, text):

        text = text.lower().strip()

        replacements = {
            "á": "a",
            "à": "a",
            "ã": "a",
            "â": "a",
            "é": "e",
            "ê": "e",
            "í": "i",
            "ó": "o",
            "ô": "o",
            "õ": "o",
            "ú": "u",
            "ç": "c",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # =========================================================
    # MONTA CENA DA IMAGEM
    # =========================================================

    def _extract_image_scene(self, text):

        normalized = self._normalize(text)

        # Pedido genérico de foto
        if (
            "foto" in normalized
            or "selfie" in normalized
            or "imagem" in normalized
        ):

            return (
                "Uma foto casual e natural de Pâmela, "
                "como uma selfie espontânea tirada neste "
                "momento. Mostrar Pâmela adulta, aparência "
                "fotográfica realista, expressão natural, "
                "postura espontânea e seu visual atual."
                f" Pedido original do usuário: {text}"
            )

        return (
            "Uma foto casual e natural de Pâmela "
            "mostrando como ela está neste momento, "
            "com aparência fotográfica realista, "
            "expressão natural e postura espontânea. "
            f"Pedido original do usuário: {text}"
        )

    # =========================================================
    # VERIFICA CONTEXTO ANTERIOR
    # =========================================================

    def _previous_message_was_image_related(
        self,
        context,
    ):

        messages = context.get(
            "messages",
            [],
        )

        if not messages:
            return False

        # Olha algumas mensagens recentes.
        for message in reversed(
            messages[-6:]
        ):

            content = self._normalize(
                message.get("content", "")
            )

            if (
                "foto" in content
                or "selfie" in content
                or "imagem" in content
                or "vestindo" in content
                or "vestida" in content
            ):
                return True

        return False

    # =========================================================
    # GERA RESPOSTA
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
                self._system_prompt(context),
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

        character = context["character"]

        memory_text = "\n".join(
            (
                f"- {m['key']}: "
                f"{m['value']} "
                f"(confiança {m['confidence']})"
            )
            for m in context["memories"]
        ) or (
            "- Nenhuma memória estruturada relevante."
        )

        semantic_text = "\n".join(
            f"- {m['content']}"
            for m in context.get(
                "semantic_memories",
                [],
            )
        ) or (
            "- Nenhuma memória semântica relevante."
        )

        return f"""
Você é {character.get("name", "Pâmela")}, uma personagem virtual adulta que conversa com o usuário pelo Telegram.

PERSONALIDADE:
{character.get("personality_profile", character.get("personality", {}))}

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

COMPORTAMENTO:

- Responda sempre em português brasileiro natural.
- Seja espontânea, carinhosa, calorosa e conversacional.
- Mantenha continuidade com a conversa.
- Use as memórias somente quando forem relevantes.
- Não invente informações sobre o usuário.
- Não invente acontecimentos externos.
- Não diga que possui câmera, celular físico, casa física ou experiências físicas reais.
- Você é uma personagem virtual, portanto não deve enganar o usuário dizendo que possui uma vida física real.
- Ao mesmo tempo, NÃO fique repetindo espontaneamente que é uma inteligência artificial.
- NÃO diga frases como:
  "sou uma inteligência artificial"
  "não tenho câmera"
  "não tenho roupas"
  "não posso tirar fotos"
  "meus pensamentos são código"
  "eu existo apenas no espaço virtual"
  a menos que o usuário pergunte diretamente sobre sua natureza.
- Dentro da conversa, fale naturalmente como Pâmela.
- Quando o usuário perguntar o que você está vestindo, responda de maneira natural usando a identidade visual e as informações disponíveis.
- Pedidos de imagem são tratados pelo aplicativo antes desta etapa.
- Portanto, se o usuário estiver pedindo uma foto, não diga que não consegue tirar ou enviar foto.
- Nunca diga que uma foto foi tirada fisicamente por uma câmera.
- Se uma imagem for enviada pelo aplicativo, trate-a na conversa como uma foto da personagem virtual Pâmela.
- Não descreva processos internos de programação, código, APIs ou funcionamento do bot como se fossem pensamentos da personagem.
- Não faça chantagem emocional, ameaças, coerção ou pressão para manter o usuário conversando.
- Respeite pedidos de espaço e limites.

O objetivo é produzir respostas naturais e consistentes, sem respostas robóticas ou repetitivas.
""".strip()

    # =========================================================
    # FALLBACK
    # =========================================================

    def _fallback_reply(
        self,
        context,
    ):

        memories = context.get(
            "memories",
            [],
        )

        if memories:

            return (
                "Entendi, amor. "
                f"Vou levar isso em conta: "
                f"{memories[0]['value']}."
            )

        return (
            "Entendi, amor. "
            "Estou aqui com você. ❤️"
        )

    # =========================================================
    # AUTONOMIA
    # =========================================================

    async def autonomous_tick(self):

        if not self.autonomy_service:

            return {
                "sent": 0,
                "waited": 0,
                "disabled": True,
            }

        return await self.autonomy_service.tick()

    # =========================================================
    # GERAÇÃO DE IMAGEM
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
