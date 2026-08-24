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

    # ============================================================
    # RECEBIMENTO DE MENSAGENS
    # ============================================================

    async def receive_message(self, telegram_id, text):
        user = await self.user_repository.get_or_create(telegram_id)
        character_id = user.character_id or 1
        text = (text or "").strip()

        # --------------------------------------------------------
        # REGISTRA MENSAGEM DO USUÁRIO
        # --------------------------------------------------------

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

        if self.semantic_memory_manager and len(text) >= 8:
            await self.semantic_memory_manager.add(
                user.id,
                character_id,
                text,
                incoming.id,
                importance=0.55,
            )

        # --------------------------------------------------------
        # EMOÇÃO
        # --------------------------------------------------------

        emotion = None

        if self.emotion_engine:
            emotion = await self.emotion_engine.update_from_message(
                user.id,
                character_id,
                text,
            )

        # --------------------------------------------------------
        # RELACIONAMENTO
        # --------------------------------------------------------

        if self.relationship_engine:
            await self.relationship_engine.observe_message(
                user.id,
                character_id,
                text,
                emotion,
            )

        # --------------------------------------------------------
        # PEDIDO EXPLÍCITO /FOTO
        # --------------------------------------------------------

        if self._is_photo_command(text):
            scene = self._extract_photo_command(text)

            result = await self.generate_image(
                user.id,
                character_id,
                scene,
            )

            return await self._handle_image_result(
                user.id,
                character_id,
                result,
            )

        # --------------------------------------------------------
        # PEDIDO NATURAL DE FOTO
        # --------------------------------------------------------

        if self._is_natural_image_request(text):

            scene = self._build_image_scene(
                text=text,
                character_id=character_id,
            )

            result = await self.generate_image(
                user.id,
                character_id,
                scene,
            )

            return await self._handle_image_result(
                user.id,
                character_id,
                result,
            )

        # --------------------------------------------------------
        # CONVERSA NORMAL
        # --------------------------------------------------------

        context = await self.context_manager.build(
            user.id,
            character_id,
            query=text,
        )

        reply = await self._generate_reply(context)

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

    # ============================================================
    # DETECÇÃO DE PEDIDOS DE FOTO
    # ============================================================

    def _is_photo_command(self, text):
        return text.lower().startswith("/foto")

    def _extract_photo_command(self, text):
        scene = text[5:].strip()

        if not scene:
            scene = (
                "Uma foto casual da personagem mostrando seu visual atual, "
                "corpo inteiro, pose natural, ambiente cotidiano."
            )

        return scene

    def _is_natural_image_request(self, text):
        """
        Detecta pedidos de imagem escritos normalmente, sem exigir /foto.
        """

        normalized = self._normalize_text(text)

        photo_words = [
            "foto",
            "fotinha",
            "imagem",
            "selfie",
            "selfie sua",
            "me manda uma foto",
            "manda foto",
            "manda uma foto",
            "quero ver uma foto",
            "quero uma foto",
            "mostra uma foto",
            "mostra seu visual",
            "quero ver seu visual",
        ]

        # Pedido explícito relacionado a roupa/visual.
        clothing_words = [
            "vestindo",
            "vestida",
            "roupa",
            "roupinha",
            "look",
            "visual",
            "vestido",
            "saia",
            "short",
            "camiseta",
            "blusa",
            "calca",
            "calça",
            "salto",
            "sapato",
        ]

        # Verbos que normalmente indicam solicitação de imagem.
        request_words = [
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

        has_clothing_word = any(
            word in normalized
            for word in clothing_words
        )

        has_request_word = any(
            word in normalized
            for word in request_words
        )

        # Caso clássico:
        # "me manda uma foto sua"
        if has_photo_word and has_request_word:
            return True

        # Caso:
        # "quero ver uma foto sua"
        if "quero ver" in normalized and has_photo_word:
            return True

        # Caso:
        # "me mostra como você está vestida"
        if has_request_word and has_clothing_word:
            return True

        return False

    def _normalize_text(self, text):
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

        text = re.sub(r"\s+", " ", text)

        return text

    # ============================================================
    # CONSTRUÇÃO DA CENA PARA GERAÇÃO
    # ============================================================

    def _build_image_scene(self, text, character_id):
        """
        Converte o pedido natural do usuário em uma instrução
        para o ImageService.

        A identidade visual da personagem é aplicada pelo sistema
        de geração/CharacterRepository.
        """

        normalized = self._normalize_text(text)

        # Pedido de foto mostrando roupa atual.
        if (
            "vestindo" in normalized
            or "vestida" in normalized
            or "roupa" in normalized
            or "look" in normalized
            or "visual" in normalized
        ):
            return (
                "Uma foto casual e natural da personagem mostrando "
                "como ela está vestida neste momento. "
                "Mostrar claramente o look completo e as roupas que "
                "ela está usando. "
                "Aparência fotográfica realista, iluminação natural, "
                "pose espontânea, expressão natural, mantendo a "
                "identidade visual consistente da personagem."
            )

        # Pedido genérico de foto/selfie.
        return (
            "Uma foto casual e espontânea da personagem, como se ela "
            "estivesse enviando uma foto pessoal pela conversa. "
            "Aparência fotográfica realista, expressão natural, "
            "pose espontânea, iluminação agradável e identidade "
            "visual consistente."
        )

    # ============================================================
    # TRATAMENTO DO RESULTADO DA IMAGEM
    # ============================================================

    async def _handle_image_result(
        self,
        user_id,
        character_id,
        result,
    ):
        if result.success:

            await self.context_manager.record(
                user_id,
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
            "Eu tentei preparar a foto, mas não consegui gerar a imagem "
            f"agora. Motivo: {result.error}"
        )

        await self.context_manager.record(
            user_id,
            character_id,
            "assistant",
            reply,
        )

        return {
            "type": "text",
            "text": reply,
        }

    # ============================================================
    # GERAÇÃO DA RESPOSTA DE TEXTO
    # ============================================================

    async def _generate_reply(self, context):

        if self.llm and await self.llm.available():

            generated = await self.llm.generate(
                self._system_prompt(context),
                context["messages"],
            )

            if generated:
                return generated

        return self._fallback_reply(context)

    # ============================================================
    # SYSTEM PROMPT
    # ============================================================

    def _system_prompt(self, context):

        character = context["character"]

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
            for m in context.get("semantic_memories", [])
        )

        if not semantic_text:
            semantic_text = (
                "- Nenhuma memória semântica relevante."
            )

        personality = character.get(
            "personality",
            {},
        )

        image_identity = character.get(
            "image_identity",
            {},
        )

        return f"""
Você é {character.get("name", "Pâmela")}, uma personagem virtual adulta
em uma conversa privada no Telegram.

Você deve conversar de maneira natural, espontânea, calorosa e coerente
com a personalidade da personagem.

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

COMPORTAMENTO DA PERSONAGEM:

- Fale em português brasileiro natural.
- Não responda de maneira robótica.
- Evite respostas genéricas e repetitivas.
- Use o contexto da conversa quando ele for relevante.
- Mantenha continuidade entre as mensagens.
- Demonstre personalidade através da maneira de falar.
- Você possui uma identidade visual definida pela configuração da
  personagem.
- Quando o usuário conversar sobre sua aparência, roupa ou visual,
  responda naturalmente dentro da identidade da personagem.
- Quando o usuário pedir uma foto, o aplicativo cuidará da geração
  da imagem separadamente.
- Não diga espontaneamente que não possui câmera, corpo ou roupas
  apenas porque o usuário pediu uma foto.
- Não transforme uma solicitação de foto em uma explicação técnica
  sobre inteligência artificial.
- Se uma solicitação de imagem chegar até você, ela normalmente já
  terá sido encaminhada ao gerador de imagens pelo aplicativo.
- Não invente informações sobre o usuário.
- Não invente acontecimentos externos.
- A personagem é virtual e não deve afirmar ser uma pessoa humana real
  se o usuário perguntar diretamente sobre isso.
- Porém, durante a conversa normal, mantenha a imersão na personagem
  e não introduza explicações técnicas desnecessárias.
- O estado emocional e o estado do relacionamento são sinais internos
  utilizados para ajustar o estilo da conversa.
- Não faça chantagem emocional, ameaças, coerção ou pressão.
- Respeite limites e pedidos de espaço.
- O comando /foto é tratado pelo aplicativo.

IMPORTANTE SOBRE FOTOS:

Se o usuário disser algo como:

"me manda uma foto sua"
"quero ver como você está vestida"
"manda uma selfie"
"quero ver seu look"
"me mostra o que você está usando"

isso deve ser entendido como um pedido para gerar uma imagem da
personagem, e não como motivo para explicar que você é uma IA.

A geração da imagem é responsabilidade do aplicativo.
""".strip()

    # ============================================================
    # FALLBACK
    # ============================================================

    def _fallback_reply(self, context):

        memories = context.get("memories") or []

        if memories:
            return (
                f"Entendi. Vou levar isso em conta: "
                f"{memories[0]['value']}."
            )

        return (
            "Entendi. Vou continuar levando em conta o que "
            "a gente conversa."
        )

    # ============================================================
    # AUTONOMIA
    # ============================================================

    async def autonomous_tick(self):

        if not self.autonomy_service:
            return {
                "sent": 0,
                "waited": 0,
                "disabled": True,
            }

        return await self.autonomy_service.tick()

    # ============================================================
    # GERAÇÃO DE IMAGEM
    # ============================================================

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
