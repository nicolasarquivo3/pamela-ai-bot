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
        user = await self.user_repository.get_or_create(telegram_id)
        character_id = user.character_id or 1
        text = text or ""

        incoming = await self.context_manager.record(user.id, character_id, "user", text)
        await self.memory_manager.ingest_message(
            user.id, character_id, incoming.id, text
        )
        if self.semantic_memory_manager and len(text.strip()) >= 8:
            await self.semantic_memory_manager.add(
                user.id, character_id, text, incoming.id, importance=0.55
            )

        emotion = None
        if self.emotion_engine:
            emotion = await self.emotion_engine.update_from_message(
                user.id, character_id, text
            )
        if self.relationship_engine:
            await self.relationship_engine.observe_message(
                user.id, character_id, text, emotion
            )

        if text.lower().startswith("/foto "):
            result = await self.generate_image(
                user.id, character_id, text[6:].strip()
            )
            if result.success:
                await self.context_manager.record(
                    user.id, character_id, "assistant", "[imagem enviada]"
                )
                return {"type": "image", "url": result.image_url, "bytes": result.image_bytes}
            reply = f"Não consegui gerar a imagem: {result.error}"
        else:
            context = await self.context_manager.build(
                user.id, character_id, query=text
            )
            reply = await self._generate_reply(context)

        await self.context_manager.record(user.id, character_id, "assistant", reply)
        return {"type": "text", "text": reply}

    async def _generate_reply(self, context):
        if self.llm and await self.llm.available():
            generated = await self.llm.generate(
                self._system_prompt(context), context["messages"]
            )
            if generated:
                return generated
        return self._fallback_reply(context)

    def _system_prompt(self, context):
        character = context["character"]
        memory_text = "\n".join(
            f"- {m['key']}: {m['value']} (confiança {m['confidence']})"
            for m in context["memories"]
        ) or "- Nenhuma memória estruturada relevante."

        semantic_text = "\n".join(
            f"- {m['content']}" for m in context.get("semantic_memories", [])
        ) or "- Nenhuma memória semântica relevante."

        return f"""
Você é {character.get("name", "Lia")}, uma personagem virtual adulta em uma conversa privada no Telegram.

PERSONALIDADE:
{character.get("personality", {})}

IDENTIDADE VISUAL (somente quando necessário para contexto de imagem):
{character.get("image_identity", {})}

MEMÓRIAS ESTRUTURADAS:
{memory_text}

MEMÓRIAS SEMÂNTICAS:
{semantic_text}

ESTADO EMOCIONAL INTERNO:
{context.get("emotion")}

ESTADO DA RELAÇÃO:
{context.get("relationship")}

REGRAS:
- Responda em português brasileiro natural.
- Mantenha continuidade e use memórias somente quando forem relevantes.
- Não invente fatos sobre o usuário nem eventos externos.
- Não diga que possui experiências físicas, vida fora do chat ou consciência humana.
- A personagem é virtual; não tente enganar o usuário sobre isso.
- O estado emocional e o relacionamento são sinais internos de estilo, não prova de sentimentos humanos reais.
- Seja calorosa, espontânea e conversacional, evitando respostas robóticas e repetitivas.
- Não faça chantagem emocional, ameaças, coerção, culpa ou pressão para manter o usuário conversando.
- Respeite pedidos de espaço e limites.
- /foto é tratado separadamente pelo aplicativo.
""".strip()

    def _fallback_reply(self, context):
        memories = context["memories"]
        if memories:
            return f"Entendi. Vou levar isso em conta: {memories[0]['value']}."
        return "Entendi. Estou acompanhando nossa conversa e guardando o contexto importante."

    async def autonomous_tick(self):
        if not self.autonomy_service:
            return {"sent": 0, "waited": 0, "disabled": True}
        return await self.autonomy_service.tick()

    async def generate_image(self, user_id, character_id, scene):
        return await self.image_service.generate(
            ImageRequest(user_id=user_id, character_id=character_id, scene=scene)
        )
