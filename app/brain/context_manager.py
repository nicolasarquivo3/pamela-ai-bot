from sqlalchemy import select
from app.database.models import Character, ConversationMessage


class ContextManager:
    def __init__(
        self,
        session,
        memory_manager,
        emotion_engine=None,
        relationship_engine=None,
        semantic_memory_manager=None,
        event_memory_service=None,
        max_messages=40,
        max_memories=16,
        max_semantic_memories=10,
        max_event_memories=8,
    ):
        self.session = session
        self.memory_manager = memory_manager
        self.emotion_engine = emotion_engine
        self.relationship_engine = relationship_engine
        self.semantic_memory_manager = semantic_memory_manager
        self.event_memory_service = event_memory_service
        self.max_messages = max_messages
        self.max_memories = max_memories
        self.max_semantic_memories = max_semantic_memories
        self.max_event_memories = max_event_memories

    async def record(self, user_id, character_id, role, content, metadata=None):
        row = ConversationMessage(
            user_id=user_id,
            character_id=character_id,
            role=role,
            content=content,
            metadata_json=metadata or {},
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def build(self, user_id, character_id, query=None, semantic_manager=None):
        result = await self.session.execute(
            select(ConversationMessage)
            .where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.character_id == character_id,
            )
            .order_by(ConversationMessage.created_at.desc())
            .limit(self.max_messages)
        )
        messages = list(reversed(list(result.scalars())))
        memories = await self.memory_manager.recall(
            user_id, character_id, query=query, limit=self.max_memories
        )

        semantic = semantic_manager or self.semantic_memory_manager
        if semantic:
            semantic_query = query or (
                messages[-1].content if messages else ""
            )
            semantic_rows = (
                await semantic.search(
                    user_id,
                    character_id,
                    semantic_query,
                    limit=self.max_semantic_memories,
                )
                if semantic_query
                else []
            )
        else:
            semantic_rows = []

        event_rows = []
        if self.event_memory_service is not None:
            try:
                event_rows = await self.event_memory_service.recall_for_context(
                    user_id,
                    character_id,
                    query=query
                    or (messages[-1].content if messages else ""),
                    limit=self.max_event_memories,
                )
            except Exception as e:
                print(f"[EVENT-MEM] recall fail: {e}", flush=True)

        char_result = await self.session.execute(
            select(Character).where(Character.id == character_id)
        )
        character = char_result.scalar_one_or_none()

        emotion = None
        relationship = None
        if self.emotion_engine:
            state = await self.emotion_engine.get(user_id, character_id)
            emotion = state
        if self.relationship_engine:
            relationship = await self.relationship_engine.get(user_id, character_id)

        recent_conversation = "\n".join(
            f"{m.role}: {m.content}" for m in messages[-self.max_messages :]
        )

        return {
            "character": {
                "id": getattr(character, "id", character_id),
                "name": getattr(character, "name", "Pâmela") if character else "Pâmela",
                "personality_profile": getattr(character, "personality_profile", None)
                or getattr(character, "personality", {})
                if character
                else {},
                "image_identity": getattr(character, "image_identity", {})
                if character
                else {},
            }
            if character
            else {
                "id": character_id,
                "name": "Pâmela",
                "personality_profile": {},
                "image_identity": {},
            },
            "emotion": emotion,
            "relationship": relationship,
            "memories": self.memory_manager.format_for_context(memories),
            "semantic_memories": (
                semantic.format_for_context(semantic_rows) if semantic else []
            ),
            "event_memories": event_rows,
            "event_memories_text": (
                self.event_memory_service.format_for_prompt(event_rows)
                if self.event_memory_service
                else ""
            ),
            "recent_conversation": recent_conversation,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
