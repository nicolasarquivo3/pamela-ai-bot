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
        max_messages=20,
        max_memories=12,
        max_semantic_memories=6,
    ):
        self.session = session
        self.memory_manager = memory_manager
        self.emotion_engine = emotion_engine
        self.relationship_engine = relationship_engine
        self.semantic_memory_manager = semantic_memory_manager
        self.max_messages = max_messages
        self.max_memories = max_memories
        self.max_semantic_memories = max_semantic_memories

    async def record(self, user_id, character_id, role, content, metadata=None):
        row = ConversationMessage(
            user_id=user_id, character_id=character_id, role=role,
            content=content, metadata_json=metadata or {}
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
            semantic_rows = await semantic.search(
                user_id, character_id, semantic_query, limit=self.max_semantic_memories
            ) if semantic_query else []
        else:
            semantic_rows = []

        char_result = await self.session.execute(
            select(Character).where(Character.id == character_id)
        )
        character = char_result.scalar_one_or_none()

        emotion = None
        relationship = None
        if self.emotion_engine:
            state = await self.emotion_engine.get(user_id, character_id)
            emotion = {
                "valence": round(state.valence, 2),
                "arousal": round(state.arousal, 2),
                "affection": round(state.affection, 2),
                "trust": round(state.trust, 2),
                "loneliness": round(state.loneliness, 2),
                "frustration": round(state.frustration, 2),
                "curiosity": round(state.curiosity, 2),
                "style": self.emotion_engine.style(state),
            }
        if self.relationship_engine:
            rel = await self.relationship_engine.get(user_id, character_id)
            relationship = self.relationship_engine.guidance(rel)

        return {
            "character": {
                "name": character.name if character else "Lia",
                "image_identity": character.image_identity if character else {},
                "personality": character.personality_profile if character else {},
            },
            "memories": self.memory_manager.format_for_context(memories),
            "semantic_memories": semantic.format_for_context(semantic_rows) if semantic else [],
            "emotion": emotion,
            "relationship": relationship,
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
                for m in messages
            ],
        }
