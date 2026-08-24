from datetime import datetime, timezone
from sqlalchemy import select
from app.database.models import SemanticMemory
from app.brain.embedder import Embedder

class SemanticMemoryManager:
    def __init__(self, session, embedder=None):
        self.session = session
        self.embedder = embedder or Embedder()

    async def add(self, user_id, character_id, content, source_message_id=None, importance=0.5):
        content = " ".join((content or "").split())
        if not content:
            return None

        vector = self.embedder.embed(content)
        row = SemanticMemory(
            user_id=user_id,
            character_id=character_id,
            source_message_id=source_message_id,
            content=content,
            embedding=vector,
            importance=float(importance),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def search(self, user_id, character_id, query, limit=6):
        vector = self.embedder.embed(query)
        stmt = (
            select(SemanticMemory)
            .where(
                SemanticMemory.user_id == user_id,
                SemanticMemory.character_id == character_id,
                SemanticMemory.embedding.is_not(None),
            )
            .order_by(SemanticMemory.embedding.cosine_distance(vector))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars())

    def format_for_context(self, rows):
        return [
            {
                "id": row.id,
                "content": row.content,
                "importance": round(float(row.importance), 2),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
