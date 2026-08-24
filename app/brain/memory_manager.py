from datetime import datetime, timezone
from sqlalchemy import select, update
from app.database.models import Memory, MemoryKind
from app.brain.deduplicator import Deduplicator, normalize

class MemoryManager:
    def __init__(self, session, extractor=None, deduplicator=None):
        self.session = session
        self.extractor = extractor
        self.deduplicator = deduplicator or Deduplicator()

    async def ingest_message(self, user_id, character_id, message_id, text):
        if not self.extractor:
            return []
        candidates = self.extractor.extract(text)
        stored = []
        for c in candidates:
            result = await self.session.execute(
                select(Memory).where(
                    Memory.user_id == user_id,
                    Memory.character_id == character_id,
                    Memory.kind == c.kind,
                    Memory.key == c.key,
                ).order_by(Memory.updated_at.desc()).limit(1)
            )
            existing = result.scalar_one_or_none()
            decision = self.deduplicator.decide(c, existing)
            if decision.action == "create":
                row = Memory(
                    user_id=user_id, character_id=character_id, kind=c.kind,
                    key=c.key, value=c.value, normalized_value=normalize(c.value),
                    importance=c.importance, confidence=c.confidence,
                    source_message_id=message_id,
                )
                self.session.add(row)
                stored.append(row)
            elif decision.action == "reinforce":
                existing.confidence = min(1.0, float(existing.confidence) * 0.8 + c.confidence * 0.2)
                existing.importance = max(float(existing.importance), c.importance)
                existing.last_confirmed_at = datetime.now(timezone.utc)
                existing.updated_at = datetime.now(timezone.utc)
                stored.append(existing)
            elif decision.action == "replace":
                existing.value = c.value
                existing.normalized_value = normalize(c.value)
                existing.confidence = c.confidence
                existing.importance = c.importance
                existing.source_message_id = message_id
                existing.last_confirmed_at = datetime.now(timezone.utc)
                existing.updated_at = datetime.now(timezone.utc)
                stored.append(existing)
        await self.session.flush()
        return stored

    async def recall(self, user_id, character_id, query=None, limit=12):
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.character_id == character_id)
        result = await self.session.execute(stmt.order_by(Memory.importance.desc(), Memory.updated_at.desc()).limit(limit))
        rows = list(result.scalars())
        if query:
            q = normalize(query)
            rows.sort(key=lambda m: (q in m.normalized_value, m.importance, m.updated_at.timestamp()), reverse=True)
        return rows

    def format_for_context(self, memories):
        return [
            {
                "kind": m.kind.value,
                "key": m.key,
                "value": m.value,
                "importance": round(float(m.importance), 2),
                "confidence": round(float(m.confidence), 2),
            }
            for m in memories
        ]
