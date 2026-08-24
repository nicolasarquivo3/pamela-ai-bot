from datetime import date, datetime, timezone
from enum import Enum
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.database.base import Base

class ImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    BOUNDARY = "boundary"
    PROFILE = "profile"

class Character(Base):
    __tablename__ = "characters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    image_identity: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    personality_profile: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    character_id: Mapped[int | None] = mapped_column(Integer)

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    character_id: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    character_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[MemoryKind] = mapped_column(SAEnum(MemoryKind, name="memory_kind"), nullable=False)
    key: Mapped[str] = mapped_column(String(160), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    source_message_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("conversation_messages.id"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    last_confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class EmotionState(Base):
    __tablename__ = "emotion_states"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    character_id: Mapped[int] = mapped_column(Integer, nullable=False)
    valence: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)
    arousal: Mapped[float] = mapped_column(Float, default=0.20, nullable=False)
    affection: Mapped[float] = mapped_column(Float, default=0.35, nullable=False)
    trust: Mapped[float] = mapped_column(Float, default=0.35, nullable=False)
    loneliness: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)
    frustration: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)
    curiosity: Mapped[float] = mapped_column(Float, default=0.35, nullable=False)
    last_trigger: Mapped[str | None] = mapped_column(String(160))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class RelationshipState(Base):
    __tablename__ = "relationship_states"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    character_id: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(40), default="new", nullable=False)
    closeness: Mapped[float] = mapped_column(Float, default=0.10, nullable=False)
    trust: Mapped[float] = mapped_column(Float, default=0.20, nullable=False)
    familiarity: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)
    reciprocity: Mapped[float] = mapped_column(Float, default=0.10, nullable=False)
    positive_interactions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_interactions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_interaction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

class ImageGeneration(Base):
    __tablename__ = "image_generations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    character_id: Mapped[int] = mapped_column(Integer, index=True)
    provider: Mapped[str | None] = mapped_column(String(50))
    job_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[ImageStatus] = mapped_column(SAEnum(ImageStatus, name="image_status"), nullable=False)
    scene: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text)
    negative_prompt: Mapped[str | None] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

class ImageUsage(Base):
    __tablename__ = "image_usage"
    __table_args__ = (UniqueConstraint("user_id", "provider", "period_date", name="uq_image_usage_user_provider_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    provider: Mapped[str] = mapped_column(String(50))
    period_date: Mapped[date] = mapped_column(Date, index=True)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class SemanticMemory(Base):
    __tablename__ = "semantic_memories"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    character_id: Mapped[int] = mapped_column(Integer, index=True)
    source_message_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversation_messages.id"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

class AutonomyState(Base):
    __tablename__ = "autonomy_states"
    __table_args__ = (
        UniqueConstraint("user_id", "character_id", name="uq_autonomy_user_character"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    character_id: Mapped[int] = mapped_column(Integer, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_outbound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quiet_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    daily_messages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    daily_date: Mapped[date | None] = mapped_column(Date)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
