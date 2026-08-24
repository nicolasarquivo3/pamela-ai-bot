from alembic import op
import sqlalchemy as sa

revision = "0004_semantic_autonomy"
down_revision = "0003_emotion_relationship"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "semantic_memories",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False, index=True),
        sa.Column(
            "source_message_id",
            sa.BigInteger(),
            sa.ForeignKey("conversation_messages.id"),
            nullable=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute(
        "ALTER TABLE semantic_memories ALTER COLUMN embedding "
        "TYPE vector(384) USING NULL::vector"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_semantic_memories_embedding_hnsw "
        "ON semantic_memories USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "autonomy_states",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False, index=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_outbound_at", sa.DateTime(timezone=True)),
        sa.Column("last_decision_at", sa.DateTime(timezone=True)),
        sa.Column("quiet_until", sa.DateTime(timezone=True)),
        sa.Column("daily_messages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("daily_date", sa.Date()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "character_id", name="uq_autonomy_user_character"),
    )

def downgrade():
    op.drop_table("autonomy_states")
    op.execute("DROP INDEX IF EXISTS ix_semantic_memories_embedding_hnsw")
    op.drop_table("semantic_memories")
