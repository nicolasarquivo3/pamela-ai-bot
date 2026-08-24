from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_brain_memory"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade():
    kind = sa.Enum("fact","preference","relationship","event","boundary","profile", name="memory_kind")
    kind.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "memories",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False, index=True),
        sa.Column("kind", kind, nullable=False),
        sa.Column("key", sa.String(160), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("normalized_value", sa.Text(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source_message_id", sa.BigInteger(), sa.ForeignKey("conversation_messages.id")),
        sa.Column("embedding", sa.Text()),
        sa.Column("last_confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE vector(384) USING NULL::vector")

def downgrade():
    op.drop_table("memories")
    op.drop_table("conversation_messages")
    sa.Enum("fact","preference","relationship","event","boundary","profile", name="memory_kind").drop(op.get_bind(), checkfirst=True)
