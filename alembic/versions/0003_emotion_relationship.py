from alembic import op
import sqlalchemy as sa

revision = "0003_emotion_relationship"
down_revision = "0002_brain_memory"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "emotion_states",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, unique=True, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False),
        sa.Column("valence", sa.Float(), nullable=False),
        sa.Column("arousal", sa.Float(), nullable=False),
        sa.Column("affection", sa.Float(), nullable=False),
        sa.Column("trust", sa.Float(), nullable=False),
        sa.Column("loneliness", sa.Float(), nullable=False),
        sa.Column("frustration", sa.Float(), nullable=False),
        sa.Column("curiosity", sa.Float(), nullable=False),
        sa.Column("last_trigger", sa.String(160)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "relationship_states",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False, unique=True, index=True),
        sa.Column("character_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(40), nullable=False),
        sa.Column("closeness", sa.Float(), nullable=False),
        sa.Column("trust", sa.Float(), nullable=False),
        sa.Column("familiarity", sa.Float(), nullable=False),
        sa.Column("reciprocity", sa.Float(), nullable=False),
        sa.Column("positive_interactions", sa.Integer(), nullable=False),
        sa.Column("total_interactions", sa.Integer(), nullable=False),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

def downgrade():
    op.drop_table("relationship_states")
    op.drop_table("emotion_states")
