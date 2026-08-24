from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()

    # Cria o ENUM apenas uma vez.
    image_status_enum = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="image_status",
    )

    image_status_enum.create(bind, checkfirst=True)

    # Para usar o ENUM já criado dentro da tabela,
    # impedimos que o SQLAlchemy tente criá-lo novamente.
    image_status_column = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="image_status",
        create_type=False,
    )

    op.create_table(
        "characters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("image_identity", postgresql.JSONB(), nullable=False),
        sa.Column("personality_profile", postgresql.JSONB(), nullable=False),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("character_id", sa.Integer(), nullable=True),
    )

    op.create_table(
        "image_generations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("character_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=True),
        sa.Column("job_id", sa.String(255), nullable=True),
        sa.Column("status", image_status_column, nullable=False),
        sa.Column("scene", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
    )

    op.create_table(
        "image_usage",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("period_date", sa.Date(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "provider",
            "period_date",
            name="uq_image_usage_user_provider_date",
        ),
    )

    # Extensão pgvector.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Personagem inicial.
    op.execute(
        """
        INSERT INTO characters
        (
            id,
            name,
            image_identity,
            personality_profile
        )
        VALUES
        (
            1,
            'Lia',
            '{"hair":"long dark brown hair","eyes":"brown","face":"oval","skin":"natural"}'::jsonb,
            '{"tone":"warm","traits":["curious","playful","affectionate"]}'::jsonb
        )
        ON CONFLICT (id) DO NOTHING
        """
    )


def downgrade():
    op.drop_table("image_usage")
    op.drop_table("image_generations")
    op.drop_table("users")
    op.drop_table("characters")

    image_status_enum = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="image_status",
    )

    image_status_enum.drop(
        op.get_bind(),
        checkfirst=True,
    )
