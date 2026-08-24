from sqlalchemy import engine_from_config, pool
from alembic import context

from app.config import settings
from app.database.base import Base
from app.database import models


# Configuração do Alembic
config = context.config


# URL do banco de dados
database_url = settings.database_url

# Remove o driver assíncrono para o Alembic
database_url = database_url.replace("+asyncpg", "")

# Escapa "%" para o ConfigParser do Alembic
database_url = database_url.replace("%", "%%")

config.set_main_option(
    "sqlalchemy.url",
    database_url,
)


# Metadata usada pelo autogenerate
target_metadata = Base.metadata


def run_migrations_offline():
    """Executa as migrações em modo offline."""

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Executa as migrações em modo online."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
