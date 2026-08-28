"""
Reset de narrativa / memorias de conversa (nao apaga Drive/album).
"""
from __future__ import annotations

from sqlalchemy import text


class MemoryResetService:
    def __init__(self, session):
        self.session = session

    async def reset_user_narrative(
        self,
        user_id: int,
        character_id: int = 1,
        clear_long_term: bool = False,
        keep_core_ltm: bool = True,
    ) -> dict:
        """
        Apaga historico de chat + eventos + semantica + reseta story_phase.
        LTM: por padrao so remove keys de cena/eventos soltos; se clear_long_term,
        apaga tudo e deixa seed recriar.
        """
        stats = {
            "messages": 0,
            "events": 0,
            "semantic": 0,
            "story": 0,
            "ltm": 0,
            "autonomy": 0,
            "memories": 0,
        }

        # conversation messages — tenta nomes comuns
        for table, col_user, col_char in (
            ("conversation_messages", "user_id", "character_id"),
            ("messages", "user_id", "character_id"),
        ):
            try:
                r = await self.session.execute(
                    text(
                        f"DELETE FROM {table} WHERE {col_user}=:u AND {col_char}=:c"
                    ),
                    {"u": user_id, "c": character_id},
                )
                stats["messages"] += r.rowcount or 0
            except Exception:
                try:
                    await self.session.rollback()
                except Exception:
                    pass

        # event memories
        try:
            r = await self.session.execute(
                text(
                    "DELETE FROM event_memories WHERE user_id=:u AND character_id=:c"
                ),
                {"u": user_id, "c": character_id},
            )
            stats["events"] = r.rowcount or 0
        except Exception:
            try:
                await self.session.rollback()
            except Exception:
                pass

        # semantic memories
        for table in (
            "semantic_memories",
            "memory_embeddings",
            "semantic_memory",
        ):
            try:
                r = await self.session.execute(
                    text(
                        f"DELETE FROM {table} WHERE user_id=:u AND character_id=:c"
                    ),
                    {"u": user_id, "c": character_id},
                )
                stats["semantic"] += r.rowcount or 0
            except Exception:
                try:
                    await self.session.rollback()
                except Exception:
                    pass

        # structured memories / facts
        for table in ("memories", "character_memories", "user_memories"):
            try:
                r = await self.session.execute(
                    text(
                        f"DELETE FROM {table} WHERE user_id=:u AND character_id=:c"
                    ),
                    {"u": user_id, "c": character_id},
                )
                stats["memories"] += r.rowcount or 0
            except Exception:
                try:
                    await self.session.rollback()
                except Exception:
                    pass

        # story phase -> visual
        try:
            r = await self.session.execute(
                text(
                    """
                    INSERT INTO story_phase (user_id, character_id, phase, intensity, notes, updated_at)
                    VALUES (:u, :c, 'visual', 0, 'cena: preparacao_balada (reset)', NOW())
                    ON CONFLICT (user_id, character_id) DO UPDATE SET
                        phase='visual',
                        intensity=0,
                        notes='cena: preparacao_balada (reset)',
                        last_advance_at=NULL,
                        updated_at=NOW()
                    """
                ),
                {"u": user_id, "c": character_id},
            )
            stats["story"] = 1
        except Exception as e:
            print(f"[RESET] story_phase: {e}", flush=True)
            try:
                await self.session.rollback()
            except Exception:
                pass

        # long term
        try:
            if clear_long_term:
                r = await self.session.execute(
                    text(
                        "DELETE FROM long_term_memories WHERE user_id=:u AND character_id=:c"
                    ),
                    {"u": user_id, "c": character_id},
                )
                stats["ltm"] = r.rowcount or 0
            else:
                # remove only "scene" / spoiled narrative keys
                r = await self.session.execute(
                    text(
                        """
                        DELETE FROM long_term_memories
                        WHERE user_id=:u AND character_id=:c
                          AND (
                            key IN (
                              'cena_atual', 'cena_atual_fase', 'danca_colada',
                              'ciume_talk', 'sem_calcinha', 'sem_sutia'
                            )
                            OR kind = 'milestone'
                          )
                        """
                    ),
                    {"u": user_id, "c": character_id},
                )
                stats["ltm"] = r.rowcount or 0
        except Exception as e:
            print(f"[RESET] ltm: {e}", flush=True)
            try:
                await self.session.rollback()
            except Exception:
                pass

        # autonomy state (cooldown counters)
        try:
            r = await self.session.execute(
                text("DELETE FROM autonomy_states WHERE user_id=:u"),
                {"u": user_id},
            )
            stats["autonomy"] = r.rowcount or 0
        except Exception:
            try:
                await self.session.rollback()
            except Exception:
                pass
            try:
                r = await self.session.execute(
                    text("DELETE FROM autonomy_state WHERE user_id=:u"),
                    {"u": user_id},
                )
                stats["autonomy"] = r.rowcount or 0
            except Exception:
                try:
                    await self.session.rollback()
                except Exception:
                    pass

        try:
            await self.session.commit()
        except Exception as e:
            print(f"[RESET] commit: {e}", flush=True)
            try:
                await self.session.rollback()
            except Exception:
                pass

        return stats
