"""
Telegram bot.
- 1 mensagem do usuario = 1 processamento (lock por user + dedup)
- Foto SEM legenda; texto em mensagem separada
- channel_post: indexa fotos do ALBUM_CHANNEL_ID
- /album_drive_sync: indexa Google Drive
"""
from __future__ import annotations

import asyncio
from collections import deque

from aiogram import Bot, Dispatcher
from aiogram.types import Message, BufferedInputFile, Update
import httpx

from app.config import settings


class TelegramApp:

    def __init__(
        self,
        agent,
        album_service=None,
        drive_album_service=None,
        face_swap_service=None,
    ):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.agent = agent
        self.album_service = album_service
        self.drive_album_service = drive_album_service
        self.face_swap_service = face_swap_service

        self._user_locks: dict[int, asyncio.Lock] = {}
        self._seen_updates: deque[int] = deque(maxlen=800)
        self._seen_update_set: set[int] = set()
        self._seen_messages: deque[str] = deque(maxlen=800)
        self._seen_message_set: set[str] = set()

        @self.dp.channel_post()
        async def on_channel_post(message: Message):
            await self._handle_channel_post(message)

        @self.dp.message()
        async def on_message(message: Message):
            if not message.from_user:
                return

            user_id = message.from_user.id
            msg_key = f"{user_id}:{message.message_id}"

            if msg_key in self._seen_message_set:
                print(
                    f"[TelegramApp] skip duplicata message_id={message.message_id}",
                    flush=True,
                )
                return
            self._mark_message(msg_key)

            lock = self._user_locks.setdefault(user_id, asyncio.Lock())
            # prioridade chat: sinaliza DriveSync pra pausar tag
            try:
                from app.runtime_flags import telegram_enter, telegram_leave
            except Exception:
                try:
                    from app.brain.runtime_flags import telegram_enter, telegram_leave
                except Exception:
                    telegram_enter = telegram_leave = None  # type: ignore

            if telegram_enter:
                await telegram_enter()
            try:
                # nao fica preso pra sempre se algo travar
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=45.0)
                except asyncio.TimeoutError:
                    print(
                        f"[TelegramApp] lock timeout user={user_id} "
                        f"— processa mesmo assim",
                        flush=True,
                    )
                    try:
                        await self._handle_one(message)
                    finally:
                        if telegram_leave:
                            await telegram_leave()
                    return
                try:
                    await self._handle_one(message)
                finally:
                    lock.release()
            finally:
                if telegram_leave:
                    await telegram_leave()

    async def _send_text_bubbles(self, message, result: dict):
        """Envia 1..N mensagens de texto com pequena pausa."""
        texts = result.get("texts")
        if isinstance(texts, list) and len([t for t in texts if (t or "").strip()]) > 1:
            clean = []
            for part in texts[:5]:
                part = (part or "").strip().replace("|||", " ").strip()
                if part:
                    clean.append(part)
            for i, part in enumerate(clean):
                await message.answer(part)
                if i < len(clean) - 1:
                    await asyncio.sleep(1.15)
            return True

        reply = (result.get("text") or result.get("reply") or "").strip()
        if "|||" in reply:
            parts = [p.strip() for p in reply.split("|||") if p.strip()]
            if len(parts) > 1:
                for i, part in enumerate(parts[:5]):
                    await message.answer(part)
                    if i < len(parts) - 1:
                        await asyncio.sleep(1.15)
                return True
        if reply:
            await message.answer(reply)
            return True
        return False

    def _mark_message(self, key: str) -> None:
        if key in self._seen_message_set:
            return
        if len(self._seen_messages) >= 800:
            old = self._seen_messages.popleft()
            self._seen_message_set.discard(old)
        self._seen_messages.append(key)
        self._seen_message_set.add(key)

    def _mark_update(self, update_id: int | None) -> bool:
        """True se ja visto (skip)."""
        if update_id is None:
            return False
        if update_id in self._seen_update_set:
            return True
        if len(self._seen_updates) >= 800:
            old = self._seen_updates.popleft()
            self._seen_update_set.discard(old)
        self._seen_updates.append(update_id)
        self._seen_update_set.add(update_id)
        return False

    async def _handle_channel_post(self, message: Message) -> None:
        if not self.album_service:
            return
        chat_id = message.chat.id if message.chat else None
        if not self.album_service.is_album_channel(chat_id):
            return

        file_id = None
        fuid = None
        w = h = None

        if message.photo:
            photos = message.photo
            best = photos[-1]
            file_id = best.file_id
            fuid = best.file_unique_id
            w = best.width
            h = best.height
        elif message.document and (message.document.mime_type or "").startswith("image/"):
            file_id = message.document.file_id
            fuid = message.document.file_unique_id
        else:
            return

        gen_cap = bool(getattr(settings, "album_caption_on_ingest", False))
        ok = await self.album_service.ingest_telegram_photo(
            file_id=file_id,
            file_unique_id=fuid,
            caption_hint=(message.caption or ""),
            width=w,
            height=h,
            generate_caption=gen_cap,
        )
        n = await self.album_service.count()
        print(f"[ALBUM] channel_post ok={ok} total={n}", flush=True)

    async def _handle_one(self, message: Message) -> None:
        session = self.agent.context_manager.session
        try:
            text = (message.text or message.caption or "").strip()
            low = text.lower()

            # --- USER PHOTO: face swap com rosto da Pâmela e reenvia ---
            user_photo_file_id = None
            if message.photo:
                user_photo_file_id = message.photo[-1].file_id
            elif (
                message.document
                and (message.document.mime_type or "").startswith("image/")
            ):
                user_photo_file_id = message.document.file_id

            if user_photo_file_id and self.face_swap_service is not None:
                try:
                    await message.answer("Processando face swap... ⏳")
                    from app.images.models import ImageResult

                    tg_file = await self.bot.get_file(user_photo_file_id)
                    file_path = tg_file.file_path
                    # download bytes
                    bio = await self.bot.download_file(file_path)
                    if hasattr(bio, "read"):
                        target_bytes = bio.read()
                    elif hasattr(bio, "getvalue"):
                        target_bytes = bio.getvalue()
                    else:
                        target_bytes = bytes(bio) if bio else b""
                    if not target_bytes:
                        # fallback httpx
                        import httpx
                        url = (
                            f"https://api.telegram.org/file/bot"
                            f"{settings.telegram_bot_token}/{file_path}"
                        )
                        async with httpx.AsyncClient(timeout=60) as client:
                            r = await client.get(url)
                            r.raise_for_status()
                            target_bytes = r.content

                    print(
                        f"[USER-PHOTO] bytes={len(target_bytes)} -> face swap",
                        flush=True,
                    )
                    base = ImageResult(
                        success=True,
                        provider="user_upload",
                        image_bytes=target_bytes,
                    )
                    swapped = await self.face_swap_service.apply(base)
                    if swapped and swapped.success and (
                        swapped.image_bytes or swapped.image_url
                    ):
                        await self._send_image_result(
                            message,
                            {
                                "image_bytes": swapped.image_bytes,
                                "image_url": swapped.image_url,
                                "caption": None,
                            },
                        )
                        # se tinha legenda/texto, ainda responde em chat
                        if text:
                            result = await self.agent.receive_message(
                                message.from_user.id, text
                            )
                            await session.commit()
                            if isinstance(result, dict):
                                await self._send_text_bubbles(message, result)
                        else:
                            await session.commit()
                        return
                    err = getattr(swapped, "error", None) if swapped else "fail"
                    print(f"[USER-PHOTO] face swap falhou: {err}", flush=True)
                    await message.answer(
                        "Não consegui aplicar o face swap nessa foto agora ❤️ "
                        "Tenta outra (rosto bem visível, de preferência)."
                    )
                    await session.commit()
                    return
                except Exception as e:
                    print(f"[USER-PHOTO] error: {e}", flush=True)
                    try:
                        await message.answer(
                            "Deu erro no face swap dessa foto. Tenta de novo? ❤️"
                        )
                    except Exception:
                        pass
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    return
            elif user_photo_file_id and self.face_swap_service is None:
                await message.answer(
                    "Face swap não está ativo no servidor agora "
                    "(FACE_SWAP / reference image). "
                    "No modo texto eu só gero IMAGE_PROMPT nas mensagens ❤️"
                )
                await session.commit()
                return

            # --- comandos album ---
            if low in ("/album", "/album_stats"):
                lines = []
                if self.album_service:
                    n = await self.album_service.count()
                    lines.append(f"Canal Telegram: {n} foto(s)")
                if self.drive_album_service:
                    nd = await self.drive_album_service.count()
                    lines.append(f"Google Drive: {nd} foto(s)")
                if lines:
                    await message.answer("Album:\n" + "\n".join(lines))
                else:
                    await message.answer("Album desativado.")
                await session.commit()
                return

            if low.startswith("/album_tag"):
                if not self.album_service:
                    await message.answer("Album canal desativado.")
                    await session.commit()
                    return
                parts = low.split()
                lim = 20
                if len(parts) > 1 and parts[1].isdigit():
                    lim = min(int(parts[1]), 50)
                await message.answer(
                    f"Tagueando ate {lim} fotos do canal com IA... aguarde."
                )
                n = await self.album_service.backfill_captions(limit=lim)
                await message.answer(f"Canal: {n} foto(s) com tag automatica.")
                await session.commit()
                return


            # --- RESET NARRATIVA / MEMORIA ---
            if low.split()[0] in (
                "/reset",
                "/reset_memoria",
                "/reset_rp",
                "/recomecar",
                "/recomeçar",
            ) if text else False:
                try:
                    try:
                        from app.brain.memory_reset import MemoryResetService
                    except Exception:
                        from app.memory_reset import MemoryResetService
                except Exception:
                    MemoryResetService = None  # type: ignore

                try:
                    tg_id = int(message.from_user.id)
                    user = await self.agent.user_repository.get_or_create(tg_id)
                    uid = int(user.id)
                    char_id = int(getattr(user, "character_id", None) or 1)
                    parts = text.split()
                    hard = any(
                        p.lower() in ("hard", "tudo", "full") for p in parts[1:]
                    )

                    await message.answer("Resetando memoria e cena... ⏳")

                    stats = {}
                    if MemoryResetService is not None:
                        mrs = MemoryResetService(session)
                        stats = await mrs.reset_user_narrative(
                            uid, char_id, clear_long_term=hard
                        )
                    else:
                        from sqlalchemy import text as sa_text
                        for q in (
                            "DELETE FROM event_memories WHERE user_id=:u AND character_id=:c",
                            "DELETE FROM conversation_messages WHERE user_id=:u AND character_id=:c",
                        ):
                            try:
                                await session.execute(sa_text(q), {"u": uid, "c": char_id})
                            except Exception as e:
                                print(f"[RESET] sql {e}", flush=True)
                                try:
                                    await session.rollback()
                                except Exception:
                                    pass
                        try:
                            await session.execute(
                                sa_text(
                                    """
                                    INSERT INTO story_phase
                                      (user_id, character_id, phase, intensity, notes, updated_at)
                                    VALUES (:u,:c,'visual',0,'reset',NOW())
                                    ON CONFLICT (user_id, character_id) DO UPDATE SET
                                      phase='visual', intensity=0, notes='reset',
                                      last_advance_at=NULL, updated_at=NOW()
                                    """
                                ),
                                {"u": uid, "c": char_id},
                            )
                        except Exception as e:
                            print(f"[RESET] story: {e}", flush=True)

                    ltm = getattr(self.agent, "long_term_memory_service", None)
                    if ltm is not None:
                        try:
                            await ltm.seed_defaults(uid, char_id)
                            await ltm.set_current_scene(
                                uid,
                                char_id,
                                "NARRATIVA ATUAL (recomeço limpo): se arrumando para a balada, "
                                "bebendo e conversando como namorados. Fase visual — medo/receio, "
                                "ousa pra agradar; SEM sexo com outros como fato.",
                            )
                        except Exception as e:
                            print(f"[RESET] ltm: {e}", flush=True)

                    story = getattr(self.agent, "story_phase_service", None)
                    if story is not None:
                        try:
                            await story.get(uid, char_id)
                        except Exception:
                            pass

                    try:
                        if hasattr(self.agent, "_gemini_safety_strikes"):
                            self.agent._gemini_safety_strikes.pop(uid, None)
                    except Exception:
                        pass

                    await session.commit()
                    await message.answer(
                        "🔄 Reset concluído.\n"
                        f"Detalhes: {stats}\n"
                        "• Fase: visual (lenta)\n"
                        "• Cena: se arrumando pra balada\n\n"
                        "Manda a primeira mensagem ❤️\n"
                        "Hard: /reset_memoria hard"
                    )
                    print(
                        f"[RESET] ok user={uid} tg={tg_id} hard={hard} {stats}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[RESET] fail: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    await message.answer(f"Reset falhou: {e}")
                return

            if low in ("/cena_balada", "/reset_cena", "/preparar_balada"):
                try:
                    tg_id = int(message.from_user.id)
                    user = await self.agent.user_repository.get_or_create(tg_id)
                    uid = int(user.id)
                    char_id = int(getattr(user, "character_id", None) or 1)
                    ltm = getattr(self.agent, "long_term_memory_service", None)
                    story = getattr(self.agent, "story_phase_service", None)
                    scene = (
                        "NARRATIVA ATUAL: se arrumando para a balada, bebendo e "
                        "conversando como namorados. Ainda não na pista."
                    )
                    if ltm is not None:
                        await ltm.set_current_scene(uid, char_id, scene)
                        await ltm.upsert(
                            uid,
                            char_id,
                            "Fase da noite: PREPARAÇÃO.",
                            kind="fact",
                            key="cena_atual_fase",
                            importance=9,
                        )
                    if story is not None:
                        from sqlalchemy import text as sa_text
                        try:
                            await story.get(uid, char_id)
                            await session.execute(
                                sa_text(
                                    "UPDATE story_phase SET phase='visual', intensity=0, "
                                    "notes=:n, updated_at=NOW() "
                                    "WHERE user_id=:u AND character_id=:c"
                                ),
                                {"n": "cena: preparacao_balada", "u": uid, "c": char_id},
                            )
                        except Exception as e:
                            print(f"[CENA] story: {e}", flush=True)
                    await session.commit()
                    await message.answer(
                        "Cena: se arrumando pra balada ❤️ Manda a primeira msg."
                    )
                except Exception as e:
                    print(f"[CENA] fail: {e}", flush=True)
                    await message.answer(f"Cena falhou: {e}")
                return


            if low.startswith("/album_drive") or low.startswith("/drive"):
                if not self.drive_album_service:
                    await message.answer(
                        "Drive desligado.\n"
                        "Configure DRIVE_ALBUM_ENABLED + folder + service account."
                    )
                    await session.commit()
                    return
                parts = text.strip().split()
                if low in (
                    "/album_drive",
                    "/drive",
                    "/album_drive_stats",
                    "/drive_stats",
                ):
                    st = None
                    if hasattr(self.drive_album_service, "stats"):
                        try:
                            st = await self.drive_album_service.stats()
                        except Exception as e:
                            print(f"[DRIVE] stats: {e}", flush=True)
                    if st:
                        await message.answer(
                            "📁 Drive album\n"
                            f"• Indexadas: {st.get('total', 0)}\n"
                            f"• Com tag: {st.get('tagged', 0)} ({st.get('pct', 0)}%)\n"
                            f"• Sem tag: {st.get('untagged', 0)}"
                        )
                    else:
                        n = await self.drive_album_service.count()
                        await message.answer(f"Drive album: {n} foto(s).")
                    await session.commit()
                    return
                if "sync" in low:
                    lim = 100
                    caption_new = True
                    for p in parts[1:]:
                        pl = p.lower()
                        if p.isdigit():
                            lim = min(int(p), 500)
                        if pl in ("fast", "rapido", "rápido", "nocap"):
                            caption_new = False
                    await message.answer(
                        f"Sincronizando ate {lim} fotos "
                        f"({'tag' if caption_new else 'fast'})..."
                    )
                    res = await self.drive_album_service.sync(
                        limit=lim, caption_new=caption_new
                    )
                    await message.answer(
                        f"Drive sync: added={res.get('added')} "
                        f"total={res.get('total')} captioned={res.get('captioned')}"
                    )
                    await session.commit()
                    return
                if "tag" in low:
                    lim = 20
                    for p in parts[1:]:
                        if p.isdigit():
                            lim = min(int(p), 50)
                    await message.answer(f"Tagueando {lim}...")
                    n = await self.drive_album_service.backfill_captions(limit=lim)
                    await message.answer(f"Drive tags: {n}")
                    await session.commit()
                    return
                await message.answer(
                    "Comandos: /album_drive | /album_drive_sync 100 | "
                    "/album_drive_sync 300 fast | /album_drive_tag 30\n"
                    "Reset RP: /reset_memoria"
                )
                await session.commit()
                return


            print(
                f"[TelegramApp] handle user={message.from_user.id} "
                f"msg={message.message_id} text={text[:80]!r}",
                flush=True,
            )

            # feedback imediato: "digitando..."
            try:
                await message.bot.send_chat_action(
                    chat_id=message.chat.id, action="typing"
                )
            except Exception:
                pass

            result = await self.agent.receive_message(
                message.from_user.id,
                text,
            )
            await session.commit()

            if not isinstance(result, dict):
                return

            img = None
            if (
                result.get("type") == "image"
                or result.get("telegram_file_id")
                or result.get("bytes")
                or result.get("url")
            ):
                img = {
                    "success": result.get("success", True),
                    "image_url": result.get("url") or result.get("image_url"),
                    "image_bytes": result.get("bytes") or result.get("image_bytes"),
                    "telegram_file_id": result.get("telegram_file_id"),
                    "provider": result.get("provider"),
                    "caption": None,
                }
            elif result.get("image") or result.get("image_result"):
                raw = result.get("image") or result.get("image_result")
                if isinstance(raw, dict):
                    img = {
                        "success": raw.get("success", True),
                        "image_url": raw.get("url") or raw.get("image_url"),
                        "image_bytes": raw.get("bytes") or raw.get("image_bytes"),
                        "telegram_file_id": raw.get("telegram_file_id"),
                        "provider": raw.get("provider"),
                        "caption": None,
                    }
                else:
                    img = {
                        "success": getattr(raw, "success", False),
                        "image_url": getattr(raw, "image_url", None),
                        "image_bytes": getattr(raw, "image_bytes", None),
                        "telegram_file_id": getattr(raw, "telegram_file_id", None),
                        "provider": getattr(raw, "provider", None),
                        "caption": None,
                    }

            # Foto primeiro (se houver), depois bolhas de texto com pausa
            if img and (
                img.get("telegram_file_id")
                or img.get("image_bytes")
                or img.get("image_url")
            ):
                await self._send_image_result(message, img)

            await self._send_text_bubbles(message, result)



        except Exception as e:
            print(f"[TelegramApp] handle error: {e}", flush=True)
            try:
                await session.rollback()
            except Exception:
                pass
            try:
                await message.answer(
                    "Amor, deu um probleminha aqui agora. Tenta de novo? ❤️"
                )
            except Exception:
                pass

    async def _send_image_result(self, message: Message, result: dict) -> bool:
        if not result:
            return False
        # sem legenda nas fotos
        caption = None
        telegram_file_id = result.get("telegram_file_id")
        image_bytes = result.get("image_bytes") or result.get("bytes")
        image_url = result.get("image_url") or result.get("url")

        print(
            f"[TelegramApp] image payload: file_id="
            f"{'yes' if telegram_file_id else 'no'} "
            f"bytes={len(image_bytes) if image_bytes else 0} "
            f"url={'yes' if image_url else 'no'} caption=False",
            flush=True,
        )

        if telegram_file_id:
            try:
                await message.answer_photo(telegram_file_id, caption=caption)
                return True
            except Exception as e:
                print(f"[TelegramApp] send file_id failed: {e}", flush=True)

        try:
            if image_bytes:
                bio = BufferedInputFile(image_bytes, filename="pamela.jpg")
                await message.answer_photo(bio, caption=caption)
                return True
        except Exception as e:
            print(f"[TelegramApp] send bytes failed: {e}", flush=True)

        if image_url:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(image_url)
                if r.status_code == 200 and len(r.content) > 1000:
                    bio = BufferedInputFile(r.content, filename="pamela.jpg")
                    await message.answer_photo(bio, caption=caption)
                    return True
            except Exception as e:
                print(f"[TelegramApp] download/send url failed: {e}", flush=True)
                try:
                    await message.answer_photo(image_url, caption=caption)
                    return True
                except Exception as e2:
                    print(f"[TelegramApp] answer_photo url failed: {e2}", flush=True)

        print("[TelegramApp] sem dados de imagem utilizaveis", flush=True)
        return False

    async def feed_webhook_update(self, update):
        if isinstance(update, dict):
            uid = update.get("update_id")
        else:
            uid = getattr(update, "update_id", None)

        if self._mark_update(uid):
            print(f"[TelegramApp] skip update_id duplicado={uid}", flush=True)
            return

        await self.dp.feed_update(
            self.bot,
            Update.model_validate(update)
            if not isinstance(update, Update)
            else update,
        )

    async def set_webhook(self):
        if settings.webhook_base_url:
            url = settings.webhook_base_url.rstrip("/") + "/telegram/webhook"
            await self.bot.set_webhook(
                url,
                secret_token=settings.webhook_secret,
                drop_pending_updates=True,
                allowed_updates=[
                    "message",
                    "channel_post",
                    "edited_channel_post",
                    "callback_query",
                ],
            )
            print(
                f"[TelegramApp] webhook set url={url} "
                f"allowed_updates=message,channel_post",
                flush=True,
            )
