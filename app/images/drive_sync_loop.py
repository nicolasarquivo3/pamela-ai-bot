"""
Loop Drive: index + tag automatico.

- Chat tem prioridade, mas tag NAO fica eternamente adiada.
- Tag roda mesmo com muitas fotos novas (senão nunca tagueia pastas grandes).
"""
from __future__ import annotations

import asyncio
import gc
import time


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 600,
        batch: int = 40,
        tag_batch: int = 8,
        enabled: bool = True,
    ):
        self.drive = drive_album_service
        self.interval = max(120, int(interval_seconds))
        self.batch = max(1, min(int(batch), 100))
        self.tag_batch = max(0, min(int(tag_batch), 20))
        self.enabled = bool(enabled) and drive_album_service is not None
        self._task = None
        self._last_tag_at = 0.0
        self._skip_busy_streak = 0

    async def start(self):
        if not self.enabled:
            print("[DriveSync] desligado", flush=True)
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="drive-sync")
        print(
            f"[DriveSync] iniciado interval={self.interval}s "
            f"index_batch={self.batch} tag_batch={self.tag_batch} "
            f"(tag sempre tenta se houver untagged)",
            flush=True,
        )

    async def _run(self):
        await asyncio.sleep(45)
        while True:
            try:
                await self._one_cycle()
            except Exception as e:
                print(f"[DriveSync] erro: {e}", flush=True)
                import traceback
                traceback.print_exc()
            try:
                gc.collect()
            except Exception:
                pass
            left = self.interval
            while left > 0:
                await asyncio.sleep(min(20, left))
                left -= 20

    async def _chat_busy_strict(self) -> bool:
        """So bloqueia se mensagem AGORA em processamento."""
        try:
            from app.runtime_flags import telegram_busy
            return bool(telegram_busy())
        except Exception:
            return False

    async def _chat_recent(self) -> bool:
        try:
            from app.runtime_flags import recently_active
            return bool(recently_active(25))  # 25s so — nao 90
        except Exception:
            return False

    async def _one_cycle(self):
        if not self.drive or not await self.drive.available():
            print("[DriveSync] drive indisponivel", flush=True)
            return

        # caption_fn?
        cfn = getattr(self.drive, "_caption_fn", None)
        print(
            f"[DriveSync] ciclo start caption_fn={'sim' if cfn else 'NAO'} "
            f"vision={getattr(self.drive, 'use_vision_caption', None)}",
            flush=True,
        )

        # 1) index (pode rodar mesmo com chat recente; so pausa se inflight)
        if await self._chat_busy_strict():
            self._skip_busy_streak += 1
            print(
                f"[DriveSync] index adiado (msg em andamento) "
                f"streak={self._skip_busy_streak}",
                flush=True,
            )
            # se stuck > 10 ciclos, ignora flag (bug leave)
            if self._skip_busy_streak < 10:
                # ainda tenta so tag se quieto? skip all
                return
            print("[DriveSync] streak alto — força ciclo (possível lock stuck)", flush=True)
            try:
                from app import runtime_flags as rf
                rf.telegram_inflight = 0
            except Exception:
                pass
        else:
            self._skip_busy_streak = 0

        res = await self.drive.sync(limit=self.batch, caption_new=False)
        added = int(res.get("added") or 0)
        total = res.get("total")
        print(
            f"[DriveSync] index novas={added} skipped={res.get('skipped')} total={total}",
            flush=True,
        )
        await asyncio.sleep(0.5)
        gc.collect()

        if self.tag_batch <= 0:
            print("[DriveSync] tag_batch=0 — tag desligado", flush=True)
            return

        if not cfn and not getattr(self.drive, "use_vision_caption", False):
            print(
                "[DriveSync] SEM caption_fn — tag impossivel. "
                "Wire album_service._auto_caption_vision no main.",
                flush=True,
            )
            return

        if not cfn:
            print(
                "[DriveSync] caption_fn=None — tentando backfill mesmo assim "
                "(pode falhar se service exigir fn)",
                flush=True,
            )

        # 2) TAG — nao pular por "muitas novas"
        if await self._chat_busy_strict():
            print("[DriveSync] tag adiada: msg em andamento", flush=True)
            return

        # stats untagged
        untagged = None
        try:
            st = await self.drive.stats()
            untagged = st.get("untagged")
            print(
                f"[DriveSync] stats total={st.get('total')} "
                f"tagged={st.get('tagged')} untagged={untagged}",
                flush=True,
            )
        except Exception as e:
            print(f"[DriveSync] stats fail: {e}", flush=True)

        if untagged is not None and int(untagged) <= 0:
            print("[DriveSync] nada pra taguear", flush=True)
            return

        done = 0
        target = self.tag_batch
        for i in range(target):
            if await self._chat_busy_strict():
                print(f"[DriveSync] tag interrompida apos {done} (chat)", flush=True)
                break
            try:
                n = await self.drive.backfill_captions(limit=1)
            except Exception as e:
                print(f"[DriveSync] backfill erro: {e}", flush=True)
                break
            n = int(n or 0)
            done += n
            if n <= 0:
                print(f"[DriveSync] backfill retornou 0 (fim ou sem fn)", flush=True)
                break
            await asyncio.sleep(2)
            gc.collect()

        self._last_tag_at = time.time()
        print(f"[DriveSync] auto-tag done={done}/{target}", flush=True)
