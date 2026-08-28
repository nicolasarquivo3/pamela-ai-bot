"""
Drive index + auto-tag SOMENTE no ocio.

- Enquanto o usuario conversa: NAO tagueia (nao atrasa resposta).
- Quando quieto (sem msg ha IDLE_SECONDS): tagueia lote.
- Render free dorme sem HTTP: use UptimeRobot em GET /cron/drive_tag
  a cada 5-10 min (acorda o server e tagueia no ocio).
"""
from __future__ import annotations

import asyncio
import gc
import time


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 300,
        batch: int = 40,
        tag_batch: int = 12,
        enabled: bool = True,
        idle_seconds: int = 90,
    ):
        self.drive = drive_album_service
        self.interval = max(90, int(interval_seconds))
        self.batch = max(1, min(int(batch), 100))
        self.tag_batch = max(0, min(int(tag_batch), 30))
        self.enabled = bool(enabled) and drive_album_service is not None
        self.idle_seconds = max(45, int(idle_seconds))
        self._task = None

    async def start(self):
        if not self.enabled:
            print("[DriveSync] desligado", flush=True)
            return
        if self._task and not self._task.done():
            return
        if hasattr(self.drive, "_ensure_caption_fn"):
            try:
                self.drive._ensure_caption_fn()
            except Exception as e:
                print(f"[DriveSync] ensure caption: {e}", flush=True)
        self._task = asyncio.create_task(self._run(), name="drive-sync")
        print(
            f"[DriveSync] ocio-only interval={self.interval}s "
            f"idle>={self.idle_seconds}s tag_batch={self.tag_batch} "
            f"(NAO tagueia durante conversa)",
            flush=True,
        )

    async def _run(self):
        await asyncio.sleep(30)
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
            # dorme em fatias: se usuario falar, so espera
            left = self.interval
            while left > 0:
                await asyncio.sleep(min(15, left))
                left -= 15
                if await self._user_active():
                    # conversa rolando — espera ficar quieto
                    await self._wait_until_idle()

    async def _user_active(self) -> bool:
        try:
            from app.runtime_flags import telegram_busy, recently_active
            if telegram_busy():
                return True
            # quieto so se nao falou ha idle_seconds
            return recently_active(float(self.idle_seconds))
        except Exception:
            return False

    async def _wait_until_idle(self):
        print("[DriveSync] usuario ativo — pausa tag; esperando ocio...", flush=True)
        for _ in range(120):  # ate ~30 min
            await asyncio.sleep(15)
            if not await self._user_active():
                print("[DriveSync] ocio detectado — pode taguear", flush=True)
                return
        print("[DriveSync] ainda ativo apos espera — segue ciclo leve", flush=True)

    async def _one_cycle(self):
        if not self.drive or not await self.drive.available():
            print("[DriveSync] drive indisponivel", flush=True)
            return

        # NUNCA tagueia se usuario em conversa
        if await self._user_active():
            print(
                f"[DriveSync] skip tag (usuario falou ha <{self.idle_seconds}s)",
                flush=True,
            )
            # index metadata e leve e pode rodar? melhor nao competir
            return

        if hasattr(self.drive, "_ensure_caption_fn"):
            self.drive._ensure_caption_fn()

        cfn = getattr(self.drive, "_caption_fn", None)
        print(
            f"[DriveSync] OCIO ciclo tag caption_fn={'SIM' if cfn else 'NAO'}",
            flush=True,
        )

        # 1) index sem caption (leve)
        try:
            res = await self.drive.sync(limit=self.batch, caption_new=False)
            print(
                f"[DriveSync] index added={res.get('added')} total={res.get('total')}",
                flush=True,
            )
        except Exception as e:
            print(f"[DriveSync] index err: {e}", flush=True)

        # re-check idle (index pode ter demorado)
        if await self._user_active():
            print("[DriveSync] usuario voltou — aborta tag", flush=True)
            return

        if self.tag_batch <= 0:
            return
        if not cfn:
            print("[DriveSync] SEM caption_fn / GEMINI keys", flush=True)
            return

        try:
            st = await self.drive.stats()
            untagged = int(st.get("untagged") or 0)
            print(
                f"[DriveSync] stats tagged={st.get('tagged')} untagged={untagged}",
                flush=True,
            )
            if untagged <= 0:
                print("[DriveSync] nada pra taguear", flush=True)
                return
        except Exception as e:
            print(f"[DriveSync] stats: {e}", flush=True)

        # tag em fatias de 3, checando ocio entre elas
        done = 0
        target = self.tag_batch
        while done < target:
            if await self._user_active():
                print(f"[DriveSync] tag interrompida (usuario) done={done}", flush=True)
                break
            try:
                n = await self.drive.backfill_captions(limit=3)
            except Exception as e:
                print(f"[DriveSync] backfill err: {e}", flush=True)
                break
            n = int(n or 0)
            done += n
            if n <= 0:
                break
            await asyncio.sleep(1)
            gc.collect()
        print(f"[DriveSync] auto-tag ocio done={done}/{target}", flush=True)
