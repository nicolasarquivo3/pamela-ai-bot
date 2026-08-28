"""
Loop em background: sincroniza pasta do Google Drive e tagueia fotos NOVAS.

Env:
  DRIVE_ALBUM_ENABLED=true
  DRIVE_AUTO_SYNC=true
  DRIVE_SYNC_INTERVAL_SECONDS=900   # 15 min
  DRIVE_SYNC_BATCH=30               # por ciclo (cota Gemini)
"""
from __future__ import annotations

import asyncio
import os


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 900,
        batch: int = 30,
        enabled: bool = True,
    ):
        self.drive = drive_album_service
        self.interval = max(120, int(interval_seconds))
        self.batch = max(1, min(int(batch), 200))
        self.enabled = bool(enabled) and drive_album_service is not None
        self._task = None

    async def start(self):
        if not self.enabled:
            print("[DriveSync] desligado", flush=True)
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())
        print(
            f"[DriveSync] iniciado interval={self.interval}s batch={self.batch}",
            flush=True,
        )

    async def _run(self):
        # primeira sync um pouco depois do boot (deixa o app subir)
        await asyncio.sleep(45)
        while True:
            try:
                if self.drive and await self.drive.available():
                    print("[DriveSync] ciclo start", flush=True)
                    res = await self.drive.sync(
                        limit=self.batch,
                        caption_new=True,
                    )
                    print(f"[DriveSync] ciclo done {res}", flush=True)
                else:
                    print("[DriveSync] drive indisponivel", flush=True)
            except Exception as e:
                print(f"[DriveSync] erro: {e}", flush=True)
            await asyncio.sleep(self.interval)
