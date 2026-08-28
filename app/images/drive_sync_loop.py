"""
Loop em background: Google Drive album.

Fase A — indexar NOVAS sem caption (rapido, sem gastar Gemini).
Fase B — se nao houver novas, taguear ate N fotos sem legenda.

Env:
  DRIVE_ALBUM_ENABLED=true
  DRIVE_AUTO_SYNC=true
  DRIVE_SYNC_INTERVAL_SECONDS=900
  DRIVE_SYNC_BATCH=50          # novas por ciclo (fast)
  DRIVE_TAG_BATCH=15           # captions por ciclo (Gemini)
"""
from __future__ import annotations

import asyncio
import os


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 900,
        batch: int = 50,
        tag_batch: int = 15,
        enabled: bool = True,
    ):
        self.drive = drive_album_service
        self.interval = max(120, int(interval_seconds))
        self.batch = max(1, min(int(batch), 300))
        self.tag_batch = max(0, min(int(tag_batch), 50))
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
            f"[DriveSync] iniciado interval={self.interval}s "
            f"index_batch={self.batch} tag_batch={self.tag_batch}",
            flush=True,
        )

    async def _run(self):
        await asyncio.sleep(45)
        while True:
            try:
                if self.drive and await self.drive.available():
                    await self._one_cycle()
                else:
                    print("[DriveSync] drive indisponivel", flush=True)
            except Exception as e:
                print(f"[DriveSync] erro: {e}", flush=True)
            await asyncio.sleep(self.interval)

    async def _one_cycle(self):
        print("[DriveSync] ciclo start", flush=True)

        # 1) Indexar novas SEM caption (rapido)
        res = await self.drive.sync(
            limit=self.batch,
            caption_new=False,
        )
        added = int(res.get("added") or 0)
        total = res.get("total")
        print(
            f"[DriveSync] index fast novas={added} "
            f"skipped={res.get('skipped')} total={total}",
            flush=True,
        )

        # 2) Se poucas novas (ou zero), taguear um lote sem caption
        if self.tag_batch > 0 and added < max(5, self.batch // 4):
            if hasattr(self.drive, "backfill_captions"):
                n = await self.drive.backfill_captions(limit=self.tag_batch)
                print(f"[DriveSync] auto-tag {n}/{self.tag_batch}", flush=True)
            elif hasattr(self.drive, "sync"):
                # fallback: sync so com caption em arquivos sem legenda
                res2 = await self.drive.sync(
                    limit=self.tag_batch,
                    caption_new=True,
                )
                print(
                    f"[DriveSync] auto-tag via sync captioned={res2.get('captioned')}",
                    flush=True,
                )
        else:
            print(
                "[DriveSync] ciclo focado em indexar; tag fica pro proximo "
                "(muitas novas nesta rodada)",
                flush=True,
            )

        print(f"[DriveSync] ciclo done total={total}", flush=True)
