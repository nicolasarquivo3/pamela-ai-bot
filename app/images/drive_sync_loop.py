"""
Loop Drive — batches pequenos pra nao estourar RAM do Render free.

Fase A: indexar NOVAS sem baixar imagem (so metadata).
Fase B: taguear poucas com download reduzido.
"""
from __future__ import annotations

import asyncio
import gc


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 900,
        batch: int = 25,
        tag_batch: int = 5,
        enabled: bool = True,
    ):
        self.drive = drive_album_service
        self.interval = max(180, int(interval_seconds))
        self.batch = max(1, min(int(batch), 80))
        self.tag_batch = max(0, min(int(tag_batch), 15))
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
            f"index_batch={self.batch} tag_batch={self.tag_batch} (low-mem)",
            flush=True,
        )

    async def _run(self):
        await asyncio.sleep(60)
        while True:
            try:
                if self.drive and await self.drive.available():
                    await self._one_cycle()
                else:
                    print("[DriveSync] drive indisponivel", flush=True)
            except Exception as e:
                print(f"[DriveSync] erro: {e}", flush=True)
            try:
                gc.collect()
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    async def _one_cycle(self):
        print("[DriveSync] ciclo start (low-mem)", flush=True)

        # index sem caption = nao baixa foto
        res = await self.drive.sync(limit=self.batch, caption_new=False)
        added = int(res.get("added") or 0)
        total = res.get("total")
        print(
            f"[DriveSync] index fast novas={added} "
            f"skipped={res.get('skipped')} total={total}",
            flush=True,
        )
        gc.collect()

        # tag poucas
        if self.tag_batch > 0 and added < max(3, self.batch // 5):
            if hasattr(self.drive, "backfill_captions"):
                n = await self.drive.backfill_captions(limit=self.tag_batch)
                print(f"[DriveSync] auto-tag {n}/{self.tag_batch}", flush=True)
            gc.collect()
        else:
            print("[DriveSync] so index nesta rodada (tag depois)", flush=True)

        print(f"[DriveSync] ciclo done total={total}", flush=True)
