"""
Loop Drive — NUNCA compete com chat do Telegram.

- Se usuario esta falando (ou falou ha < 90s): so indexa metadata (sem tag)
  ou adia o ciclo.
- Tag Gemini: no maximo 1-2 fotos, com sleep entre elas.
- Chat tem prioridade absoluta.
"""
from __future__ import annotations

import asyncio
import gc


class DriveSyncLoop:
    def __init__(
        self,
        drive_album_service,
        interval_seconds: int = 1200,
        batch: int = 25,
        tag_batch: int = 2,
        enabled: bool = True,
    ):
        self.drive = drive_album_service
        self.interval = max(180, int(interval_seconds))
        self.batch = max(1, min(int(batch), 80))
        self.tag_batch = max(0, min(int(tag_batch), 5))
        self.enabled = bool(enabled) and drive_album_service is not None
        self._task = None

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
            f"(prioridade=chat)",
            flush=True,
        )

    async def _run(self):
        await asyncio.sleep(90)  # deixa o webhook estabilizar no boot
        while True:
            try:
                await self._one_cycle()
            except Exception as e:
                print(f"[DriveSync] erro: {e}", flush=True)
            try:
                gc.collect()
            except Exception:
                pass
            # dorme em fatias pra poder "acordar" se quiser no futuro
            left = self.interval
            while left > 0:
                step = min(30, left)
                await asyncio.sleep(step)
                left -= step
                # se chat ativo, espera mais antes do proximo ciclo
                try:
                    from app.runtime_flags import recently_active
                    if recently_active(60):
                        await asyncio.sleep(45)
                except Exception:
                    pass

    async def _chat_busy(self) -> bool:
        try:
            from app.runtime_flags import telegram_busy, recently_active
            return telegram_busy() or recently_active(90)
        except Exception:
            return False

    async def _one_cycle(self):
        if await self._chat_busy():
            print(
                "[DriveSync] adiado: usuario ativo no chat (prioridade Telegram)",
                flush=True,
            )
            return

        if not self.drive or not await self.drive.available():
            print("[DriveSync] drive indisponivel", flush=True)
            return

        print("[DriveSync] ciclo start (low-mem, yield-to-chat)", flush=True)

        # 1) index SEM baixar/tag (leve)
        if await self._chat_busy():
            print("[DriveSync] abort index: chat chegou", flush=True)
            return

        res = await self.drive.sync(limit=self.batch, caption_new=False)
        added = int(res.get("added") or 0)
        total = res.get("total")
        print(
            f"[DriveSync] index fast novas={added} "
            f"skipped={res.get('skipped')} total={total}",
            flush=True,
        )
        await asyncio.sleep(1)
        gc.collect()

        # 2) tag so se chat quieto
        if self.tag_batch <= 0:
            print("[DriveSync] tag desligado (tag_batch=0)", flush=True)
            return

        if await self._chat_busy():
            print("[DriveSync] tag adiada: chat ativo", flush=True)
            return

        if added >= max(3, self.batch // 4):
            print("[DriveSync] so index nesta rodada (muitas novas)", flush=True)
            return

        if hasattr(self.drive, "backfill_captions"):
            # tag 1 a 1 com pausa e check de chat
            done = 0
            target = self.tag_batch
            for _ in range(target):
                if await self._chat_busy():
                    print(
                        f"[DriveSync] tag interrompida apos {done} (chat)",
                        flush=True,
                    )
                    break
                n = await self.drive.backfill_captions(limit=1)
                done += int(n or 0)
                if not n:
                    break
                # libera event loop pro webhook
                await asyncio.sleep(3)
                gc.collect()
            print(f"[DriveSync] auto-tag {done}/{target}", flush=True)
        print(f"[DriveSync] ciclo done total={total}", flush=True)
