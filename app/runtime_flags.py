"""
Flags de runtime: chat do usuario SEMPRE tem prioridade sobre Drive sync/tag.
"""
from __future__ import annotations

import asyncio
import time

# quantas mensagens Telegram em processamento
telegram_inflight = 0
_lock = asyncio.Lock()
last_user_activity = 0.0  # epoch


async def telegram_enter():
    global telegram_inflight, last_user_activity
    async with _lock:
        telegram_inflight += 1
        last_user_activity = time.time()


async def telegram_leave():
    global telegram_inflight, last_user_activity
    async with _lock:
        telegram_inflight = max(0, telegram_inflight - 1)
        last_user_activity = time.time()


def telegram_busy() -> bool:
    return telegram_inflight > 0


def recently_active(seconds: float = 90.0) -> bool:
    if telegram_inflight > 0:
        return True
    if last_user_activity <= 0:
        return False
    return (time.time() - last_user_activity) < seconds
