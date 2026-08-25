"""Evita repetir a mesma foto em sequencia (memoria do processo)."""
from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import Deque


class RecentPhotoGuard:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._keys: Deque[str] = deque(maxlen=max_size)
        self._set: set[str] = set()

    def _key(self, url: str | None = None, photo_id=None, content: bytes | None = None) -> str | None:
        if photo_id is not None:
            return f"id:{photo_id}"
        if content and len(content) > 100:
            return "hash:" + hashlib.md5(content[:50000]).hexdigest()
        if url:
            return f"url:{url.split('?')[0].rstrip('/').lower()}"
        return None

    def seen(self, url: str | None = None, photo_id=None, content: bytes | None = None) -> bool:
        k = self._key(url, photo_id, content)
        return bool(k and k in self._set)

    def remember(self, url: str | None = None, photo_id=None, content: bytes | None = None) -> None:
        k = self._key(url, photo_id, content)
        if not k or k in self._set:
            return
        if len(self._keys) >= self.max_size:
            old = self._keys.popleft()
            self._set.discard(old)
        self._keys.append(k)
        self._set.add(k)


RECENT = RecentPhotoGuard(max_size=50)


def make_jitter_seed() -> int:
    return int(time.time() * 1000) % 10_000_000
