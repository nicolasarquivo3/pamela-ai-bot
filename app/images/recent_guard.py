"""Evita repetir a mesma foto (URL, id e hash do conteudo)."""
from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import Deque


class RecentPhotoGuard:
    def __init__(self, max_size: int = 120):
        self.max_size = max_size
        self._keys: Deque[str] = deque(maxlen=max_size)
        self._set: set[str] = set()

    def _keys_for(
        self,
        url: str | None = None,
        photo_id=None,
        content: bytes | None = None,
    ) -> list[str]:
        keys: list[str] = []
        if photo_id is not None:
            keys.append(f"id:{photo_id}")
        if url:
            clean = url.split("?")[0].rstrip("/").lower()
            keys.append(f"url:{clean}")
            # path final (mesmo CDN com host diferente)
            path = clean.rsplit("/", 1)[-1]
            if path and len(path) > 8:
                keys.append(f"path:{path}")
        if content and len(content) > 100:
            keys.append("hash:" + hashlib.md5(content[:80000]).hexdigest())
            # hash do meio (detecta crop leve)
            mid = content[len(content) // 3 : len(content) // 3 + 40000]
            if mid:
                keys.append("hash2:" + hashlib.md5(mid).hexdigest())
        return keys

    def seen(
        self,
        url: str | None = None,
        photo_id=None,
        content: bytes | None = None,
    ) -> bool:
        for k in self._keys_for(url, photo_id, content):
            if k in self._set:
                return True
        return False

    def remember(
        self,
        url: str | None = None,
        photo_id=None,
        content: bytes | None = None,
    ) -> None:
        for k in self._keys_for(url, photo_id, content):
            if k in self._set:
                continue
            if len(self._keys) >= self.max_size:
                old = self._keys.popleft()
                self._set.discard(old)
            self._keys.append(k)
            self._set.add(k)

    def size(self) -> int:
        return len(self._set)


RECENT = RecentPhotoGuard(max_size=120)


def make_jitter_seed() -> int:
    return int(time.time() * 1000) % 10_000_000
