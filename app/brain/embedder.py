"""
Lightweight, dependency-free 384-dimensional text embedder.

It uses normalized word/character n-gram hashing. It is deterministic, local,
CPU-only and produces vectors compatible with pgvector. The implementation is
deliberately dependency-free so the bot remains deployable on small free tiers.
It is a retrieval-oriented embedding, not a claim of human semantic
understanding.
"""
import hashlib
import math
import re
import unicodedata

DIMENSIONS = 384
_WORD_RE = re.compile(r"[a-z0-9à-ÿ]+", re.I)

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(text.split())

def _bucket(token: str, salt: int = 0) -> int:
    digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % DIMENSIONS

def embed(text: str) -> list[float]:
    text = _normalize(text)
    if not text:
        return [0.0] * DIMENSIONS

    values = [0.0] * DIMENSIONS
    words = _WORD_RE.findall(text)

    for word in words:
        # Multiple hashes reduce collision artifacts while keeping the vector tiny.
        for salt, weight in ((0, 1.0), (1, 0.7), (2, 0.45)):
            values[_bucket(word, salt)] += weight

        # Character n-grams help "viajar"/"viagem", plurals and inflections.
        padded = f"^{word}$"
        for n, weight in ((3, 0.35), (4, 0.25)):
            for i in range(max(0, len(padded) - n + 1)):
                values[_bucket(padded[i:i+n], 10 + n)] += weight

    # A small signal for short multi-word phrases.
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        values[_bucket(phrase, 30)] += 0.55

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return [0.0] * DIMENSIONS
    return [round(v / norm, 8) for v in values]

class Embedder:
    dimensions = DIMENSIONS

    def embed(self, text: str) -> list[float]:
        return embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [embed(text) for text in texts]
