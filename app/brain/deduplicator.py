import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from app.database.models import MemoryKind
from app.brain.memory_extractor import MemoryCandidate

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()

@dataclass
class MemoryDecision:
    action: str  # create | reinforce | replace | ignore
    candidate: MemoryCandidate
    existing_id: int | None = None

class Deduplicator:
    def decide(self, candidate, existing):
        if not existing:
            return MemoryDecision("create", candidate)
        a, b = normalize(candidate.value), normalize(existing.value)
        if a == b or SequenceMatcher(None, a, b).ratio() >= 0.92:
            return MemoryDecision("reinforce", candidate, existing.id)
        # Same key but materially different value: replace only when the new
        # statement is at least as confident; this supports corrections.
        if candidate.confidence >= float(existing.confidence):
            return MemoryDecision("replace", candidate, existing.id)
        return MemoryDecision("ignore", candidate, existing.id)
