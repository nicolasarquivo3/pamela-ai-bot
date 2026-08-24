import re
from dataclasses import dataclass
from app.database.models import MemoryKind

@dataclass
class MemoryCandidate:
    kind: MemoryKind
    key: str
    value: str
    importance: float
    confidence: float

class MemoryExtractor:
    """Heuristic extractor: no paid LLM is required.
    It deliberately stores stable, user-provided facts/preferences and avoids
    guessing sensitive attributes from casual conversation.
    """

    PATTERNS = [
        (MemoryKind.PREFERENCE, "favorite_food", r"\b(?:minha comida favorita é|eu amo comer|gosto muito de)\s+(.+?)(?:[.!?]|$)", 0.75),
        (MemoryKind.PREFERENCE, "favorite_music", r"\b(?:minha música favorita é|eu gosto de ouvir)\s+(.+?)(?:[.!?]|$)", 0.65),
        (MemoryKind.PREFERENCE, "hobby", r"\b(?:meu hobby é|meus hobbies são|eu gosto de)\s+(.+?)(?:[.!?]|$)", 0.55),
        (MemoryKind.PROFILE, "name", r"\b(?:meu nome é|pode me chamar de)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '\-]{1,60})(?:[.!?]|$)", 0.95),
    ]

    def extract(self, text: str) -> list[MemoryCandidate]:
        out = []
        clean = " ".join(text.strip().split())
        for kind, key, pattern, importance in self.PATTERNS:
            for m in re.finditer(pattern, clean, flags=re.I):
                value = m.group(1).strip(" ,.")
                if 1 < len(value) <= 160:
                    out.append(MemoryCandidate(kind, key, value, importance, 0.78))
        # Explicit corrections should be high-confidence.
        m = re.search(r"\b(?:na verdade|corrigindo|o certo é)\s+(.+?)(?:[.!?]|$)", clean, re.I)
        if m:
            out.append(MemoryCandidate(MemoryKind.FACT, "correction", m.group(1).strip(), 0.9, 0.9))
        return out
