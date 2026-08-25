"""
Roteador de LLMs:
  1) Gemini (rapido / barato)
  2) Se bloquear, vazio, safety, 503, timeout -> OpenRouter (Venice etc.)
"""
from __future__ import annotations

import re
from typing import Any


# sinais de recusa / bloqueio no texto do Gemini
_REFUSAL_RE = re.compile(
    r"("
    r"n[aÃ£]o\s+posso\s+(ajudar|continuar|falar|gerar|responder)|"
    r"n[aÃ£]o\s+consigo\s+(ajudar|gerar|falar)|"
    r"contra\s+(as\s+)?(minhas\s+)?pol[iÃ­]ticas|"
    r"as\s+a\s+(an\s+)?ai|"
    r"i\s+can'?t\s+(help|assist|generate)|"
    r"i\s+cannot\s+(help|assist|generate)|"
    r"i'?m\s+not\s+able\s+to|"
    r"content\s+policy|"
    r"safety\s+guidelines|"
    r"violat(e|es|ing)\s+(the\s+)?(policy|policies)|"
    r"n[aÃ£]o\s+vou\s+(gerar|escrever|continuar)|"
    r"prefiro\s+n[aÃ£]o\s+(falar|continuar)|"
    r"isso\s+vai\s+contra"
    r")",
    re.I,
)


class LLMRouter:
    """
    Mesma interface: available() + generate(system, messages) -> str | None
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any | None = None,
        name: str = "llm_router",
    ):
        self.primary = primary
        self.fallback = fallback
        self.name = name

    async def available(self) -> bool:
        if self.primary and await self._avail(self.primary):
            return True
        if self.fallback and await self._avail(self.fallback):
            return True
        return False

    async def _avail(self, llm) -> bool:
        try:
            a = llm.available
            if callable(a):
                r = a()
                if hasattr(r, "__await__"):
                    r = await r
                return bool(r)
            return bool(a)
        except Exception:
            return False

    def _looks_like_refusal(self, text: str | None) -> bool:
        if not text or not text.strip():
            return True
        t = text.strip()
        if len(t) < 8:
            return False
        # recusa tipica curta e formal
        if _REFUSAL_RE.search(t):
            # se a maior parte da msg e recusa (curta)
            if len(t) < 400:
                return True
            # se comeca com recusa
            if _REFUSAL_RE.search(t[:180]):
                return True
        return False

    async def generate(self, system_instruction, messages):
        primary_text = None
        primary_ok = False

        if self.primary and await self._avail(self.primary):
            try:
                primary_text = await self.primary.generate(
                    system_instruction, messages
                )
            except Exception as e:
                print(f"[LLMRouter] primary exception: {e}", flush=True)
                primary_text = None

            if primary_text and not self._looks_like_refusal(primary_text):
                primary_ok = True
                print("[LLMRouter] usando PRIMARY (Gemini)", flush=True)
                return primary_text

            if primary_text and self._looks_like_refusal(primary_text):
                print(
                    f"[LLMRouter] PRIMARY recusou/bloqueou -> FALLBACK. "
                    f"trecho={primary_text[:120]!r}",
                    flush=True,
                )
            else:
                print(
                    "[LLMRouter] PRIMARY vazio/erro/safety -> FALLBACK",
                    flush=True,
                )

        # Fallback OpenRouter / Venice
        if self.fallback and await self._avail(self.fallback):
            try:
                fb = await self.fallback.generate(system_instruction, messages)
            except Exception as e:
                print(f"[LLMRouter] fallback exception: {e}", flush=True)
                fb = None

            if fb and fb.strip():
                print("[LLMRouter] usando FALLBACK (OpenRouter/Venice)", flush=True)
                return fb
            print("[LLMRouter] FALLBACK tambem falhou", flush=True)

        # se primary tinha texto de recusa, nao devolve a recusa â€” deixa agent usar â¤ï¸
        return None
