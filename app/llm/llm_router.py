"""
Roteador: Gemini -> OpenRouter free; filtra recusa e CoT/meta em ingles.
"""
from __future__ import annotations

import re
from typing import Any


_REFUSAL_RE = re.compile(
    r"("
    r"n[aã]o\s+posso\s+(ajudar|continuar|falar|gerar|responder)|"
    r"n[aã]o\s+consigo\s+(ajudar|gerar|falar)|"
    r"contra\s+(as\s+)?(minhas\s+)?pol[ií]ticas|"
    r"i\s+can'?t\s+(help|assist|generate)|"
    r"i\s+cannot\s+(help|assist|generate)|"
    r"content\s+policy|safety\s+guidelines|"
    r"n[aã]o\s+vou\s+(gerar|escrever|continuar)|"
    r"prefiro\s+n[aã]o\s+(falar|continuar)"
    r")",
    re.I,
)

_COT_RE = re.compile(
    r"(?is)("
    r"okay,?\s+let'?s\s+see|looking at the|semantic\s+memor|"
    r"referencing\s+a\s+memory|provided in the prompt|"
    r"they'?re\s+referencing|there are several entries|"
    r"i need to stay in character|the user is asking|"
    r"entries about this scenario|<think>|based on the (context|memories|prompt)"
    r")"
)


class LLMRouter:
    def __init__(self, primary: Any, fallback: Any | None = None, name: str = "llm_router"):
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
        if not text or not str(text).strip():
            return True
        t = text.strip()
        if len(t) < 8:
            return False
        if _REFUSAL_RE.search(t) and (len(t) < 400 or _REFUSAL_RE.search(t[:180])):
            return True
        return False

    def _looks_like_cot(self, text: str | None) -> bool:
        if not text:
            return True
        t = text.strip()
        if _COT_RE.search(t):
            return True
        en = len(
            re.findall(
                r"\b(the|they|this|looking|memory|memories|prompt|should|user|referencing|semantic|scenario|entries)\b",
                t,
                re.I,
            )
        )
        pt = len(re.findall(r"[áàâãéêíóôõúçÁÉÍÓÚ]", t))
        if en >= 3 and pt < 2 and len(t) > 80:
            return True
        if t.lstrip().startswith("- ") and en >= 2:
            return True
        return False

    async def generate(self, system_instruction, messages):
        if self.primary and await self._avail(self.primary):
            try:
                primary_text = await self.primary.generate(
                    system_instruction, messages
                )
            except Exception as e:
                print(f"[LLMRouter] primary exception: {e}", flush=True)
                primary_text = None

            if (
                primary_text
                and not self._looks_like_refusal(primary_text)
                and not self._looks_like_cot(primary_text)
            ):
                print("[LLMRouter] usando PRIMARY (Gemini)", flush=True)
                return primary_text

            if primary_text and self._looks_like_cot(primary_text):
                print("[LLMRouter] PRIMARY CoT/meta -> FALLBACK", flush=True)
            elif primary_text and self._looks_like_refusal(primary_text):
                print("[LLMRouter] PRIMARY recusou -> FALLBACK", flush=True)
            else:
                print("[LLMRouter] PRIMARY vazio/erro/safety -> FALLBACK", flush=True)

        if self.fallback and await self._avail(self.fallback):
            try:
                fb = await self.fallback.generate(system_instruction, messages)
            except Exception as e:
                print(f"[LLMRouter] fallback exception: {e}", flush=True)
                fb = None

            if (
                fb
                and fb.strip()
                and not self._looks_like_refusal(fb)
                and not self._looks_like_cot(fb)
            ):
                print("[LLMRouter] usando FALLBACK (OpenRouter)", flush=True)
                return fb
            print("[LLMRouter] FALLBACK falhou ou CoT", flush=True)

        return None
