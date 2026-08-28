"""
LLM em cascata:
  1) Gemini
  2) (so apos 2o SAFETY) modelos free menos filtrados (Venice/Dolphin etc.)
  3) OpenRouter free generico

O controle "1a vez SAFETY = so espera" fica no Agent.
Este router expoe primary / nsfw / free separados.
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
    def __init__(
        self,
        primary: Any,
        nsfw_fallback: Any | None = None,
        free_fallback: Any | None = None,
        fallback: Any | None = None,  # compat: vira free_fallback
        name: str = "llm_router",
    ):
        self.primary = primary
        self.nsfw_fallback = nsfw_fallback
        self.free_fallback = free_fallback or fallback
        self.fallback = self.free_fallback  # alias antigo
        self.name = name
        self.last_stage: str | None = None
        self.last_primary_kind: str | None = None

    async def available(self) -> bool:
        for llm in (self.primary, self.nsfw_fallback, self.free_fallback):
            if llm and await self._avail(llm):
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
                r"\b(the|they|this|looking|memory|memories|prompt|should|user|"
                r"referencing|semantic|scenario|entries)\b",
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

    def _ok(self, text: str | None) -> bool:
        return bool(
            text
            and str(text).strip()
            and not self._looks_like_refusal(text)
            and not self._looks_like_cot(text)
        )

    def _primary_kind(self) -> str:
        k = getattr(self.primary, "last_error_kind", None) if self.primary else None
        return k or "empty"

    async def generate_primary(self, system_instruction, messages) -> str | None:
        self.last_stage = "primary"
        self.last_primary_kind = None
        if not self.primary or not await self._avail(self.primary):
            self.last_primary_kind = "unavailable"
            return None
        try:
            text = await self.primary.generate(system_instruction, messages)
        except Exception as e:
            print(f"[LLMRouter] primary exception: {e}", flush=True)
            self.last_primary_kind = "exception"
            return None
        self.last_primary_kind = self._primary_kind()
        if self._ok(text):
            print("[LLMRouter] PRIMARY ok (Gemini)", flush=True)
            return text.strip()
        if text and self._looks_like_cot(text):
            self.last_primary_kind = "cot"
        elif text and self._looks_like_refusal(text):
            self.last_primary_kind = "refusal"
        elif not self.last_primary_kind or self.last_primary_kind == "empty":
            # se Gemini marcou safety
            if getattr(self.primary, "last_error_kind", None) == "safety":
                self.last_primary_kind = "safety"
            else:
                self.last_primary_kind = self.last_primary_kind or "empty"
        print(
            f"[LLMRouter] PRIMARY falhou kind={self.last_primary_kind}",
            flush=True,
        )
        return None

    async def generate_nsfw(self, system_instruction, messages) -> str | None:
        self.last_stage = "nsfw"
        if not self.nsfw_fallback or not await self._avail(self.nsfw_fallback):
            print("[LLMRouter] NSFW tier indisponivel", flush=True)
            return None
        try:
            text = await self.nsfw_fallback.generate(system_instruction, messages)
        except Exception as e:
            print(f"[LLMRouter] nsfw exception: {e}", flush=True)
            return None
        if self._ok(text):
            print("[LLMRouter] usando NSFW tier (menos filtro)", flush=True)
            return text.strip()
        print("[LLMRouter] NSFW tier falhou/CoT", flush=True)
        return None

    async def generate_free(self, system_instruction, messages) -> str | None:
        self.last_stage = "free"
        if not self.free_fallback or not await self._avail(self.free_fallback):
            print("[LLMRouter] FREE tier indisponivel", flush=True)
            return None
        try:
            text = await self.free_fallback.generate(system_instruction, messages)
        except Exception as e:
            print(f"[LLMRouter] free exception: {e}", flush=True)
            return None
        if self._ok(text):
            print("[LLMRouter] usando FREE OpenRouter", flush=True)
            return text.strip()
        print("[LLMRouter] FREE falhou/CoT", flush=True)
        return None

    async def generate(self, system_instruction, messages):
        """
        Compat: tenta primary; se falhar (qualquer motivo) nsfw depois free.
        Agent usa generate_primary + strikes para o fluxo especial SAFETY.
        """
        t = await self.generate_primary(system_instruction, messages)
        if t:
            return t
        t = await self.generate_nsfw(system_instruction, messages)
        if t:
            return t
        return await self.generate_free(system_instruction, messages)
