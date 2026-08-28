"""
OpenRouter LLM (OpenAI-compatible) — fallback free quando Gemini SAFETY/503.
"""
from __future__ import annotations

import re
import httpx


DEFAULT_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-nano-9b-v2:free",
]


_COT_RE = re.compile(
    r"(?is)"
    r"("
    r"okay,?\s+let'?s\s+see|"
    r"let'?s\s+see\.|"
    r"looking at the|"
    r"i need to stay in character|"
    r"according to the safety|"
    r"check the behavior guidelines|"
    r"possible approach:|"
    r"the user is asking|"
    r"my previous response|"
    r"wait,?\s+in the last|"
    r"^\s*reasoning\s*:|"
    r"<think>|</think>|"
    r"chain[- ]of[- ]thought|"
    r"i'?m an? (ai|assistant|language model)|"
    r"they'?re\s+referencing|"
    r"referencing\s+a\s+memory|"
    r"semantic\s+memor|"
    r"provided in the prompt|"
    r"there are several entries|"
    r"this is in the|"
    r"based on the (context|memories|prompt)|"
    r"in the prompt|"
    r"^\s*-\s+they|"
    r"^\s*-\s+the user|"
    r"let me (check|see|think)|"
    r"the user (is|has|wants|seems)|"
    r"i should (respond|reply|stay)|"
    r"entries about this scenario"
    r")"
)


def looks_like_meta_or_english_cot(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _COT_RE.search(t):
        return True
    en = len(
        re.findall(
            r"\b(the|they|this|that|looking|memory|memories|prompt|should|user|about|referencing|semantic|scenario|entries)\b",
            t,
            re.I,
        )
    )
    pt = len(re.findall(r"[áàâãéêíóôõúçÁÉÍÓÚ]", t))
    if en >= 3 and pt < 2 and len(t) > 80:
        return True
    if t.lstrip().startswith("-") and en >= 2:
        return True
    return False


def strip_cot_and_extract_character(text: str) -> str | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()
    t = re.sub(r"(?is)</?think>", "", t).strip()

    if looks_like_meta_or_english_cot(t):
        for pat in (
            r"(?is)(?:final response|resposta final|reply|output)\s*[:\-]\s*(.+)$",
            r'(?is)"([^"]{15,500})"',
            r"(?is)'([^']{15,500})'",
            r"(?im)^([^\n]*[áàâãéêíóôõúçÁÉÍÓÚ][^\n]{10,})$",
        ):
            m = re.search(pat, t)
            if m:
                cand = m.group(1).strip()
                if cand and not looks_like_meta_or_english_cot(cand):
                    return cand[:800]
        print("[OpenRouter] descartou CoT/meta/ingles", flush=True)
        return None

    t = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", t, flags=re.I).strip()
    return t if t else None


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str | None = None,
        timeout: int = 90,
        max_output_tokens: int = 400,
        site_url: str = "https://pamela-ai.onrender.com",
        app_name: str = "pamela-ai-bot",
        extra_models: list[str] | None = None,
    ):
        self.api_key = (api_key or "").strip()
        self.model = (model or DEFAULT_FREE_MODELS[0]).strip()
        self.timeout = int(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.site_url = site_url
        self.app_name = app_name

        extras = list(extra_models) if extra_models else list(DEFAULT_FREE_MODELS)
        ordered: list[str] = []
        for m in [self.model] + extras:
            m = (m or "").strip()
            if m and m not in ordered:
                ordered.append(m)
        self.models = ordered
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _to_openai_messages(self, system_instruction: str, messages: list) -> list[dict]:
        guard = (
            "\n\n=== SAIDA OBRIGATORIA ===\n"
            "Responda SOMENTE como Pâmela, portugues brasileiro, 1-4 frases.\n"
            "PROIBIDO: ingles, raciocinio, analisar memoria/prompt, "
            "'looking at', 'semantic memories', 'the user', 'they are referencing', "
            "listas com traco explicando o contexto, mencionar guidelines/AI.\n"
            "PROIBIDO: [foto] [imagem].\n"
            "Use as memorias POR DENTRO (aja como se lembrasse), nunca diga que leu o prompt.\n"
        )
        out = [{"role": "system", "content": (system_instruction or "") + guard}]
        for message in messages or []:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            role = message.get("role") or "user"
            if role not in ("user", "assistant", "system"):
                role = "user"
            out.append({"role": role, "content": content})
        return out

    async def _call_model(self, model: str, openai_messages: list) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        payload = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": self.max_output_tokens,
            "temperature": 0.85,
        }
        print(f"[OpenRouter] modelo={model}", flush=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)

        print(f"[OpenRouter] HTTP {response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"[OpenRouter] ERRO: {response.text[:800]}", flush=True)
            return None

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return None

        msg = choices[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text:
            print("[OpenRouter] resposta vazia", flush=True)
            return None

        cleaned = strip_cot_and_extract_character(text)
        if not cleaned:
            return None

        print(f"[OpenRouter] sucesso model={model} chars={len(cleaned)}", flush=True)
        return cleaned

    async def generate(self, system_instruction, messages):
        if not await self.available():
            return None

        openai_messages = self._to_openai_messages(system_instruction, messages)
        if len(openai_messages) <= 1:
            return None

        last_err = None
        for model in self.models:
            try:
                text = await self._call_model(model, openai_messages)
                if text:
                    return text
            except httpx.TimeoutException as e:
                print(f"[OpenRouter] TIMEOUT {model}: {e}", flush=True)
                last_err = e
            except Exception as e:
                print(f"[OpenRouter] ERRO {model}: {e}", flush=True)
                last_err = e

        print(f"[OpenRouter] todos falharam: {last_err}", flush=True)
        return None
