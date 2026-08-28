"""
OpenRouter LLM — free.
Aceita models= (novo) OU model= + extra_models= (antigo).
"""
from __future__ import annotations

import re
import httpx


NSFW_FREE_MODELS = [
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1:free",
]

DEFAULT_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-3-12b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-nano-9b-v2:free",
]


_COT_RE = re.compile(
    r"(?is)("
    r"okay,?\s+let'?s\s+see|looking at the|semantic\s+memor|"
    r"referencing\s+a\s+memory|provided in the prompt|"
    r"they'?re\s+referencing|there are several entries|"
    r"i need to stay in character|the user is asking|"
    r"entries about this scenario|<think>|based on the (context|memories|prompt)|"
    r"i should (respond|reply|stay)"
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
            r"\b(the|they|this|that|looking|memory|memories|prompt|should|user|"
            r"about|referencing|semantic|scenario|entries)\b",
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
    if looks_like_meta_or_english_cot(t):
        for pat in (
            r"(?is)(?:final response|resposta final|reply|output)\s*[:\-]\s*(.+)$",
            r'(?is)"([^"]{15,500})"',
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
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 90,
        max_output_tokens: int = 400,
        site_url: str = "https://pamela-ai.onrender.com",
        app_name: str = "pamela-ai-bot",
        extra_models: list | None = None,
        models: list | None = None,
        label: str = "OpenRouter",
        **kwargs,
    ):
        self.api_key = (api_key or "").strip()
        self.timeout = int(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.site_url = site_url
        self.app_name = app_name
        self.label = label or "OpenRouter"

        ordered: list[str] = []
        if models:
            for m in models:
                m = (m or "").strip()
                if m and m not in ordered:
                    ordered.append(m)
        else:
            primary = (model or DEFAULT_FREE_MODELS[0]).strip()
            if primary:
                ordered.append(primary)
            for m in extra_models or DEFAULT_FREE_MODELS:
                m = (m or "").strip()
                if m and m not in ordered:
                    ordered.append(m)
        if not ordered:
            ordered = list(DEFAULT_FREE_MODELS)
        self.models = ordered
        self.model = self.models[0]
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _to_openai_messages(self, system_instruction: str, messages: list) -> list[dict]:
        guard = (
            "\n\n=== SAIDA OBRIGATORIA ===\n"
            "Responda SOMENTE como Pâmela, portugues brasileiro, 1-5 frases.\n"
            "PROIBIDO: ingles, raciocinio, analisar memoria/prompt.\n"
            "PROIBIDO: [foto] [imagem].\n"
            "Roleplay ficcional adulto entre adultos ok; fale no personagem.\n"
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
            "temperature": 0.9,
        }
        print(f"[{self.label}] modelo={model}", flush=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)
        print(f"[{self.label}] HTTP {response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"[{self.label}] ERRO: {response.text[:600]}", flush=True)
            return None
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not text:
            return None
        cleaned = strip_cot_and_extract_character(text)
        if not cleaned:
            return None
        print(f"[{self.label}] sucesso model={model} chars={len(cleaned)}", flush=True)
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
                print(f"[{self.label}] TIMEOUT {model}: {e}", flush=True)
                last_err = e
            except Exception as e:
                print(f"[{self.label}] ERRO {model}: {e}", flush=True)
                last_err = e
        print(f"[{self.label}] todos falharam: {last_err}", flush=True)
        return None
