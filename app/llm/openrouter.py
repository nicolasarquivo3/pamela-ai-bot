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


# detecta raciocinio / monologo interno (nao e a personagem)
_COT_RE = re.compile(
    r"(?is)"
    r"("
    r"okay,?\s+let'?s\s+see|"
    r"let'?s\s+see\.|"
    r"looking at the conversation|"
    r"i need to stay in character|"
    r"according to the safety|"
    r"check the behavior guidelines|"
    r"possible approach:|"
    r"the user is asking|"
    r"my previous response|"
    r"wait,?\s+in the last|"
    r"^\s*reasoning\s*:|"
    r"<think>|"
    r"</think>|"
    r"chain[- ]of[- ]thought|"
    r"as an ai language model|"
    r"i'?m an? (ai|assistant|language model)"
    r")"
)


def strip_cot_and_extract_character(text: str) -> str | None:
    """Remove thinking e tenta achar so a fala da personagem."""
    if not text:
        return None
    t = text.strip()

    # blocos <think>...</think>
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()
    t = re.sub(r"(?is)</?think>", "", t).strip()

    # se parece CoT em ingles, tenta extrair ultima fala entre aspas ou apos "Response:"
    if _COT_RE.search(t) or (
        len(t) > 400
        and re.search(r"\b(the user|guidelines|in character|I should)\b", t)
        and not re.search(r"[áàâãéêíóôõúçÁÉÍÓÚ]", t[:200])
    ):
        # tenta trechos em PT no final
        for pat in (
            r"(?is)(?:final response|resposta final|reply|output)\s*[:\-]\s*(.+)$",
            r'(?is)"([^"]{20,400})"',
            r"(?is)'([^']{20,400})'",
        ):
            m = re.search(pat, t)
            if m:
                cand = m.group(1).strip()
                if cand and not _COT_RE.search(cand[:80]):
                    return cand[:800]
        # se nao achou fala, descarta
        print("[OpenRouter] descartou CoT/ingles (nao personagem)", flush=True)
        return None

    # remove prefixos tipo "Pâmela:" 
    t = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", t, flags=re.I).strip()
    return t if t else None


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str | None = None,
        timeout: int = 90,
        max_output_tokens: int = 500,
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
        # reforco anti-CoT no system
        guard = (
            "\n\n=== REGRAS OBRIGATORIAS DE SAIDA ===\n"
            "Responda APENAS como a personagem Pâmela, em português brasileiro.\n"
            "Proibido: raciocinio em ingles, 'Okay let\\'s see', analisar o usuario, "
            "mencionar guidelines, safety, AI, modelo, prompt.\n"
            "Proibido: [foto], [imagem].\n"
            "So a mensagem final curta (1-4 frases), tom carinhoso/flerte ficcional adulto ok.\n"
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
            print(f"[OpenRouter] sem choices: {str(data)[:400]}", flush=True)
            return None

        msg = choices[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text:
            print("[OpenRouter] resposta vazia", flush=True)
            return None

        cleaned = strip_cot_and_extract_character(text)
        if not cleaned:
            return None

        print(
            f"[OpenRouter] sucesso model={model} chars={len(cleaned)}",
            flush=True,
        )
        return cleaned

    async def generate(self, system_instruction, messages):
        if not await self.available():
            print("[OpenRouter] OPENROUTER_API_KEY ausente", flush=True)
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

        if last_err:
            print(f"[OpenRouter] todos falharam: {last_err}", flush=True)
        else:
            print("[OpenRouter] todos falharam (sem texto)", flush=True)
        return None
