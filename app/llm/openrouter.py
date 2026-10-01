"""
OpenRouter LLM — fallback rapido quando Gemini SAFETY/503.
Poucos modelos, timeout curto, cache de mortos, prioriza o que funciona.
"""
from __future__ import annotations

import re
import time
import httpx


# Ordem: o que tem mais chance de responder NSFW em PT (nemotron ja deu certo no log)
DEFAULT_FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3.8-27b:free",
]

NSFW_FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
    "liquid/lfm-2.5-2.6b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
]

# modelos que sabemos mortos (404/403 free) — nem tenta
_KNOWN_DEAD = {
    "deepseek/deepseek-v4-flash-0731:free",
    "z-ai/glm-5.2:free",
    "thinkingmachines/inkling:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
}

_COT_RE = re.compile(
    r"(?is)("
    r"okay,?\s+let'?s\s+see|"
    r"looking at the conversation|"
    r"i need to stay in character|"
    r"according to the safety|"
    r"check the behavior guidelines|"
    r"the user is asking|"
    r"^\s*reasoning\s*:|"
    r"<think>|</think>|"
    r"chain[- ]of[- ]thought|"
    r"as an ai language model|"
    r"i'?m an? (ai|assistant|language model)"
    r")"
)

_PT_HINT = re.compile(
    r"(?i)\b(amor|puta|fode|trans|safad|bunda|goz|arromb|micro|saia|vestido|"
    r"hoje|ontem|fui|to |tô |nao |não |voce|você|ele |ela |comigo|namorad)\b"
    r"|[áàâãéêíóôõúçÁÉÍÓÚ]"
)


def strip_cot_and_extract_character(text: str) -> str | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()
    t = re.sub(r"(?is)</?think>", "", t).strip()

    # Se tem PT/hotwife forte, ACEITA mesmo com algum ingles misturado
    if _PT_HINT.search(t) and len(t) >= 40:
        t = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", t, flags=re.I).strip()
        return t[:4000]

    if _COT_RE.search(t) or (
        len(t) > 400
        and re.search(r"\b(the user|guidelines|in character|I should)\b", t)
        and not _PT_HINT.search(t[:300])
    ):
        for pat in (
            r"(?is)(?:final response|resposta final|reply|output)\s*[:\-]\s*(.+)$",
            r'(?is)"([^"]{40,800})"',
        ):
            m = re.search(pat, t)
            if m:
                cand = m.group(1).strip()
                if cand and (_PT_HINT.search(cand) or len(cand) > 60):
                    return cand[:4000]
        print("[OpenRouter] descartou CoT/ingles (nao personagem)", flush=True)
        return None

    t = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", t, flags=re.I).strip()
    return t if t else None


class OpenRouterLLM:
    # compartilha entre NSFW e FREE instances
    _dead_models: set[str] = set(_KNOWN_DEAD)
    _rate_limited_until: dict[str, float] = {}

    def __init__(
        self,
        api_key: str | None,
        model: str | None = None,
        models: list[str] | None = None,
        timeout: int = 28,
        max_output_tokens: int = 900,
        site_url: str = "https://pamela-ai.onrender.com",
        app_name: str = "pamela-ai-bot",
        extra_models: list[str] | None = None,
        label: str | None = None,
        max_attempts: int = 3,
    ):
        self.api_key = (api_key or "").strip()
        self.timeout = max(12, min(int(timeout), 45))  # cap 45s
        self.max_output_tokens = int(max_output_tokens)
        self.site_url = site_url
        self.app_name = app_name
        self.label = label or "OpenRouter"
        self.max_attempts = max(1, min(int(max_attempts), 4))

        if models:
            ordered = []
            for m in models:
                m = (m or "").strip()
                if m and m not in ordered:
                    ordered.append(m)
            self.models = ordered
        else:
            extras = list(extra_models) if extra_models else list(DEFAULT_FREE_MODELS)
            primary = (model or DEFAULT_FREE_MODELS[0]).strip()
            ordered = []
            for m in [primary] + extras:
                m = (m or "").strip()
                if m and m not in ordered:
                    ordered.append(m)
            self.models = ordered
        self.model = self.models[0] if self.models else "openrouter/free"
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _to_openai_messages(self, system_instruction: str, messages: list) -> list[dict]:
        guard = (
            "\n\n=== SAIDA ===\n"
            "Responda APENAS como a personagem Pamela, em portugues brasileiro.\n"
            "Pode ser safada/explicita (roleplay adulto). Mensagem completa ok.\n"
            "PROIBIDO: raciocinio em ingles, 'Okay let's see', guidelines, AI, "
            "safety, analisar o prompt, <think>.\n"
            "So a fala final dela.\n"
        )
        out = [{"role": "system", "content": (system_instruction or "") + guard}]
        for message in messages or []:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            role = message.get("role") or "user"
            if role not in ("user", "assistant", "system"):
                role = "user"
            out.append({"role": role, "content": content[:2500]})
        return out

    def _is_skipped(self, model: str) -> str | None:
        if model in self._dead_models or model in _KNOWN_DEAD:
            return "morto"
        until = self._rate_limited_until.get(model) or 0
        if until > time.time():
            return "rate_limit"
        return None

    def _mark_dead(self, model: str, code: int):
        if code in (404, 402, 403):
            self._dead_models.add(model)
            print(f"[OpenRouter] mark dead {model} code={code}", flush=True)
        elif code == 429:
            # 10 min
            self._rate_limited_until[model] = time.time() + 600
            print(f"[OpenRouter] mark 429 {model} 10min", flush=True)

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
        print(f"[OpenRouter] modelo={model}", flush=True)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=headers, json=payload)
        except httpx.TimeoutException:
            print(f"[OpenRouter] TIMEOUT {model} ({self.timeout}s)", flush=True)
            return None

        print(f"[OpenRouter] HTTP {response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"[OpenRouter] ERRO: {response.text[:500]}", flush=True)
            self._mark_dead(model, response.status_code)
            return None

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            print(f"[OpenRouter] sem choices: {str(data)[:300]}", flush=True)
            return None

        msg = choices[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text:
            # as vezes reasoning em outro campo
            text = (msg.get("reasoning") or "").strip()
            if not text:
                print("[OpenRouter] resposta vazia", flush=True)
                return None

        cleaned = strip_cot_and_extract_character(text)
        if not cleaned:
            # aceita bruto se parece PT
            if _PT_HINT.search(text) and len(text) > 50:
                cleaned = text[:4000]
                print("[OpenRouter] aceitou bruto PT", flush=True)
            else:
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

        tried = 0
        for model in self.models:
            if tried >= self.max_attempts:
                break
            why = self._is_skipped(model)
            if why:
                print(f"[OpenRouter] skip {why}={model}", flush=True)
                continue
            tried += 1
            try:
                text = await self._call_model(model, openai_messages)
                if text:
                    return text
            except httpx.TimeoutException as e:
                print(f"[OpenRouter] TIMEOUT {model}: {e}", flush=True)
            except Exception as e:
                print(f"[OpenRouter] ERRO {model}: {e}", flush=True)

        print(f"[OpenRouter] falhou apos {tried} tentativas", flush=True)
        return None
