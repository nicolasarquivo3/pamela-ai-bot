"""
OpenRouter LLM (OpenAI-compatible).

Lista free atualizada (ago/2026): os slugs :free mudam.
Ordem: menos filtrados primeiro, depois router generico.
"""
from __future__ import annotations

import httpx


# Modelos free que ainda existem na API (ago/2026).
# Venice :free morreu; versao paga exige credito.
DEFAULT_FREE_MODELS = [
    # Router automatico free
    "openrouter/free",
    # Compactos / chat
    "liquid/lfm-2.5-2.6b:free",
    "thinkingmachines/inkling-small:free",
    "thinkingmachines/inkling:free",
    # Geral
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "z-ai/glm-5.2:free",
    "poolside/laguna-xs-2.1:free",
    "poolside/laguna-s-2.1:free",
    "stealth/ox-alpha",
    # Se tiver credito OpenRouter (nao free):
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
]


class OpenRouterLLM:
    """
    generate(system_instruction, messages) -> str | None
    Mesma interface do GeminiLLM.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str | None = None,
        timeout: int = 90,
        max_output_tokens: int = 1000,
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
        # garante principal no inicio, sem duplicar
        ordered: list[str] = []
        for m in [self.model] + extras:
            m = (m or "").strip()
            if m and m not in ordered:
                ordered.append(m)
        self.models = ordered
        self.extra_models = ordered[1:]
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def available(self) -> bool:
        return bool(self.api_key)

    def _to_openai_messages(self, system_instruction: str, messages: list) -> list[dict]:
        out = [{"role": "system", "content": system_instruction or ""}]
        for message in messages or []:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            role = message.get("role") or "user"
            if role == "assistant":
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": "user", "content": content})
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
        print(f"[OpenRouter] modelo={model}", flush=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)

        print(f"[OpenRouter] HTTP {response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"[OpenRouter] ERRO: {response.text[:1500]}", flush=True)
            return None

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            print(f"[OpenRouter] sem choices: {str(data)[:500]}", flush=True)
            return None

        msg = choices[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text:
            print("[OpenRouter] resposta vazia", flush=True)
            return None

        print(f"[OpenRouter] sucesso model={model} chars={len(text)}", flush=True)
        return text

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
