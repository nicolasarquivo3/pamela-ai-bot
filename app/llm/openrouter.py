"""
OpenRouter LLM (OpenAI-compatible).
Padrao: Venice Dolphin uncensored free.
"""
from __future__ import annotations

import httpx


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        timeout: int = 90,
        max_output_tokens: int = 1000,
        site_url: str = "https://pamela-ai.onrender.com",
        app_name: str = "pamela-ai-bot",
        extra_models: list[str] | None = None,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.timeout = int(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.site_url = site_url
        self.app_name = app_name
        self.extra_models = extra_models or [
            "poolside/laguna-xs.2:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ]
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
            print(f"[OpenRouter] sem choices: {data}", flush=True)
            return None

        msg = (choices[0].get("message") or {})
        text = (msg.get("content") or "").strip()
        if not text:
            print("[OpenRouter] resposta vazia", flush=True)
            return None

        print("[OpenRouter] sucesso", flush=True)
        return text

    async def generate(self, system_instruction, messages):
        if not await self.available():
            print("[OpenRouter] OPENROUTER_API_KEY ausente", flush=True)
            return None

        openai_messages = self._to_openai_messages(system_instruction, messages)
        if len(openai_messages) <= 1:
            return None

        models = [self.model] + [m for m in self.extra_models if m != self.model]
        last_err = None
        for model in models:
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
        return None
