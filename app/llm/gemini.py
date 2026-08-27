"""
Gemini LLM com suporte a VARIAS API keys (rodizio de cota).

Env:
  GEMINI_API_KEY=uma_key
  ou
  GEMINI_API_KEYS=key1,key2,key3

Quando uma key der 429 / quota / RESOURCE_EXHAUSTED, tenta a proxima.
Safety / bloqueio de conteudo NAO gasta as outras keys (mesmo modelo = mesmo filtro)
  -> devolve None para o LLMRouter cair no OpenRouter free.
"""
from __future__ import annotations

import itertools
from typing import Any


import httpx


def _parse_keys(*sources: str | None) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for src in sources:
        if not src:
            continue
        for part in str(src).replace(";", ",").replace("\n", ",").split(","):
            k = part.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            keys.append(k)
    return keys


class GeminiLLM:
    """
    Adaptador REST para Gemini.
    Retorna None em erro, safety block, candidate vazio.
    Com multiplas keys: rodizio automatico em cota/rate-limit.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        timeout: int = 60,
        max_output_tokens: int = 500,
        api_keys: str | list[str] | None = None,
    ):
        raw_list: list[str] = []
        if isinstance(api_keys, list):
            raw_list.extend(api_keys)
        elif isinstance(api_keys, str):
            raw_list.append(api_keys)
        if api_key:
            raw_list.append(api_key)

        # achata "a,b" dentro de cada item
        self.keys = _parse_keys(*raw_list) if raw_list else []
        # compat: self.api_key = primeira
        self.api_key = self.keys[0] if self.keys else None
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )
        # indice da ultima key que funcionou (sticky)
        self._idx = 0

    async def available(self) -> bool:
        return bool(self.keys)

    def _is_quota_error(self, status: int, body: str) -> bool:
        if status in (429, 503):
            return True
        b = (body or "").lower()
        return any(
            x in b
            for x in (
                "resource_exhausted",
                "quota",
                "rate limit",
                "rate_limit",
                "too many requests",
                "exceeded",
            )
        )

    async def generate(self, system_instruction, messages):
        if not await self.available():
            print("[Gemini] Nenhuma GEMINI_API_KEY / GEMINI_API_KEYS configurada.", flush=True)
            return None

        contents = []
        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": content}]})

        if not contents:
            return None

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "temperature": 0.85,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ],
        }

        n = len(self.keys)
        # tenta a partir da key sticky, depois as outras
        order = list(range(self._idx, n)) + list(range(0, self._idx))

        last_err = "no_key_tried"
        for attempt, i in enumerate(order):
            key = self.keys[i]
            headers = {
                "x-goog-api-key": key,
                "Content-Type": "application/json",
            }
            key_label = f"key#{i+1}/{n}({key[:6]}...)"

            try:
                print(
                    f"[Gemini] Enviando modelo={self.model} {key_label}",
                    flush=True,
                )
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.url, headers=headers, json=payload
                    )

                print(f"[Gemini] HTTP {response.status_code} {key_label}", flush=True)
                body = response.text or ""

                if response.status_code != 200:
                    print("[Gemini] ERRO DA API:", body[:800], flush=True)
                    if self._is_quota_error(response.status_code, body):
                        print(
                            f"[Gemini] cota/rate-limit em {key_label} "
                            f"-> tenta proxima key",
                            flush=True,
                        )
                        last_err = "quota"
                        continue
                    # erro de key invalida: tenta proxima
                    if response.status_code in (400, 401, 403) and any(
                        x in body.lower()
                        for x in ("api key", "api_key", "invalid", "permission")
                    ):
                        print(
                            f"[Gemini] key invalida/sem permissao {key_label} "
                            f"-> proxima",
                            flush=True,
                        )
                        last_err = "bad_key"
                        continue
                    # outros erros HTTP: nao queima todas as keys no mesmo prompt
                    return None

                data = response.json()

                feedback = data.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    print(
                        f"[Gemini] BLOQUEADO promptFeedback="
                        f"{feedback.get('blockReason')}",
                        flush=True,
                    )
                    # safety: nao tenta outras keys Gemini
                    return None

                candidates = data.get("candidates") or []
                if not candidates:
                    print("[Gemini] Nenhum candidate (provavel safety).", flush=True)
                    return None

                candidate = candidates[0]
                finish = (candidate.get("finishReason") or "").upper()
                if finish in (
                    "SAFETY",
                    "RECITATION",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "OTHER",
                ):
                    print(
                        f"[Gemini] finishReason={finish} -> trata como bloqueio",
                        flush=True,
                    )
                    return None

                content = candidate.get("content") or {}
                parts = content.get("parts") or []
                text = "".join(
                    part.get("text", "") for part in parts if part.get("text")
                ).strip()

                if not text:
                    print("[Gemini] Candidate sem texto.", flush=True)
                    return None

                self._idx = i  # sticky: continua nesta key
                print(
                    f"[Gemini] Resposta OK {key_label} chars={len(text)}",
                    flush=True,
                )
                return text

            except httpx.TimeoutException as exc:
                print(f"[Gemini] TIMEOUT {key_label}: {exc}", flush=True)
                last_err = "timeout"
                # timeout: tenta outra key
                continue
            except httpx.HTTPError as exc:
                print(f"[Gemini] ERRO HTTP {key_label}: {exc}", flush=True)
                last_err = "http"
                continue
            except Exception as exc:
                print(
                    f"[Gemini] ERRO INESPERADO {key_label}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                last_err = "exception"
                continue

        print(
            f"[Gemini] todas as {n} keys falharam (ultimo={last_err})",
            flush=True,
        )
        return None
