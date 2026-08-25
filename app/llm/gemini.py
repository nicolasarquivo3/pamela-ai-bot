import httpx


class GeminiLLM:
    """
    Adaptador REST para Gemini.
    Retorna None em erro, safety block, candidate vazio
    (dispara fallback OpenRouter no LLMRouter).
    """

    def __init__(
        self,
        api_key,
        model="gemini-2.5-flash-lite",
        timeout=60,
        max_output_tokens=500,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )

    async def available(self):
        return bool(self.api_key)

    async def generate(self, system_instruction, messages):
        if not await self.available():
            print("[Gemini] GEMINI_API_KEY nao configurada.", flush=True)
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

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            print(f"[Gemini] Enviando requisicao para modelo: {self.model}", flush=True)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=headers, json=payload)

            print(f"[Gemini] HTTP {response.status_code}", flush=True)

            if response.status_code != 200:
                print("[Gemini] ERRO DA API:", response.text[:3000], flush=True)
                return None

            data = response.json()

            feedback = data.get("promptFeedback") or {}
            if feedback.get("blockReason"):
                print(
                    f"[Gemini] BLOQUEADO promptFeedback={feedback.get('blockReason')}",
                    flush=True,
                )
                return None

            candidates = data.get("candidates") or []
            if not candidates:
                print("[Gemini] Nenhum candidate (provavel safety).", flush=True)
                print("[Gemini] Resposta:", data, flush=True)
                return None

            candidate = candidates[0]
            finish = (candidate.get("finishReason") or "").upper()
            if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "OTHER"):
                print(f"[Gemini] finishReason={finish} -> trata como bloqueio", flush=True)
                return None

            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            text = "".join(
                part.get("text", "") for part in parts if part.get("text")
            ).strip()

            if not text:
                print("[Gemini] Candidate sem texto.", flush=True)
                print("[Gemini] Candidate:", candidate, flush=True)
                return None

            print("[Gemini] Resposta recebida com sucesso.", flush=True)
            return text

        except httpx.TimeoutException as exc:
            print(f"[Gemini] TIMEOUT: {exc}", flush=True)
            return None
        except httpx.HTTPError as exc:
            print(f"[Gemini] ERRO HTTP: {exc}", flush=True)
            return None
        except Exception as exc:
            print(f"[Gemini] ERRO INESPERADO: {type(exc).__name__}: {exc}", flush=True)
            return None
