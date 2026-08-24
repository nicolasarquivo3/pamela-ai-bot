import httpx

class GeminiLLM:
    """Gemini REST adapter. Uses the free-eligible Gemini API model configured
    in the environment; the application itself never enables billing.
    """

    def __init__(self, api_key, model="gemini-2.5-flash-lite", timeout=60, max_output_tokens=500):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def available(self):
        return bool(self.api_key)

    async def generate(self, system_instruction, messages):
        if not await self.available():
            return None

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=headers, json=payload)

            if response.status_code in (401, 403):
                return None
            if response.status_code == 429:
                return None

            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates") or []
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if p.get("text"))
            return text.strip() or None

        except (httpx.TimeoutException, httpx.HTTPError):
            return None
