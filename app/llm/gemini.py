import httpx


class GeminiLLM:
    """
    Adaptador REST para a Gemini API.

    Usa o endpoint GenerateContent e mantém o histórico
    recebido pelo AgentBrain.
    """

    def __init__(
        self,
        api_key,
        model="gemini-3.5-flash-lite",
        timeout=60,
        max_output_tokens=500,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

        self.url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

    async def available(self):
        return bool(self.api_key)

    async def generate(
        self,
        system_instruction,
        messages,
    ):
        if not await self.available():
            print(
                "[Gemini] ERRO: GEMINI_API_KEY não configurada."
            )
            return None

        if not messages:
            print(
                "[Gemini] ERRO: nenhum histórico de mensagens recebido."
            )
            return None

        contents = []

        for message in messages:
            content = message.get("content", "")

            if not content:
                continue

            role = message.get("role", "user")

            # Gemini aceita apenas user/model no histórico.
            if role == "assistant":
                role = "model"
            else:
                role = "user"

            contents.append(
                {
                    "role": role,
                    "parts": [
                        {
                            "text": str(content)
                        }
                    ],
                }
            )

        if not contents:
            print(
                "[Gemini] ERRO: histórico vazio após conversão."
            )
            return None

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": system_instruction
                    }
                ]
            },
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        print(
            f"[Gemini] Enviando requisição para modelo: "
            f"{self.model}"
        )

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                )

            print(
                f"[Gemini] HTTP {response.status_code}"
            )

            if response.status_code != 200:

                print(
                    "[Gemini] ERRO DA API:"
                )

                print(
                    response.text[:5000]
                )

                return None

            data = response.json()

            candidates = (
                data.get("candidates")
                or []
            )

            if not candidates:

                print(
                    "[Gemini] Nenhum candidate retornado."
                )

                print(
                    "[Gemini] Resposta:"
                )

                print(data)

                return None

            candidate = candidates[0]

            content = (
                candidate.get("content")
                or {}
            )

            parts = (
                content.get("parts")
                or []
            )

            text_parts = []

            for part in parts:

                text = part.get("text")

                if text:
                    text_parts.append(text)

            text = "".join(text_parts).strip()

            if not text:

                print(
                    "[Gemini] Candidate retornado "
                    "sem texto."
                )

                print(
                    "[Gemini] Candidate:"
                )

                print(candidate)

                return None

            print(
                "[Gemini] Resposta recebida "
                "com sucesso."
            )

            return text

        except httpx.TimeoutException as exc:

            print(
                f"[Gemini] TIMEOUT: {exc}"
            )

            return None

        except httpx.HTTPError as exc:

            print(
                f"[Gemini] ERRO HTTP: {exc}"
            )

            return None

        except Exception as exc:

            print(
                "[Gemini] ERRO INESPERADO: "
                f"{type(exc).__name__}: {exc}"
            )

            return None
