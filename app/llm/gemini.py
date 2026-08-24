import httpx


class GeminiLLM:
    """
    Adapter REST para Gemini GenerateContent.

    O modelo é recebido pela configuração:
        GEMINI_MODEL=gemini-3.5-flash-lite
    """

    def __init__(
        self,
        api_key,
        model="gemini-3.5-flash-lite",
        timeout=60,
        max_output_tokens=1000,
    ):
        self.api_key = api_key
        self.model = self._normalize_model(model)
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

        self.url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

    @staticmethod
    def _normalize_model(model):
        """
        Aceita tanto:
            gemini-3.5-flash-lite

        quanto:
            models/gemini-3.5-flash-lite

        e sempre normaliza para o formato correto.
        """

        model = (model or "").strip()

        if model.startswith("models/"):
            model = model[len("models/"):]

        return model

    async def available(self):
        return bool(
            self.api_key
            and self.model
        )

    async def generate(
        self,
        system_instruction,
        messages,
    ):
        if not await self.available():
            print(
                "[Gemini] "
                "GEMINI_API_KEY não configurada."
            )
            return None

        contents = []

        for message in messages:

            content = str(
                message.get("content") or ""
            ).strip()

            if not content:
                continue

            original_role = message.get(
                "role",
                "user",
            )

            role = (
                "model"
                if original_role == "assistant"
                else "user"
            )

            contents.append(
                {
                    "role": role,
                    "parts": [
                        {
                            "text": content
                        }
                    ],
                }
            )

        # O Gemini precisa receber pelo menos uma mensagem.
        if not contents:
            print(
                "[Gemini] Nenhuma mensagem válida "
                "para enviar."
            )
            return None

        # O último turno deve ser do usuário.
        #
        # Isso evita enviar um histórico terminado
        # em "model", algo especialmente importante
        # para os modelos Gemini 3.x.
        if contents[-1]["role"] == "model":
            print(
                "[Gemini] Último turno era model. "
                "Resposta não será gerada."
            )
            return None

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            system_instruction or ""
                        )
                    }
                ]
            },

            "contents": contents,

            "generationConfig": {
                "maxOutputTokens": (
                    self.max_output_tokens
                ),
            },
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:

            print(
                "[Gemini] Enviando requisição:"
            )

            print(
                f"[Gemini] Modelo: {self.model}"
            )

            print(
                f"[Gemini] URL: {self.url}"
            )

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
                    "[Gemini] Nenhum candidate "
                    "retornado."
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
                    text_parts.append(
                        text
                    )

            text = "".join(
                text_parts
            ).strip()

            if not text:

                print(
                    "[Gemini] Candidate sem texto."
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
