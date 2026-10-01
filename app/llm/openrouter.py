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
    r"here'?s\s+a\s+thinking\s+process|"
    r"thinking\s+process\s*:|"
    r"let\s+me\s+(review|craft|check|think|analyze)|"
    r"looking at the conversation|"
    r"i need to (stay in character|respond as|craft)|"
    r"according to the safety|"
    r"check the (behavior )?guidelines|"
    r"the user is asking|"
    r"the user (now )?says|"
    r"key rules\s*:|"
    r"^\s*reasoning\s*:|"
    r"<think>|</think>|"
    r"chain[- ]of[- ]thought|"
    r"as an ai language model|"
    r"i'?m an? (ai|assistant|language model)|"
    r"continue the roleplay|"
    r"respond as pamela|"
    r"something like\s*:|"
    r"let me craft|"
    r"i should (move|respond|be careful)|"
    r"actually,?\s+i need|"
    r"but i need to be careful|"
    r"the last (message|thing)|"
    r"roleplay as"
    r")"
)

_PT_HINT = re.compile(
    r"(?i)\b(amor|puta|fode|trans|safad|bunda|goz|arromb|micro|saia|vestido|"
    r"hoje|ontem|fui|to |tô |nao |não |voce|você|ele |ela |comigo|namorad|"
    r"rebol|esfreg|beijo|pista|balada|depois|quero|ja |já )\b"
    r"|[áàâãéêíóôõúçÁÉÍÓÚ]"
)

# blocos meta em ingles para cortar
_META_BLOCK_RE = re.compile(
    r"(?is)("
    r"here'?s\s+a\s+thinking\s+process\s*:.*|"
    r"thinking\s+process\s*:.*"
    r")"
)


def _is_mostly_english_meta(t: str) -> bool:
    if not t:
        return True
    if _COT_RE.search(t):
        # se so um pouco no meio, ainda pode extrair PT
        en = len(re.findall(r"\b(the|and|with|that|this|user|should|need|rules|character|roleplay|message|response)\b", t, re.I))
        pt = len(re.findall(r"[áàâãéêíóôõúç]|(\b(que|nao|não|com|uma|para|ela|ele|foi|to|tô|já|ja)\b)", t, re.I))
        if en >= 8 and en > pt:
            return True
    # muitas palavras EN tipicas de CoT
    if re.search(r"(?i)here'?s a thinking|let me review|i need to respond as|key rules", t):
        return True
    return False


def _extract_portuguese_speech(t: str) -> str | None:
    """Pega so trechos que parecem fala da personagem em PT."""
    if not t:
        return None
    # corta a partir de marcadores de thinking
    cut_markers = [
        r"(?is)here'?s\s+a\s+thinking\s+process.*",
        r"(?is)\bthinking\s+process\s*:.*",
        r"(?is)\blet\s+me\s+review\b.*",
        r"(?is)\blet\s+me\s+craft\b.*",
        r"(?is)\bi\s+need\s+to\s+respond\s+as\b.*",
        r"(?is)\bkey\s+rules\s*:.*",
        r"(?is)\bthe\s+user\s+(is asking|now says)\b.*",
        r"(?is)\bi\s+should\b.*",
        r"(?is)\bsomething\s+like\s*:.*",
        r"(?is)\bcontinue\s+the\s+roleplay\b.*",
        r"(?is)<think>.*",
    ]
    cleaned = t
    for pat in cut_markers:
        cleaned = re.sub(pat, "\n", cleaned)
    # se o thinking veio NO COMECO e a fala PT no comeco antes do thinking
    # pega texto ANTES do primeiro marcador
    m = re.search(
        r"(?is)(here'?s\s+a\s+thinking|thinking\s+process\s*:|let\s+me\s+review|"
        r"i\s+need\s+to\s+respond|key\s+rules\s*:|the\s+user\s+is\s+asking)",
        t,
    )
    if m and m.start() > 30:
        before = t[: m.start()].strip()
        if _PT_HINT.search(before) and len(before) >= 40:
            cleaned = before

    # remove linhas claramente em ingles meta
    lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            continue
        if _COT_RE.search(s):
            continue
        if re.search(
            r"(?i)^(the user|i need|let me|key rules|okay|actually|something like|"
            r"continue the|respond as|based on|according to|rule \d)",
            s,
        ):
            continue
        # linha majoritariamente ingles sem acento e com palavras EN
        en_w = len(re.findall(r"\b[a-zA-Z]{3,}\b", s))
        pt_w = len(re.findall(r"[áàâãéêíóôõúçÁÉÍÓÚ]|\b(que|não|nao|com|uma|pra|pro|ele|ela|meu|minha|já|to|tô)\b", s, re.I))
        if en_w >= 8 and pt_w == 0 and re.search(r"\b(the|and|with|should|user|rules)\b", s, re.I):
            continue
        lines.append(s)
    out = " ".join(lines).strip()
    out = re.sub(r"\s{2,}", " ", out)
    # aspas com fala PT
    if len(out) < 40:
        quotes = re.findall(r'["“”]([^"“”]{30,600})["“”]', t)
        for q in quotes:
            if _PT_HINT.search(q):
                out = q.strip()
                break
    if not out or len(out) < 25:
        return None
    if not _PT_HINT.search(out):
        return None
    # se ainda tem muito ingles meta no meio
    if re.search(r"(?i)thinking process|let me review|i need to respond as|key rules", out):
        out = re.split(
            r"(?i)thinking process|let me review|i need to respond|key rules",
            out,
            maxsplit=1,
        )[0].strip()
    if len(out) < 25 or not _PT_HINT.search(out):
        return None
    out = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", out, flags=re.I).strip()
    return out[:2500]


def strip_cot_and_extract_character(text: str) -> str | None:
    """Remove thinking/ingles e devolve so a fala da personagem em PT."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()
    t = re.sub(r"(?is)</?think>", "", t).strip()

    # caso classico nemotron: fala PT + "Here's a thinking process..."
    if re.search(
        r"(?i)here'?s\s+a\s+thinking|thinking\s+process\s*:|let\s+me\s+review|"
        r"i\s+need\s+to\s+respond\s+as|key\s+rules\s*:",
        t,
    ):
        extracted = _extract_portuguese_speech(t)
        if extracted:
            print(
                f"[OpenRouter] extraiu fala PT de CoT chars={len(extracted)}",
                flush=True,
            )
            return extracted
        print("[OpenRouter] CoT ingles sem fala PT util — descarta", flush=True)
        return None

    # misturado mas com marcadores
    if _COT_RE.search(t) and _is_mostly_english_meta(t):
        extracted = _extract_portuguese_speech(t)
        if extracted:
            print(f"[OpenRouter] limpou meta chars={len(extracted)}", flush=True)
            return extracted
        print("[OpenRouter] descartou CoT/ingles (nao personagem)", flush=True)
        return None

    # PT limpo
    if _PT_HINT.search(t) and len(t) >= 40:
        # ainda remove frases meta se grudaram no fim
        t2 = re.split(
            r"(?i)\n\s*(here'?s a thinking|let me review|i need to|key rules)",
            t,
            maxsplit=1,
        )[0].strip()
        t2 = re.sub(r"^(P[aâ]mela|Pamela)\s*:\s*", "", t2, flags=re.I).strip()
        return t2[:2500]

    extracted = _extract_portuguese_speech(t)
    if extracted:
        return extracted
    return None



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
            # NUNCA devolver thinking em ingles; so tenta extrair de novo
            if re.search(r"(?i)thinking process|let me review|key rules", text or ""):
                print("[OpenRouter] recusou bruto com thinking EN", flush=True)
                return None
            if _PT_HINT.search(text or "") and len(text or "") > 50:
                # so se NAO tem bloco meta
                if not _COT_RE.search(text):
                    cleaned = text[:2500]
                    print("[OpenRouter] aceitou bruto PT limpo", flush=True)
                else:
                    return None
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
