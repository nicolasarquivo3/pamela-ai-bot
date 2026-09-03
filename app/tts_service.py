"""
TTS em cascata:
  1) ElevenLabs free tier (voz mais sensual) — se tiver ELEVENLABS_API_KEY
  2) Se cota/erro/falha -> edge-tts (gratis ilimitado, sussurro)

Env:
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...   (opcional)
  ELEVENLABS_MODEL=eleven_multilingual_v2
  TTS_ENABLED=true
  TTS_VOICE=pt-BR-FranciscaNeural
  TTS_RATE=-35%
  TTS_PITCH=-8Hz
  TTS_VOLUME=-12%
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import httpx


# ---------- filtros de fala (compartilhados) ----------

_ACTION_START = re.compile(
    r"^(?:"
    r"pego|abro|fecho|caminho|saio|entro|olho|olhando|mordo|lambe|des[cç]o|subo|"
    r"tiro|coloco|ajusto|mexo|dou|vou|corro|paro|ligo|desligo|envio|digito|"
    r"escrevo|gravo|seguro|aperto|sinto|"
    r"me\s+aproximo|me\s+afasto|me\s+viro|me\s+sento|me\s+deito|"
    r"ele\s+|ela\s+|os\s+caras|o\s+cara|a\s+m[aã]o|com\s+os\s+dedos|"
    r"pausa|continua|descreve|narr[ae]|a[cç][aã]o|comando|ooc|"
    r"system|prompt|image_prompt"
    r")\b",
    re.I,
)
_ACTION_ANY = re.compile(
    r"\b(?:"
    r"pego\s+o\s+celular|dedos\s+tremendo|tiro\s+uma\s+foto|no\s+espelho|"
    r"caminho\s+at[eé]|corredor|faculdade|abro\s+a\s+porta|"
    r"com\s+os\s+dedos|com\s+a\s+m[aã]o|respondo\s+gemendo|"
    r"te\s+respondo\s+|digito\s+|envio\s+um|"
    r"\[(?:foto|imagem|audio|voice|comando|action|ooc)\]|"
    r"(?:^|\s)/(?:foto|reset|album|drive|audio|voice)\b"
    r")",
    re.I,
)
_META = re.compile(
    r"(?i)(?:"
    r"image_prompt|system\s*prompt|as\s+a\s+language\s+model|"
    r"ooc\b|out\s+of\s+character|"
    r"\[(?:system|inst|cmd|command|comando)\]|"
    r"\{[\"']?(?:type|action|command)"
    r")"
)


def _strip_noise(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"---\s*IMAGE_PROMPT:.*", "", t, flags=re.S | re.I)
    t = re.sub(r"IMAGE_PROMPT:.*", "", t, flags=re.S | re.I)
    t = re.sub(r"(?m)^\s*/[a-zA-Z0-9_]+\b.*$", " ", t)
    t = re.sub(r"\s/[a-zA-Z0-9_]{3,}\b", " ", t)
    t = re.sub(
        r"\[\s*(foto|imagem|photo|selfie|audio|voice|comando|action|ooc)\s*\]",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(r"\*+[^*]{1,120}\*+", " ", t)
    t = re.sub(r"\([^)]{1,120}\)", " ", t)
    t = re.sub(r"_[^_]{1,80}_", " ", t)
    t = re.sub(r"[*_`#~>]+", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E0-\U0001F1FF]+",
        "",
        t,
    )
    for e in ("❤️", "♥", "❤", "🔥", "😈", "😏", "💋", "🥵", "😉", "😊", "😅", "✨"):
        t = t.replace(e, "")
    return re.sub(r"\s+", " ", t).strip()


def _is_forbidden(s: str) -> bool:
    low = (s or "").strip().lower()
    if not low or len(low) < 4:
        return True
    if _META.search(low) or _ACTION_START.search(low) or _ACTION_ANY.search(low):
        return True
    if re.fullmatch(r"(amor|meu deus|nossa|ai|ah|hmm+)([.!\s…]*)", low):
        return True
    if re.search(r'[{}\[\]|=]{2,}|"type"\s*:', s):
        return True
    return False


def _is_phone_speech(s: str) -> bool:
    if _is_forbidden(s):
        return False
    low = s.lower()
    return bool(
        re.search(
            r"\b(amor|querido|beb[eê]|voc[eê]|te\s+|teu|tua|me\s+|"
            r"quero|quer|fico|t[oô]\s+|sinto|pensa|imagina|ouve|ouvir|escuta|"
            r"vem|fica|tes[aã]o|calcinha|safad|gostos|molhad|"
            r"cama|colo|beijo|olhando|derreto|baixinho|assim|forte|"
            r"gostando|pensando|sem\s+nada)\b",
            low,
        )
    )


def _fallback_whisper(original: str) -> str:
    low = (original or "").lower()
    if re.search(r"\b(tes[aã]o|cama|goz|gemid|foder|sexo|molhad)\b", low):
        return "Amor... fico com tanto tesão... por você..."
    if re.search(r"\b(calcinha|saia|sem roupa|micro|lingerie)\b", low):
        return "Amor... tô quase sem nada... só pra você..."
    if re.search(r"\b(olhando|olha|corredor|faculdade)\b", low):
        return "Amor... só de você me olhando... eu derreto..."
    if re.search(r"\b(ouvir|voz|sussurr|baixinho)\b", low):
        return "Amor... quer me ouvir baixinho... assim?"
    return "Amor... tô pensando em você... bem baixinho..."


def _to_whisper(text: str, max_chars: int = 90) -> str:
    t = text.strip()
    t = re.sub(r"\b(meu deus!?|nossa!?|kk+|ha(ha)+|rs+)\b", "", t, flags=re.I)
    t = re.sub(r"[!]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(
        r"\b(pego o celular|com os dedos(?: tremendo)?|te respondo(?: gemendo)?|"
        r"gemendo baixinho|digito|envio)\b[,:]?\s*",
        "",
        t,
        flags=re.I,
    )
    repl = [
        (r"(?i)\beu fico com muito tes[aã]o\b", "fico com tanto tesão"),
        (r"(?i)\bcom muito tes[aã]o\b", "com tanto tesão"),
        (r"(?i)\bmuito tes[aã]o\b", "tanto tesão"),
        (r"(?i)\bs[oó] de pensar em voc[eê] me olhando\b", "só de você me olhando"),
        (r"(?i)\bem como a gente fica depois na cama\b", "na cama depois"),
        (r"(?i)\beu fico\b", "fico"),
        (r"(?i)\bestou\b", "tô"),
    ]
    for a, b in repl:
        t = re.sub(a, b, t)
    t = re.sub(r"\s*,\s*", "... ", t)
    t = re.sub(r"\s+e\s+", "... ", t, count=1)
    t = re.sub(r"\.{2,}", "...", t)
    t = re.sub(r"(\.\.\.\s*)+", "... ", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    t = re.sub(r"(?i)^(?:amor\s*\.\.\.\s*)+", "", t).strip()
    t = re.sub(r"(?i)^amor\b\s*", "", t).strip()
    if t:
        rest = t if t[0].isupper() else (t[0].lower() + t[1:])
        t = "Amor... " + rest
    else:
        t = "Amor..."
    t = re.sub(r"\s*\.\.\.\s*", "... ", t)
    t = re.sub(r"(\.\.\.\s*){2,}", "... ", t).strip()
    if len(t) > max_chars:
        parts = [p.strip() for p in t.split("...") if p.strip()]
        keep, total = [], 0
        for p in parts:
            if not keep:
                keep.append(p)
                total += len(p)
                continue
            if len(keep) >= 3 or total + len(p) > max_chars:
                break
            keep.append(p)
            total += len(p)
        t = "... ".join(keep)
        if not t.lower().startswith("amor"):
            t = "Amor... " + t
    t = t.strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    t = t.rstrip(".")
    if not t.endswith("...") and not t.endswith("?"):
        t += "..."
    return t


def clean_for_speech(text: str, max_chars: int = 90) -> str:
    if _META.search(text or ""):
        return _to_whisper(_fallback_whisper(text), max_chars)

    raw = _strip_noise(text)
    chunks: list[str] = []
    for m in re.finditer(r"[\"“']([^\"”']{5,120})[\"”']", raw):
        chunks.append(m.group(1).strip())
    for part in re.split(r"[\n]+", raw):
        if ":" in part:
            after = part.split(":")[-1].strip().strip("\"'“”")
            after = re.sub(
                r"^(baixinho|baixo|pro voc[eê]|pra voc[eê]|gemendo)\s*",
                "",
                after,
                flags=re.I,
            )
            if after:
                chunks.append(after)
    for p in re.split(r"(?<=[\.\!\?\…])\s+", raw):
        p = p.strip().strip("\"'“”").strip(" -–—")
        if p:
            chunks.append(p)

    good = []
    for c in chunks:
        c = c.strip()
        if _is_forbidden(c) or not _is_phone_speech(c):
            continue
        good.append(c)

    if good:
        def score(s: str) -> tuple:
            low = s.lower()
            sc = 0
            if re.search(r"\b(quer me|me ouvir|tes[aã]o|calcinha|mais forte|derreto)\b", low):
                sc += 6
            if s.endswith("?"):
                sc += 3
            if 10 <= len(s) <= 70:
                sc += 3
            if len(s) > 100:
                sc -= 3
            return (-sc, len(s))

        good.sort(key=score)
        chosen = good[0]
        if _ACTION_ANY.search(chosen) or _ACTION_START.search(chosen.strip()):
            chosen = _fallback_whisper(text)
        out = _to_whisper(chosen, max_chars)
    else:
        out = _to_whisper(_fallback_whisper(text), max_chars)

    bare = re.sub(r"(?i)amor|\.|\s", "", out)
    if len(bare) < 4 or _ACTION_ANY.search(out):
        out = _to_whisper(_fallback_whisper(text), max_chars)
    return out


# ---------- ElevenLabs ----------

class ElevenLabsTTS:
    """
    Free tier ElevenLabs. Quando estoura cota (401/402/429) sinaliza esgotado.
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        timeout: int = 60,
    ):
        self.api_key = (api_key or os.getenv("ELEVENLABS_API_KEY") or "").strip()
        # Rachel / Bella / Elli etc — user pode trocar. Default: "Sarah" multilingual-ish public
        # Voice "Rachel" EXAVITQu4vr4xnSDxMaL is classic; for more soft: "Elli" MF3mGyEYCl7XYWbV9V6O
        # IMPORTANTE: no free tier NAO use voice ID da biblioteca publica.
        # Use Voice ID de voz criada/clonada NA SUA conta (Voices -> My Voices).
        self.voice_id = (
            voice_id
            or os.getenv("ELEVENLABS_VOICE_ID")
            or ""
        ).strip()
        self.model_id = (
            model_id
            or os.getenv("ELEVENLABS_MODEL")
            or "eleven_multilingual_v2"
        ).strip()
        self.timeout = timeout
        # estado de cota
        self.quota_exhausted = False
        self.quota_until = 0.0  # epoch: retry after
        self._cooldown_sec = int(os.getenv("ELEVENLABS_COOLDOWN_SEC") or 3600)

    def available(self) -> bool:
        if not self.api_key:
            return False
        if not self.voice_id:
            print(
                "[TTS/11] ELEVENLABS_VOICE_ID vazio — use Voice ID da SUA conta "
                "(My Voices), nao da biblioteca. Usando so edge-tts.",
                flush=True,
            )
            return False
        if self.quota_exhausted and time.time() < self.quota_until:
            return False
        if self.quota_exhausted and time.time() >= self.quota_until:
            self.quota_exhausted = False
            print("[TTS/11] cooldown acabou — tenta ElevenLabs de novo", flush=True)
        return True

    def _mark_exhausted(self, reason: str) -> None:
        self.quota_exhausted = True
        self.quota_until = time.time() + self._cooldown_sec
        print(
            f"[TTS/11] cota/limite: {reason} — fallback edge por {self._cooldown_sec}s",
            flush=True,
        )

    async def synthesize(self, text: str) -> bytes | None:
        if not self.available():
            return None
        clean = clean_for_speech(text)
        if not clean:
            return None
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        # settings mais "sussurro/sensuais"
        stability = float(os.getenv("ELEVENLABS_STABILITY") or "0.35")
        similarity = float(os.getenv("ELEVENLABS_SIMILARITY") or "0.75")
        style = float(os.getenv("ELEVENLABS_STYLE") or "0.45")
        payload = {
            "text": clean,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity,
                "style": style,
                "use_speaker_boost": True,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code in (401, 402, 429):
                body = (r.text or "")[:300]
                # free tier: library voices bloqueadas na API
                if "paid_plan_required" in body or "library voices" in body.lower():
                    print(
                        "[TTS/11] voz da biblioteca NAO funciona no free via API. "
                        "Crie/clone uma voz na sua conta e use o Voice ID dela em ELEVENLABS_VOICE_ID.",
                        flush=True,
                    )
                self._mark_exhausted(f"HTTP {r.status_code} {body}")
                return None
            if r.status_code == 400 and re.search(
                r"quota|credit|limit|exceed|payment", r.text or "", re.I
            ):
                self._mark_exhausted(f"HTTP 400 quota: {(r.text or '')[:160]}")
                return None
            if r.status_code != 200:
                print(
                    f"[TTS/11] HTTP {r.status_code} body={(r.text or '')[:180]}",
                    flush=True,
                )
                # 5xx: nao marca cota, so falha
                return None
            data = r.content
            if not data or len(data) < 200:
                print("[TTS/11] empty audio", flush=True)
                return None
            print(
                f"[TTS/11] ok voice={self.voice_id} model={self.model_id} "
                f"bytes={len(data)} spoken={clean!r}",
                flush=True,
            )
            return data
        except Exception as e:
            print(f"[TTS/11] error: {e}", flush=True)
            return None


# ---------- Edge TTS (fallback) ----------

class EdgeTTSService:
    name = "edge_tts"

    def __init__(
        self,
        voice: str | None = None,
        enabled: bool = True,
        rate: str | None = None,
        pitch: str | None = None,
        volume: str | None = None,
        max_chars: int = 90,
        style: str | None = None,
    ):
        self.voice = (voice or os.getenv("TTS_VOICE") or "pt-BR-FranciscaNeural").strip()
        self.enabled = bool(enabled) and (
            os.getenv("TTS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        self.rate = (rate or os.getenv("TTS_RATE") or "-35%").strip()
        self.pitch = (pitch or os.getenv("TTS_PITCH") or "-8Hz").strip()
        self.volume = (volume or os.getenv("TTS_VOLUME") or "-12%").strip()
        self.max_chars = int(os.getenv("TTS_MAX_CHARS") or max_chars)
        self.style = (style or os.getenv("TTS_STYLE") or "whisper").strip().lower()
        # SSML DESLIGADO: edge-tts lia as tags em voz alta
        self.use_ssml = False

    def clean_for_speech(self, text: str) -> str:
        return clean_for_speech(text, self.max_chars)

    async def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception:
            return False


    def _plain_for_edge(self, text: str) -> str:
        """
        Texto puro para edge-tts (SEM SSML).
        Reticencias viram pausas naturais na fala; sem tags XML.
        """
        s = (text or "").strip()
        # se vazou SSML de versao antiga, remove tags
        if "<speak" in s.lower() or "<prosody" in s.lower() or "<break" in s.lower():
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
        # normaliza reticencias (pausa)
        s = re.sub(r"\.{2,}", "...", s)
        s = re.sub(r"\s*\.\.\.\s*", "... ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    async def synthesize(self, text: str) -> bytes | None:
        if not self.enabled:
            return None
        clean = clean_for_speech(text, self.max_chars)
        clean = self._plain_for_edge(clean)
        if not clean or len(clean) < 3:
            return None
        # barreira: nunca sintetizar markup
        if re.search(r"(?i)<\s*(speak|prosody|break|voice)\b", clean):
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
        try:
            import edge_tts
        except Exception as e:
            print(f"[TTS/edge] import fail: {e}", flush=True)
            return None
        tmp = None
        try:
            # SEMPRE texto puro + rate/pitch/volume nos parametros
            communicate = edge_tts.Communicate(
                clean,
                self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            await communicate.save(tmp)
            data = Path(tmp).read_bytes()
            if not data or len(data) < 200:
                print("[TTS/edge] empty output", flush=True)
                return None
            print(
                f"[TTS/edge] ok rate={self.rate} pitch={self.pitch} vol={self.volume} "
                f"whisper={clean!r} bytes={len(data)}",
                flush=True,
            )
            return data
        except Exception as e:
            print(f"[TTS/edge] error: {e}", flush=True)
            return None
        finally:
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass




# ---------- Cascata: Eleven -> Edge ----------

class CascadeTTSService:
    """
    API unica usada pelo bot/agent.
    synthesize(): tenta ElevenLabs; se cota/erro -> edge-tts.
    """

    name = "cascade_tts"

    def __init__(
        self,
        eleven: ElevenLabsTTS | None = None,
        edge: EdgeTTSService | None = None,
    ):
        self.eleven = eleven
        self.edge = edge or EdgeTTSService()
        self.enabled = True

    def clean_for_speech(self, text: str) -> str:
        return clean_for_speech(text)

    async def available(self) -> bool:
        if self.eleven and self.eleven.available():
            return True
        if self.edge:
            return await self.edge.available()
        return False

    async def synthesize(self, text: str) -> bytes | None:
        # 1) ElevenLabs se tiver key e cota
        if self.eleven and self.eleven.available():
            data = await self.eleven.synthesize(text)
            if data:
                return data
            print("[TTS] ElevenLabs falhou/cota -> edge-tts", flush=True)
        else:
            if self.eleven and self.eleven.quota_exhausted:
                print("[TTS] ElevenLabs em cooldown de cota -> edge-tts", flush=True)
            elif not (self.eleven and self.eleven.api_key):
                print("[TTS] sem ELEVENLABS_API_KEY -> edge-tts", flush=True)

        # 2) edge fallback
        if self.edge:
            return await self.edge.synthesize(text)
        return None
