"""
TTS gratis edge-tts (pt-BR), estilo intimo: lento, tom baixo, trecho curto.
Nao soa como locucao de anuncio.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


class EdgeTTSService:
    name = "edge_tts"

    def __init__(
        self,
        voice: str | None = None,
        enabled: bool = True,
        rate: str | None = None,
        pitch: str | None = None,
        max_chars: int = 220,
        style: str | None = None,
    ):
        self.voice = (voice or os.getenv("TTS_VOICE") or "pt-BR-FranciscaNeural").strip()
        self.enabled = bool(enabled) and (
            os.getenv("TTS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        self.rate = (rate or os.getenv("TTS_RATE") or "-18%").strip()
        self.pitch = (pitch or os.getenv("TTS_PITCH") or "-3Hz").strip()
        self.max_chars = int(os.getenv("TTS_MAX_CHARS") or max_chars)
        self.style = (style or os.getenv("TTS_STYLE") or "intimate").strip().lower()
        self.use_ssml = os.getenv("TTS_SSML", "true").lower() in ("1", "true", "yes", "on")

    async def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception:
            return False

    def _strip_noise(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        t = re.sub(r"---\s*IMAGE_PROMPT:.*", "", t, flags=re.S | re.I)
        t = re.sub(r"\[\s*(foto|imagem|photo|selfie)\s*\]", "", t, flags=re.I)
        t = re.sub(r"[*_`#~>]+", "", t)
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E0-\U0001F1FF]+",
            "",
            t,
        )
        t = t.replace("❤️", "").replace("♥", "").replace("❤", "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _pick_intimate_excerpt(self, text: str) -> str:
        t = self._strip_noise(text)
        if not t:
            return ""
        parts = re.split(r"(?<=[\.\!\?\…])\s+", t)
        parts = [p.strip() for p in parts if p and len(p.strip()) > 8]
        sexy_kw = re.compile(
            r"tes[aã]o|safad|gostos|molhad|gemid|goz|foder|fode|transar|"
            r"beijo|colo|cama|ouvido|sussurr|provoc|querido|amor|corpo|"
            r"saia|calcinha|peito|bunda|quero\s+voc",
            re.I,
        )
        scored = []
        for p in parts:
            score = 0
            if sexy_kw.search(p):
                score += 5
            if len(p) < 120:
                score += 2
            if re.search(r"\b(eu|meu|minha|voc[eê]|amor)\b", p, re.I):
                score += 1
            if re.search(r"\b(porque|pois|ent[aã]o|na verdade|tipo assim)\b", p, re.I):
                score -= 1
            scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        chosen = []
        total = 0
        for sc, p in scored:
            if not chosen:
                chosen.append(p)
                total += len(p)
                continue
            if total + len(p) > self.max_chars:
                break
            if sc >= 3 and len(chosen) < 2:
                chosen.append(p)
                total += len(p)
        if not chosen:
            chosen = [t[: self.max_chars]]
        out = " ".join(chosen).strip()
        if len(out) > self.max_chars:
            cut = out[: self.max_chars]
            for sep in (". ", "! ", "? ", ", "):
                j = cut.rfind(sep)
                if j > 40:
                    cut = cut[: j + (0 if sep == ", " else 1)]
                    break
            out = cut.strip()
        return out

    def _to_spoken_intimate(self, text: str) -> str:
        t = text
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"\.{3,}", "...", t)
        t = re.sub(r"\b(kk+|ha(ha)+|rs+)\b", "", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"\b(amor|querido|beb[eê])\b\s*", r"\1... ", t, flags=re.I, count=1)
        return t.strip()

    def clean_for_speech(self, text: str) -> str:
        return self._to_spoken_intimate(self._pick_intimate_excerpt(text))

    def _wrap_ssml(self, text: str) -> str:
        # XML escape
        safe = text.replace("&", "&" + "amp;").replace("<", "&" + "lt;").replace(">", "&" + "gt;")
        safe = safe.replace("...", '<break time="400ms"/>')
        safe = safe.replace("\u2026", '<break time="400ms"/>')
        if self.style in ("intimate", "sexy", "soft"):
            return (
                "<speak version='1.0' xml:lang='pt-BR'>"
                "<prosody rate='slow' pitch='-2st' volume='soft'>"
                f"{safe}"
                "</prosody></speak>"
            )
        return f"<speak version='1.0' xml:lang='pt-BR'>{safe}</speak>"

    async def synthesize(self, text: str) -> bytes | None:
        if not self.enabled:
            return None
        clean = self.clean_for_speech(text)
        if not clean or len(clean) < 3:
            return None
        try:
            import edge_tts
        except Exception as e:
            print(f"[TTS] import fail: {e}", flush=True)
            return None

        speak_text = self._wrap_ssml(clean) if self.use_ssml else clean
        tmp = None
        try:
            communicate = edge_tts.Communicate(
                speak_text,
                self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            await communicate.save(tmp)
            data = Path(tmp).read_bytes()
            if (not data or len(data) < 200) and self.use_ssml:
                print("[TTS] SSML vazio — texto puro", flush=True)
                communicate = edge_tts.Communicate(
                    clean, self.voice, rate=self.rate, pitch=self.pitch
                )
                await communicate.save(tmp)
                data = Path(tmp).read_bytes()
            if not data or len(data) < 200:
                print("[TTS] empty output", flush=True)
                return None
            print(
                f"[TTS] ok voice={self.voice} style={self.style} "
                f"rate={self.rate} pitch={self.pitch} "
                f"chars={len(clean)} bytes={len(data)} excerpt={clean[:60]!r}",
                flush=True,
            )
            return data
        except Exception as e:
            print(f"[TTS] synthesize error: {e}", flush=True)
            if self.use_ssml:
                try:
                    communicate = edge_tts.Communicate(
                        clean, self.voice, rate="-20%", pitch="-4Hz"
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tmp2 = f.name
                    await communicate.save(tmp2)
                    data = Path(tmp2).read_bytes()
                    Path(tmp2).unlink(missing_ok=True)
                    if data and len(data) > 200:
                        return data
                except Exception as e2:
                    print(f"[TTS] fallback error: {e2}", flush=True)
            return None
        finally:
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
