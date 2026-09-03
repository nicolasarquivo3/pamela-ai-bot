"""
TTS grátis com edge-tts (vozes Microsoft Edge, pt-BR).
Sem API key. Gera MP3 para o Telegram (send_audio).
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
        rate: str = "+6%",
        pitch: str = "+0Hz",
        max_chars: int = 320,
    ):
        self.voice = (voice or os.getenv("TTS_VOICE") or "pt-BR-FranciscaNeural").strip()
        self.enabled = bool(enabled) and (
            os.getenv("TTS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        self.rate = rate or os.getenv("TTS_RATE", "+6%")
        self.pitch = pitch or os.getenv("TTS_PITCH", "+0Hz")
        self.max_chars = int(max_chars)

    async def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception:
            return False

    def clean_for_speech(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        t = re.sub(r"---\s*IMAGE_PROMPT:.*", "", t, flags=re.S | re.I)
        t = re.sub(r"\[\s*(foto|imagem|photo|selfie)\s*\]", "", t, flags=re.I)
        t = re.sub(r"[*_`#~>]+", "", t)
        t = re.sub(r"https?://\S+", "", t)
        # emojis pesados demais no TTS: reduz repetições
        t = re.sub(r"([❤️😈🔥😏😉😊💋])\1+", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > self.max_chars:
            cut = t[: self.max_chars]
            for sep in (". ", "! ", "? ", "… ", "; "):
                j = cut.rfind(sep)
                if j > 50:
                    cut = cut[: j + 1]
                    break
            t = cut.strip()
        return t

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
        tmp = None
        try:
            communicate = edge_tts.Communicate(
                clean,
                self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            await communicate.save(tmp)
            data = Path(tmp).read_bytes()
            if not data or len(data) < 200:
                print("[TTS] empty output", flush=True)
                return None
            print(
                f"[TTS] ok voice={self.voice} chars={len(clean)} bytes={len(data)}",
                flush=True,
            )
            return data
        except Exception as e:
            print(f"[TTS] synthesize error: {e}", flush=True)
            return None
        finally:
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
