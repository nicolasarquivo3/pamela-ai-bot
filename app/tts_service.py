"""
TTS gratis edge-tts (pt-BR).
So FALA intima no telefone — nunca narração de RP.
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
        max_chars: int = 140,
        style: str | None = None,
    ):
        self.voice = (voice or os.getenv("TTS_VOICE") or "pt-BR-FranciscaNeural").strip()
        self.enabled = bool(enabled) and (
            os.getenv("TTS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        self.rate = (rate or os.getenv("TTS_RATE") or "-25%").strip()
        self.pitch = (pitch or os.getenv("TTS_PITCH") or "-6Hz").strip()
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
        t = re.sub(r"\*+[^*]+\*+", " ", t)
        t = re.sub(r"_+[^_]+_+", " ", t)
        t = re.sub(r"[*_`#~>]+", "", t)
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E0-\U0001F1FF]+",
            "",
            t,
        )
        for e in ("❤️", "♥", "❤", "🔥", "😈", "😏", "💋", "🥵", "😉", "😊", "😅"):
            t = t.replace(e, "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _extract_after_speech_cue(self, text: str) -> list[str]:
        """
        'gemendo baixinho: quer me ouvir?' -> 'quer me ouvir?'
        'te falo: amor, vem' -> 'amor, vem'
        """
        out = []
        for m in re.finditer(
            r"(?::|\bfalo\b|\bdigo\b|\bsussurro\b|\bgemendo\b|\brespondo\b)\s*[:\-]?\s*[\"“']?([^\"”'\n]{8,160})",
            text,
            flags=re.I,
        ):
            frag = m.group(1).strip(" .")
            if frag:
                out.append(frag)
        # aspas
        for m in re.finditer(r"[\"“']([^\"”']{8,160})[\"”']", text):
            out.append(m.group(1).strip())
        return out

    def _is_narration(self, s: str) -> bool:
        low = s.strip().lower()
        if re.match(
            r"^(pego|abro|fecho|caminho|saio|entro|olho|mordo|desço|desco|subo|"
            r"tiro|coloco|ajusto|mexo|dou|vou|corro|paro|ligo|envio|digito|"
            r"escrevo|gravo|seguro|ele |ela |a m[aã]o)",
            low,
        ):
            return True
        if re.search(
            r"\b(com os dedos|dedos tremendo|pego o celular|no espelho|"
            r"tiro uma foto|ab[ro]o a c[aâ]mera|caminho at[eé])\b",
            low,
        ):
            return True
        return False

    def _is_spoken(self, s: str) -> bool:
        low = s.lower().strip()
        if len(low) < 6:
            return False
        if self._is_narration(s):
            return False
        # lixo residual
        if re.match(r"^[,.\-\s]+", low):
            return False
        if re.search(
            r"\b(amor|querido|beb[eê]|voc[eê]|te quero|t[oô] |estou |"
            r"sinto|pensa|imagina|ouve|escuta|vem |fica |me faz|me deixa|"
            r"gostoso|safad|tes[aã]o|molhad|calcinha|pra voc|quer me|"
            r"mais forte|n[aã]o para|assim)\b",
            low,
        ):
            return True
        if low.endswith("?") and re.search(r"\b(voc[eê]|quer|gosta|topa|me)\b", low):
            return True
        return False

    def _fallback_whisper(self, original: str) -> str:
        low = (original or "").lower()
        if re.search(r"\b(transand|sexo|foder|fode|goz|gemid|boquete|pau|mais forte)\b", low):
            return "Amor... assim... não para..."
        if re.search(r"\b(calcinha|saia|sem roupa|nu[ae]|lingerie|micro)\b", low):
            return "Amor... tô quase sem nada... só pra você..."
        if re.search(r"\b(balada|festa|danc|danç|pista)\b", low):
            return "Amor... tô na pista... pensando em você..."
        if re.search(r"\b(saudade|cama|colo|beijo)\b", low):
            return "Amor... tô com saudade do seu colo..."
        if re.search(r"\b(provoc|safad|tes[aã]o|ouvir|celular)\b", low):
            return "Amor... quer me ouvir assim... baixinho?"
        return "Amor... tô aqui... pensando em você..."

    def _polish(self, text: str) -> str:
        t = text.strip()
        t = re.sub(r"\b(meu deus!?|nossa!?|kk+|ha(ha)+|rs+)\b", "", t, flags=re.I)
        t = re.sub(r"!{2,}", ".", t)
        t = re.sub(r"\?{2,}", "?", t)
        t = re.sub(r"\.{2,}", "...", t)
        t = re.sub(r"(\.\.\.)\s*\1+", r"...", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"\s+([,\.!\?])", r"\1", t)
        t = t.strip(" ,;.")
        # se comeca com lixo
        t = re.sub(r"^[,.\-\s]+", "", t)
        # amor com pausa
        if re.match(r"(?i)amor\b", t) and not t.lower().startswith("amor..."):
            t = re.sub(r"(?i)^amor\b\s*", "Amor... ", t, count=1)
        elif re.search(r"(?i)\bamor\b", t) and "..." not in t[:24]:
            t = re.sub(r"(?i)\bamor\b", "amor...", t, count=1)
        # capitaliza
        if t and t[0].islower():
            t = t[0].upper() + t[1:]
        return t.strip()

    def clean_for_speech(self, text: str) -> str:
        raw = self._strip_noise(text)
        candidates: list[str] = []

        # 1) dialogo depois de : / gemendo / falo
        candidates.extend(self._extract_after_speech_cue(raw))

        # 2) frases
        parts = re.split(r"(?<=[\.\!\?\…])\s+|\n+", raw)
        for p in parts:
            p2 = p.strip().strip("\"'“”‘’").strip(" -–—")
            if self._is_spoken(p2):
                candidates.append(p2)

        # 3) se frase misturou narracao + fala, tenta cortar no ":" 
        for p in parts:
            if ":" in p:
                after = p.split(":", 1)[-1].strip().strip("\"'“”")
                if self._is_spoken(after) or (len(after) > 8 and re.search(r"\b(voc[eê]|amor|quer)\b", after, re.I)):
                    candidates.append(after)

        # filtra narracao e lixo curto
        good = []
        for c in candidates:
            c = self._polish(c)
            if len(c) < 8:
                continue
            if self._is_narration(c):
                continue
            if re.search(r"\b(pego o|dedos tremendo|celular|escrevo|caminho)\b", c, re.I):
                continue
            good.append(c)

        # prefere pergunta intima ou confissao curta
        def score(s: str) -> tuple:
            low = s.lower()
            sc = 0
            if re.search(r"\b(quer me|me ouvir|tes[aã]o|calcinha|mais forte|não para|nao para)\b", low):
                sc += 5
            if s.endswith("?"):
                sc += 3
            if 20 <= len(s) <= 110:
                sc += 2
            if self._is_narration(s):
                sc -= 10
            return (-sc, len(s))

        if good:
            good.sort(key=score)
            out = good[0]
        else:
            out = self._polish(self._fallback_whisper(text))

        if len(out) > self.max_chars:
            cut = out[: self.max_chars]
            for sep in ("... ", ". ", "! ", "? ", ", "):
                j = cut.rfind(sep)
                if j > 25:
                    cut = cut[: j + len(sep.rstrip())]
                    break
            out = cut.strip()

        out = self._polish(out)
        if len(out) < 8 or self._is_narration(out):
            out = self._polish(self._fallback_whisper(text))
        return out

    def _wrap_ssml(self, text: str) -> str:
        safe = (
            text.replace("&", "&" + "amp;")
            .replace("<", "&" + "lt;")
            .replace(">", "&" + "gt;")
        )
        safe = safe.replace("...", '<break time="550ms"/>')
        safe = safe.replace("\u2026", '<break time="550ms"/>')
        safe = re.sub(r",\s*", ', <break time="200ms"/> ', safe)
        return (
            "<speak version='1.0' xml:lang='pt-BR'>"
            "<prosody rate='slow' pitch='-3st' volume='soft'>"
            f"{safe}"
            "</prosody></speak>"
        )

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
                speak_text, self.voice, rate=self.rate, pitch=self.pitch
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
                f"[TTS] ok voice={self.voice} rate={self.rate} pitch={self.pitch} "
                f"spoken={clean!r} bytes={len(data)}",
                flush=True,
            )
            return data
        except Exception as e:
            print(f"[TTS] synthesize error: {e}", flush=True)
            try:
                communicate = edge_tts.Communicate(
                    clean, self.voice, rate="-25%", pitch="-6Hz"
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
