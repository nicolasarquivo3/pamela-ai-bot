"""Extrai e memoriza roupa. Prioridade: pedido > memoria > default."""
from __future__ import annotations

import re
from typing import Any

CLOTHING_PATTERNS: list[tuple[str, str]] = [
    (r"\bvestido\s+longo\b", "long elegant dress"),
    (r"\bvestido\s+curto\b", "short mini dress"),
    (r"\bmicro\s*vestido\b", "micro mini dress"),
    (r"\bvestido\s+preto\b", "black micro mini dress"),
    (r"\bvestido\s+branco\b", "white micro mini dress"),
    (r"\bvestido\s+vermelho\b", "red micro mini dress"),
    (r"\bvestido\s+azul\b", "blue micro mini dress"),
    (r"\bvestido\s+rosa\b", "pink mini dress"),
    (r"\bvestido\s+de\s+renda\b", "lace micro mini dress"),
    (r"\bvestido\s+de\s+paet[eê]\b", "sequin micro mini dress"),
    (r"\bvestido\s+justo\b", "tight bodycon mini dress"),
    (r"\bvestido\s+bodycon\b", "bodycon mini dress"),
    (r"\bvestido\s+de\s+festa\b", "party mini dress"),
    (r"\bpaet[eê]\b", "sequin"),
    (r"\bvestido\b", "micro mini dress"),
    (r"\bminissaia\s+jeans\b", "denim micro mini skirt"),
    (r"\bminissaia\s+preta\b", "black micro mini skirt"),
    (r"\bminissaia\b", "micro mini skirt"),
    (r"\bsaia\s+jeans\b", "denim micro mini skirt"),
    (r"\bsaia\s+preta\b", "black micro mini skirt"),
    (r"\bsaia\s+curta\b", "micro mini skirt"),
    (r"\bsaia\s+de\s+couro\b", "leather mini skirt"),
    (r"\bmicro\s*saia\b", "micro mini skirt"),
    (r"\bsaia\b", "micro mini skirt"),
    (r"\bcropped\b", "crop top midriff"),
    (r"\bcrop\s*top\b", "crop top midriff"),
    (r"\bblusa\s+curta\b", "crop top"),
    (r"\bblusa\b", "blouse top"),
    (r"\bjaqueta\s+de\s+couro\b", "leather jacket"),
    (r"\bjaqueta\b", "jacket"),
    (r"\bcouro\b", "leather"),
    (r"\blingerie\b", "sexy lingerie"),
    (r"\blina\b", "sexy lingerie"),
    (r"\bcalcinha\b", "lingerie"),
    (r"\bsuti[aã]n?\b", "lingerie bra"),
    (r"\bbiqu[ií]ni\b", "bikini"),
    (r"\bmai[oô]\b", "swimsuit"),
    (r"\bshorts\b", "short shorts"),
    (r"\bcal[cç]a\s+jeans\b", "jeans"),
    (r"\bcal[cç]a\b", "pants"),
    (r"\bmacac[aã]o\b", "jumpsuit"),
    (r"\bbody\b", "bodysuit"),
    (r"\bmeia[\-\s]?cal[cç]a\b", "pantyhose stockings"),
    (r"\bmeia\s+arrast[aã]o\b", "fishnet stockings"),
    (r"\bsalto\s+alto\b", "high heels"),
    (r"\bsalto\b", "high heels"),
    (r"\bscarpin\b", "high heel pumps"),
    (r"\bbota\s+over\b", "thigh high boots"),
    (r"\bbota\b", "boots"),
    (r"\bt[eê]nis\b", "sneakers"),
    (r"\bpreto\b|\bpreta\b", "black"),
    (r"\bbranco\b|\bbranca\b", "white"),
    (r"\bvermelh[oa]\b", "red"),
    (r"\bazul\b", "blue"),
    (r"\brosa\b", "pink"),
    (r"\bdourad[oa]\b", "gold sequin"),
    (r"\bpratead[oa]\b", "silver sequin"),
    (r"\bbalada\b|\bfesta\b|\bclub\b", "night club party"),
    (r"\bpraia\b", "beach"),
    (r"\bacademia\b|\bgym\b", "gym fitness"),
    (r"\bquarto\b|\bcama\b", "bedroom"),
    (r"\bespelho\b|\bselfie\b", "mirror selfie"),
]

DEFAULT_OUTFIT = "micro mini dress high heels fashion"
_CURRENT: dict[str, str] = {}


def _key(user_id: Any, character_id: Any) -> str:
    return f"{user_id}:{character_id}"


def extract_outfit_bits(text: str) -> list[str]:
    raw = (text or "").strip()
    m = re.search(r"contexto da fotografia:\s*(.+)", raw, re.I | re.S)
    if m:
        raw = m.group(1)
    m2 = re.search(r"roupa(?:\s+e\s+pose)?(?:\s+desta\s+vez)?\s*:\s*(.+)", raw, re.I)
    if m2:
        raw = m2.group(1)
    m3 = re.search(r"OUTFIT:\s*([^|]+)", raw, re.I)
    if m3:
        core = m3.group(1).strip()
        if core and "source=" not in core:
            return [core]

    for bad in (
        "Criar uma fotografia", "fotografia espontânea", "personagem Pâmela",
        "personagem Pamela", "mulher adulta", "identidade visual",
        "Interpretar o pedido", "Preservar os detalhes",
        "me manda uma foto", "manda uma foto", "tira uma foto", "selfie",
    ):
        raw = re.sub(re.escape(bad), " ", raw, flags=re.I)

    found: list[str] = []
    matched_spans: list[tuple[int, int]] = []
    for pat, eng in CLOTHING_PATTERNS:
        m = re.search(pat, raw, re.I)
        if not m:
            continue
        span = m.span()
        if any(span[0] >= a and span[1] <= b for a, b in matched_spans):
            continue
        matched_spans = [(a, b) for a, b in matched_spans if not (a >= span[0] and b <= span[1])]
        found = [f for f in found if f not in eng and eng not in f]
        if eng not in found:
            found.append(eng)
            matched_spans.append(span)

    colors = {"black", "white", "red", "blue", "pink", "gold sequin", "silver sequin"}
    if any(any(c in f for c in colors) and f not in colors for f in found):
        found = [f for f in found if f not in colors]
    if any("dress" in f and f != "micro mini dress" for f in found):
        found = [f for f in found if f != "micro mini dress"]
    if any("skirt" in f and f != "micro mini skirt" for f in found):
        found = [f for f in found if f != "micro mini skirt"]
    return found[:6]


def bits_to_query(bits: list[str], *, force_default_if_empty: bool = True) -> str:
    if not bits:
        return f"sexy young woman {DEFAULT_OUTFIT} portrait" if force_default_if_empty else ""
    return f"sexy young woman {' '.join(bits[:6])} fashion portrait photo"


def get_current_outfit(user_id: Any, character_id: Any) -> str | None:
    return _CURRENT.get(_key(user_id, character_id))


def set_current_outfit(user_id: Any, character_id: Any, outfit_en: str) -> None:
    outfit_en = (outfit_en or "").strip()
    if not outfit_en:
        return
    _CURRENT[_key(user_id, character_id)] = outfit_en
    print(f"[OUTFIT] saved {user_id}/{character_id}: {outfit_en}", flush=True)


def resolve_outfit(user_text: str, user_id: Any = None, character_id: Any = None):
    bits = extract_outfit_bits(user_text)
    if bits:
        q = bits_to_query(bits, force_default_if_empty=False)
        if user_id is not None and character_id is not None:
            set_current_outfit(user_id, character_id, " ".join(bits[:6]))
        return q, "user"

    if re.search(
        r"mesma\s+roupa|roupa\s+que\s+(voc[eê]|vc)\s+est|como\s+est[aá]\s+vest|"
        r"ainda\s+com\s+(a\s+)?roupa|nessa\s+roupa|com\s+essa\s+roupa",
        user_text or "",
        re.I,
    ):
        cur = get_current_outfit(user_id, character_id) if user_id is not None else None
        if cur:
            return f"sexy young woman {cur} fashion portrait photo", "memory"

    cur = get_current_outfit(user_id, character_id) if user_id is not None else None
    if cur:
        return f"sexy young woman {cur} fashion portrait photo", "memory"

    return f"sexy young woman {DEFAULT_OUTFIT} portrait", "default"


def build_image_scene(user_text: str, user_id: Any = None, character_id: Any = None) -> str:
    outfit_q, source = resolve_outfit(user_text, user_id, character_id)
    outfit_core = (
        outfit_q.replace("sexy young woman ", "")
        .replace(" fashion portrait photo", "")
        .replace(" portrait", "")
        .strip()
    )
    pedido = (user_text or "").strip()[:200]
    scene = f"OUTFIT: {outfit_core} | PEDIDO: {pedido} | source={source}"
    print(f"[OUTFIT] scene source={source} outfit={outfit_core!r}", flush=True)
    return scene


def outfit_from_scene(scene: str) -> str:
    scene = scene or ""
    m = re.search(r"OUTFIT:\s*([^|]+)", scene, re.I)
    if m:
        core = m.group(1).strip()
        if core:
            return f"sexy young woman {core} fashion portrait photo"
    return bits_to_query(extract_outfit_bits(scene), force_default_if_empty=True)
