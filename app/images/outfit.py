"""Extrai e memoriza ROUPA (nao local). Local vai para a scene/query separado."""
from __future__ import annotations

import re
from typing import Any

# Apenas ROUPA / acessorios (NUNCA local/acao)
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
    (r"\bvestido\s+de\s+paet[eÃª]\b", "sequin micro mini dress"),
    (r"\bvestido\s+justo\b", "tight bodycon mini dress"),
    (r"\bvestido\s+bodycon\b", "bodycon mini dress"),
    (r"\bvestido\s+de\s+festa\b", "party sequin mini dress"),
    (r"\bpaet[eÃª]\b", "sequin"),
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
    (r"\bsuti[aÃ£]n?\b", "lingerie bra"),
    (r"\bbiqu[iÃ­]ni\b", "bikini"),
    (r"\bmai[oÃ´]\b", "swimsuit"),
    (r"\bshorts\b", "short shorts"),
    (r"\bcal[cÃ§]a\s+jeans\b", "jeans"),
    (r"\bcal[cÃ§]a\b", "pants"),
    (r"\bmacac[aÃ£]o\b", "jumpsuit"),
    (r"\bbody\b", "bodysuit"),
    (r"\bmeia[\-\s]?cal[cÃ§]a\b", "pantyhose stockings"),
    (r"\bmeia\s+arrast[aÃ£]o\b", "fishnet stockings"),
    (r"\bsalto\s+alto\b", "high heels"),
    (r"\bsalto\b", "high heels"),
    (r"\bscarpin\b", "high heel pumps"),
    (r"\bbota\s+over\b", "thigh high boots"),
    (r"\bbota\b", "boots"),
    (r"\bt[eÃª]nis\b", "sneakers"),
    (r"\bpreto\b|\bpreta\b", "black"),
    (r"\bbranco\b|\bbranca\b", "white"),
    (r"\bvermelh[oa]\b", "red"),
    (r"\bazul\b", "blue"),
    (r"\brosa\b", "pink"),
    (r"\bdourad[oa]\b", "gold sequin"),
    (r"\bpratead[oa]\b", "silver sequin"),
]

# Local / cena (NUNCA salva como roupa)
LOCATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bbalada\b|\bfesta\b|\bclub\b|\bboate\b", "night club party lights"),
    (r"\bpraia\b|\bmar\b", "beach sunny outdoor"),
    (r"\bacademia\b|\bgym\b|\btreino\b", "gym fitness"),
    (r"\bquarto\b|\bcama\b", "bedroom soft light"),
    (r"\bespelho\b|\bselfie\b", "mirror selfie"),
    (r"\bcarr?o\b|\bdirig", "in car selfie"),
    (r"\bruas?\b|\brua\b|\bstreet\b", "street style outdoor"),
    (r"\bbanheiro\b|\bbanho\b", "bathroom mirror selfie"),
    (r"\bcasa\b|\bsofa\b|\bsala\b", "home living room"),
    (r"\bbar\b", "bar night lights"),
]

# Palavras que nunca devem ser "outfit" sozinhas
_NON_CLOTHING = {
    "night club party",
    "night club party lights",
    "beach",
    "beach sunny outdoor",
    "gym fitness",
    "bedroom",
    "bedroom soft light",
    "mirror selfie",
    "in car selfie",
    "street style outdoor",
    "bathroom mirror selfie",
    "home living room",
    "bar night lights",
    "fashion",
    "portrait",
    "photo",
}

DEFAULT_OUTFIT = "micro mini dress high heels"
_CURRENT: dict[str, str] = {}


def _key(user_id: Any, character_id: Any) -> str:
    return f"{user_id}:{character_id}"


def _clean_outfit_core(core: str) -> str:
    core = (core or "").strip()
    core = re.sub(r"\b(fashion|portrait|photo)\b", " ", core, flags=re.I)
    core = re.sub(r"\s+", " ", core).strip()
    low = core.lower()
    # se so local, descarta
    if low in _NON_CLOTHING or not core:
        return ""
    # remove tokens de local do core
    for loc in list(_NON_CLOTHING):
        core = re.sub(re.escape(loc), " ", core, flags=re.I)
    core = re.sub(r"\s+", " ", core).strip()
    if core.lower() in _NON_CLOTHING or len(core) < 4:
        return ""
    return core


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
        core = _clean_outfit_core(m3.group(1))
        if core:
            return [core]

    for bad in (
        "Criar uma fotografia", "fotografia espontÃ¢nea", "personagem PÃ¢mela",
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
        if eng not in found and eng.lower() not in _NON_CLOTHING:
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


def extract_location_bits(text: str) -> list[str]:
    raw = (text or "").lower()
    found: list[str] = []
    for pat, eng in LOCATION_PATTERNS:
        if re.search(pat, raw, re.I) and eng not in found:
            found.append(eng)
    return found[:2]


def bits_to_query(bits: list[str], *, force_default_if_empty: bool = True) -> str:
    clean = [_clean_outfit_core(b) for b in (bits or [])]
    clean = [b for b in clean if b]
    if not clean:
        return f"sexy young woman {DEFAULT_OUTFIT}" if force_default_if_empty else ""
    return f"sexy young woman {' '.join(clean[:5])}"


def get_current_outfit(user_id: Any, character_id: Any) -> str | None:
    cur = _CURRENT.get(_key(user_id, character_id))
    if not cur:
        return None
    cleaned = _clean_outfit_core(cur)
    return cleaned or None


def set_current_outfit(user_id: Any, character_id: Any, outfit_en: str) -> None:
    outfit_en = _clean_outfit_core(outfit_en or "")
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
        r"mesma\s+roupa|roupa\s+que\s+(voc[eÃª]|vc)\s+est|como\s+est[aÃ¡]\s+vest|"
        r"ainda\s+com\s+(a\s+)?roupa|nessa\s+roupa|com\s+essa\s+roupa",
        user_text or "",
        re.I,
    ):
        cur = get_current_outfit(user_id, character_id) if user_id is not None else None
        if cur:
            return f"sexy young woman {cur}", "memory"

    cur = get_current_outfit(user_id, character_id) if user_id is not None else None
    if cur:
        return f"sexy young woman {cur}", "memory"

    return f"sexy young woman {DEFAULT_OUTFIT}", "default"


def build_image_scene(user_text: str, user_id: Any = None, character_id: Any = None) -> str:
    outfit_q, source = resolve_outfit(user_text, user_id, character_id)
    outfit_core = (
        outfit_q.replace("sexy young woman ", "")
        .strip()
    )
    outfit_core = _clean_outfit_core(outfit_core) or DEFAULT_OUTFIT
    locs = extract_location_bits(user_text or "")
    loc_part = f" | LOC: {' '.join(locs)}" if locs else ""
    pedido = (user_text or "").strip()[:200]
    scene = f"OUTFIT: {outfit_core}{loc_part} | PEDIDO: {pedido} | source={source}"
    print(f"[OUTFIT] scene source={source} outfit={outfit_core!r} loc={locs}", flush=True)
    return scene


def outfit_from_scene(scene: str) -> str:
    """Base query de roupa a partir da scene (sem local misturado)."""
    scene = scene or ""
    m = re.search(r"OUTFIT:\s*([^|]+)", scene, re.I)
    if m:
        core = _clean_outfit_core(m.group(1))
        if core:
            return f"sexy young woman {core}"
    return bits_to_query(extract_outfit_bits(scene), force_default_if_empty=True)


def location_from_scene(scene: str) -> str:
    scene = scene or ""
    m = re.search(r"LOC:\s*([^|]+)", scene, re.I)
    if m:
        return m.group(1).strip()
    bits = extract_location_bits(scene)
    return " ".join(bits[:2]) if bits else ""
