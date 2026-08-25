"""Prompt AI priorizando OUTFIT / pedido do usuario."""
from __future__ import annotations

import re
from typing import Any

try:
    from app.images.outfit import outfit_from_scene, extract_outfit_bits
except Exception:
    outfit_from_scene = None  # type: ignore
    extract_outfit_bits = None  # type: ignore


class PromptBuilder:
    NEGATIVE_PROMPT = (
        "deformed, ugly, bad anatomy, bad hands, extra fingers, missing fingers, "
        "mutated hands, poorly drawn face, mutation, blurry, low quality, "
        "cartoon, anime, drawing, painting, illustration, 3d render, cgi, "
        "child, minor, underage, baby"
    )

    def build(self, character: Any, request: Any) -> str:
        identity_lines: list[str] = []
        if character is not None:
            data = character if isinstance(character, dict) else None
            if data is None and hasattr(character, "__dict__"):
                data = {
                    k: getattr(character, k, None)
                    for k in (
                        "name", "age", "appearance", "hair", "eyes", "body",
                        "style", "personality", "description",
                        "visual_description", "face_description",
                    )
                    if getattr(character, k, None)
                }
            elif data is None:
                data = {}
            for key, value in (data or {}).items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value = ", ".join(
                        f"{sk}: {sv}" for sk, sv in value.items() if sv is not None
                    )
                value = str(value).strip()
                if value:
                    identity_lines.append(f"{key}: {value}")

        identity_text = "\n".join(identity_lines)
        scene = (getattr(request, "scene", None) or "").strip()
        style = (getattr(request, "style", None) or "").strip() or "photorealistic selfie photo"

        outfit_line = ""
        if outfit_from_scene and scene:
            try:
                oq = outfit_from_scene(scene)
                outfit_line = oq.replace("sexy young woman ", "").strip()
            except Exception:
                outfit_line = ""
        if not outfit_line and extract_outfit_bits:
            bits = extract_outfit_bits(scene)
            if bits:
                outfit_line = " ".join(bits)

        scene_clean = scene
        m = re.search(r"PEDIDO:\s*([^|]+)", scene, re.I)
        if m:
            scene_clean = m.group(1).strip()
        scene_clean = re.sub(r"OUTFIT:\s*[^|]+\|?", " ", scene_clean)
        scene_clean = re.sub(r"source=\w+", " ", scene_clean)
        scene_clean = re.sub(r"\s+", " ", scene_clean).strip()

        parts = [
            "PHOTOREALISTIC adult woman, realistic photo, DSLR, natural skin texture,",
            "same fictional adult female character, consistent face identity,",
            "full body or mirror selfie as appropriate, sharp focus, realistic lighting.",
        ]
        if identity_text:
            parts.append("CHARACTER IDENTITY:")
            parts.append(identity_text)
        if outfit_line:
            parts.append(f"CLOTHING (must match exactly): {outfit_line}")
            parts.append("She is wearing exactly that outfit. Do not change the clothing.")
        if scene_clean:
            parts.append(f"SCENE / USER REQUEST: {scene_clean}")
        parts.append(f"STYLE: {style}, photorealistic, not CGI, not anime, not illustration.")
        parts.append("Natural pose, adult proportions, realistic hands and face.")

        prompt = "\n".join(parts)
        print(f"[PROMPT] outfit={outfit_line!r} scene={scene_clean[:80]!r}", flush=True)
        return prompt
