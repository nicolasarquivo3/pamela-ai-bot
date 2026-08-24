class PromptBuilder:
    NEGATIVE_PROMPT = (
        "deformed anatomy, bad anatomy, malformed hands, malformed feet, "
        "extra fingers, missing fingers, extra limbs, missing limbs, "
        "duplicate person, duplicated body parts, distorted face, "
        "asymmetrical eyes, unnatural eyes, deformed teeth, "
        "plastic skin, waxy skin, artificial skin, oversmoothed skin, "
        "cartoon, anime, illustration, 3d render, CGI, doll-like appearance, "
        "low resolution, blurry, pixelated, noisy image, "
        "watermark, text, logo, signature, frame, cropped head, cropped feet"
    )

    def build(self, character, request):
        identity = character.image_identity if character else {}

        identity_lines = []

        for key, value in identity.items():
            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                value = ", ".join(str(item) for item in value)

            elif isinstance(value, dict):
                value = ", ".join(
                    f"{sub_key}: {sub_value}"
                    for sub_key, sub_value in value.items()
                    if sub_value is not None
                )

            value = str(value).strip()

            if value:
                identity_lines.append(f"{key}: {value}")

        identity_text = "\n".join(identity_lines)

        scene = (request.scene or "").strip()
        style = (request.style or "").strip()

        parts = [
            "PHOTOREALISTIC ADULT CHARACTER",
            (
                "Create a photorealistic image of the same fictional adult "
                "female character described below."
            ),
            (
                "She is an adult woman. Preserve her established facial "
                "identity and visual characteristics consistently."
            ),
            (
                "Natural adult human proportions, realistic body structure, "
                "natural skin pores and texture, subtle skin imperfections, "
                "realistic hair strands, realistic eyes, realistic hands, "
                "realistic lighting and physically plausible shadows."
            ),
            (
                "The image should look like a real photograph taken with a "
                "modern professional camera, not like digital art or CGI."
            ),
            (
                "Maintain visual continuity with previous images of the "
                "character whenever identity information is available."
            ),
        ]

        if identity_text:
            parts.extend(
                [
                    "",
                    "CHARACTER IDENTITY:",
                    identity_text,
                ]
            )

        if scene:
            parts.extend(
                [
                    "",
                    "REQUESTED SCENE:",
                    scene,
                ]
            )

        if style:
            parts.extend(
                [
                    "",
                    "PHOTOGRAPHY / VISUAL STYLE:",
                    style,
                ]
            )

        parts.extend(
            [
                "",
                "COMPOSITION:",
                (
                    "Prioritize a natural photographic composition. "
                    "Keep the character clearly visible and anatomically "
                    "complete whenever the requested framing allows it."
                ),
                (
                    "Respect the requested pose, clothing, location, "
                    "camera angle and framing."
                ),
                "",
                "SAFETY:",
                (
                    "The character is an adult. Do not depict minors. "
                    "Do not depict nudity or explicit sexual activity."
                ),
            ]
        )

        return "\n".join(parts)
