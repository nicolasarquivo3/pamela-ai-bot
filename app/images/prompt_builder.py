class PromptBuilder:
    NEGATIVE_PROMPT = "deformed anatomy, extra fingers, extra limbs, duplicate person, distorted face, plastic skin, cartoon, anime, illustration, watermark, text, logo"

    def build(self, character, request):
        identity = character.image_identity if character else {}
        identity_text = "\n".join(f"{k}: {v}" for k, v in identity.items() if v)
        parts = [
            "Photorealistic adult woman.",
            "Natural adult proportions, natural skin texture, realistic lighting, candid real-world photography.",
            "Consistent facial identity.",
            "CHARACTER:",
            identity_text,
            "SCENE:",
            request.scene,
            "STYLE:",
            request.style,
            "Do not depict explicit sexual activity or nudity.",
        ]
        return "\n".join(parts)
