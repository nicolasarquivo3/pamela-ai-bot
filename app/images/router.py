from app.images.models import ImageResult

class ImageProviderRouter:
    def __init__(self, providers):
        self.providers = providers
    async def generate(self, request, prompt):
        errors = []
        for provider in self.providers:
            if not await provider.available():
                continue
            result = await provider.generate(request, prompt)
            if result.success:
                return result
            errors.append(f"{provider.name}:{result.error}")
        return ImageResult(False, error="; ".join(errors) or "no_provider_available")
