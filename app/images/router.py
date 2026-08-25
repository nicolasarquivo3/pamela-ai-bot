from app.images.models import ImageResult
import inspect
import asyncio


class ImageProviderRouter:
    def __init__(self, providers):
        self.providers = providers

    async def _is_available(self, provider) -> bool:
        """Aceita available como método async, método sync ou property bool."""
        try:
            avail = getattr(provider, "available", None)
            if avail is None:
                return True
            if isinstance(avail, bool):
                return avail
            if callable(avail):
                result = avail()
                if inspect.isawaitable(result):
                    return bool(await result)
                return bool(result)
            return bool(avail)
        except Exception as e:
            print(f"[IMAGE] available check failed for {getattr(provider,'name',provider)}: {e}", flush=True)
            return False

    async def generate(self, request, prompt):
        errors = []
        for provider in self.providers:
            if not await self._is_available(provider):
                errors.append(f"{provider.name}:unavailable")
                continue
            try:
                gen = provider.generate
                if inspect.iscoroutinefunction(gen):
                    result = await gen(request, prompt)
                else:
                    result = await asyncio.to_thread(gen, request, prompt)
            except Exception as e:
                errors.append(f"{provider.name}:{e}")
                print(f"[IMAGE] provider {provider.name} error: {e}", flush=True)
                continue

            if result and getattr(result, "success", False):
                return result
            errors.append(f"{provider.name}:{getattr(result, 'error', 'failed')}")
        return ImageResult(False, error="; ".join(errors) or "no_provider_available")
