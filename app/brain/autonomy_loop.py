"""
Loop interno de autonomia — nao depende de cron externo.
Chame start_autonomy_loop(agent) no main apos criar o agent.
"""
from __future__ import annotations

import asyncio


async def autonomy_loop(agent, interval_seconds: int = 600):
    """
    A cada interval_seconds chama autonomous_tick.
    Default 10 min.
    """
    print(
        f"[AutonomyLoop] iniciado interval={interval_seconds}s",
        flush=True,
    )
    # espera um pouco no boot para app subir
    await asyncio.sleep(45)
    while True:
        try:
            if getattr(agent, "autonomy_service", None):
                result = await agent.autonomous_tick()
                print(f"[AutonomyLoop] result={result}", flush=True)
            else:
                print("[AutonomyLoop] autonomy_service ausente", flush=True)
        except Exception as e:
            print(f"[AutonomyLoop] erro: {e}", flush=True)
        await asyncio.sleep(max(60, int(interval_seconds)))


def start_autonomy_loop(agent, interval_seconds: int = 600):
    return asyncio.create_task(autonomy_loop(agent, interval_seconds))
