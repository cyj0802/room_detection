import os, httpx, logging
log = logging.getLogger("unreal")

DISABLE_WEBHOOK = os.getenv("DISABLE_WEBHOOK", "false").lower() == "true"

async def post_json(url: str | None, payload: dict) -> bool:
    if DISABLE_WEBHOOK or not url:
        log.info("[UNREAL] webhook disabled; skipping POST")
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return True
    except Exception as e:
        log.error(f"[UNREAL] webhook error: {e}")
        return False
