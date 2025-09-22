# app/api/callbacks.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.pipeline import JOBS, proceed_next_after_callback

router = APIRouter(prefix="/v1/callbacks", tags=["callbacks"])

class UnrealCallback(BaseModel):
    job_id: str
    stage: str   # ex) "WALLS_SENT"
    status: str  # "ok" or "error"
    message: str | None = None

@router.post("/unreal")
async def unreal_callback(body: UnrealCallback):
    job = JOBS.get(body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if body.status != "ok":
        job["stage"] = "ERROR"
        job["log"].append(f"Unreal error at {body.stage}: {body.message}")
        return {"ack": True}

    await proceed_next_after_callback(job, body.stage)
    return {"ack": True, "next_stage": job["stage"]}
