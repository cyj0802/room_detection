# app/api/jobs.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services.pipeline import save_upload, JOBS, run_stage_walls
from app.services.pipeline import JOBS, run_stage_openings
from app.services.pipeline import JOBS, run_stage_rooms
from app.services.pipeline import JOBS, run_stage_furniture

from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

@router.post("/")
async def create_job(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="2D 도면 이미지 업로드 (jpg/png)"),
):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only png/jpg allowed")

    job_id, image_path = save_upload(file.file, file.filename)

    job = {
        "job_id": job_id,
        "stage": "INIT",
        "image_path": image_path,
        "results": {},
        "log": [],
    }
    JOBS[job_id] = job

    # ★ 업로드 직후 모서리 추론 시작
    background.add_task(run_stage_walls, job)

    return {
        "job_id": job_id,
        "stage": job["stage"],
        "filename": file.filename,
        "path": image_path,
    }

@router.get("/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

# 생성된 모서리 JSON 확인하는 코드 
@router.get("/{job_id}/walls.json")
async def get_walls_json(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("walls_json")
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="walls.json not ready")
    return FileResponse(p, media_type="application/json", filename="walls.json")

# openings 찾기
@router.post("/{job_id}/openings/run")
async def run_openings(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await run_stage_openings(job)
    return {"ok": True, "stage": job["stage"], "openings_json": job["results"].get("openings_json")}

# 생성된 opening json 확인하는 코드
@router.get("/{job_id}/openings.json")
async def get_openings_json(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("openings_json")
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="openings.json not ready")
    return FileResponse(p, media_type="application/json", filename="openings.json")

# rooms 찾기
@router.post("/{job_id}/rooms/run")
async def run_rooms(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await run_stage_rooms(job)
    return {"ok": True, "stage": job["stage"], "rooms_json": job["results"].get("rooms_json")}

# 생성된 rooms json 확인하는 코드
@router.get("/{job_id}/rooms.json")
async def get_rooms_json(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("rooms_json")
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="rooms.json not ready")
    return FileResponse(p, media_type="application/json", filename="rooms.json")

# furniture 찾기
@router.post("/{job_id}/furniture/run")
async def run_furniture(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await run_stage_furniture(job)
    return {"ok": True, "stage": job["stage"], "furniture_json": job["results"].get("furniture_json")}

# 생성된 opening json 확인하는 코드
@router.get("/{job_id}/furniture.json")
async def get_furniture_json(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("furniture_json")
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="furniture.json not ready")
    return FileResponse(p, media_type="application/json", filename="furniture.json")