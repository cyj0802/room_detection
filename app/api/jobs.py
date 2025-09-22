from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services.pipeline import (
    save_upload, JOBS,
    run_stage_walls, run_stage_openings, run_stage_rooms, run_stage_furniture,
    run_stage_rooms_ocr
)
from fastapi.responses import (FileResponse, Response)
from pathlib import Path
from app.services.preview_layers import make_preview  
import cv2
import numpy as np
import json
from app.services.pipeline import JOBS
from pathlib import Path


router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

# 업로드된 파일 저장 및 고유 job_id 생성 -> 업로드 직후 자동으로 모서리 추론 시작
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

    background.add_task(run_stage_walls, job)

    return {
        "job_id": job_id,
        "stage": job["stage"],
        "filename": file.filename,
        "path": image_path,
    }

# job id로 현재 진행 중인 작업 상태 확인하는 코드
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

# openings 탐지 실행
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

# rooms 탐지
@router.post("/{job_id}/rooms/run")
async def run_rooms(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="job not found")
    await run_stage_rooms(job)
    return {"ok": True, "stage": job["stage"], "rooms_json": job["results"].get("rooms_json")}

# ocr로 후처리
@router.post("/{job_id}/rooms/ocr/run")
async def run_rooms_ocr(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="job not found")
    await run_stage_rooms_ocr(job)
    return {"ok": True, "stage": job["stage"], "rooms_json": job["results"].get("rooms_json")}

# 생성된 rooms json 확인하는 코드
@router.get("/{job_id}/rooms.json")
async def get_rooms_json(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("rooms_json")
    if not p or not Path(p).exists(): raise HTTPException(status_code=404, detail="rooms.json not ready")
    return FileResponse(p, media_type="application/json", filename="rooms.json")

# furniture 탐지
@router.post("/{job_id}/furniture/run")
async def run_furniture(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="job not found")
    await run_stage_furniture(job)
    return {"ok": True, "stage": job["stage"], "furniture_json": job["results"].get("furniture_json")}

# 생성된 furniture json 확인하는 코드
@router.get("/{job_id}/furniture.json")
async def get_furniture_json(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="job not found")
    p = job["results"].get("furniture_json")
    if not p or not Path(p).exists(): raise HTTPException(status_code=404, detail="furniture.json not ready")
    return FileResponse(p, media_type="application/json", filename="furniture.json")

# json 코드 읽고 파싱하는 함수
def _load_json(p: Path | None):
    if not p or not p.exists():
        return {"items": []}
    return json.loads(p.read_text(encoding="utf-8"))

# json 로드 후 전체 레이어 합성하여 미리보기 png 생성
@router.get("/{job_id}/preview.png")
async def get_preview_png(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    job_dir = Path("data/jobs") / job_id
    walls     = _load_json(job_dir / "walls.json")
    openings  = _load_json(job_dir / "openings.json")
    rooms = _load_json(job_dir / "rooms.json")
    furniture = _load_json(job_dir / "furniture.json")

    img = make_preview(walls, openings, rooms, furniture)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="encode error")
    return Response(content=buf.tobytes(), media_type="image/png")
