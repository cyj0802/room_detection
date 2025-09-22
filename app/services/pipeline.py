from __future__ import annotations
from pathlib import Path
from typing import Dict
import json, uuid, shutil, os

from dotenv import load_dotenv
load_dotenv()

from app.services.unreal_client import post_json
from app.inference.corners import load_corners_detector
from app.inference.openings import load_openings_detector
from app.inference.rooms import load_rooms_segmentor
from app.inference.furniture import load_furniture_segmentor

from app.services.rooms_ocr import annotate_rooms_with_ocr

BASE = Path("data/jobs")
BASE.mkdir(parents=True, exist_ok=True)

JOBS: Dict[str, dict] = {}  # 데모용 인메모리(추후 DB 대체 가능)

WEIGHTS_CORNERS = os.getenv("WEIGHTS_CORNERS", "app/models/corners.pt")
WEIGHTS_OPENINGS = os.getenv("WEIGHTS_OPENINGS", "app/models/openings.pt")
WEIGHTS_ROOMS = os.getenv("WEIGHTS_ROOMS", "app/models/rooms.pt")
WEIGHTS_FURNITURE = os.getenv("WEIGHTS_FURNITURE", "app/models/furniture.pt")

DEVICE = os.getenv("DEVICE")
UNREAL_WEBHOOK = os.getenv("UNREAL_WEBHOOK")  

# 이미지 저장
def save_upload(fileobj, filename: str):
    job_id = uuid.uuid4().hex[:12]
    job_dir = BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dst = job_dir / filename
    with dst.open("wb") as f:
        shutil.copyfileobj(fileobj, f)
    return job_id, str(dst)

# json 파일 저장
def _save_json(job_id: str, name: str, payload: dict) -> str:
    p = BASE / job_id / name
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)

# 업로드된 이미지에 대해 코너 탐지
async def run_stage_walls(job: dict):
    """
    업로드된 이미지에 대해 코너 탐지 → walls.json 저장 → Unreal에 POST
    """
    # 1) 모델 싱글톤 로드
    detector = load_corners_detector(WEIGHTS_CORNERS, DEVICE)

    # 2) 추론
    res = detector.predict(job["image_path"], conf=0.25, iou=0.45)  # dict

    # 3) 결과 저장
    out_path = _save_json(job["job_id"], "walls.json", res)
    job["results"]["walls_json"] = out_path

    # 4) 상태 전환 + Unreal 웹훅 전송
    job["stage"] = "WALLS_SENT"
    if not UNREAL_WEBHOOK:
        job["log"].append("UNREAL_WEBHOOK not set; skip POST.")
        return
    await post_json(UNREAL_WEBHOOK, {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "payload": res,  # {"type":"corners","items":[...],"meta":{...}}
    })

# 문/창문 탐지
async def run_stage_openings(job: dict):

    detector = load_openings_detector(WEIGHTS_OPENINGS, DEVICE)
    
    res = detector.predict(job["image_path"], conf=0.25, iou=0.45)
    
    out_path = _save_json(job["job_id"], "openings.json", res)
    job["results"]["openings_json"] = out_path
    
    job["stage"] = "OPENINGS_SENT"
    if not UNREAL_WEBHOOK:
            job["log"].append("UNREAL_WEBHOOK not set; skip POST.")
            return
    await post_json(UNREAL_WEBHOOK, {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "payload": res,  # {"type":"openings","items":[...],"meta":{...}}
    })

# 방 탐지
async def run_stage_rooms(job: dict):

    detector = load_rooms_segmentor(WEIGHTS_ROOMS, DEVICE)
    
    res = detector.predict(job["image_path"], conf=0.25, iou=0.45)
    
    out_path = _save_json(job["job_id"], "rooms.json", res)
    job["results"]["rooms_json"] = out_path
    
    job["stage"] = "ROOMS_SENT"
    if not UNREAL_WEBHOOK:
            job["log"].append("UNREAL_WEBHOOK not set; skip POST.")
            return
    await post_json(UNREAL_WEBHOOK, {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "payload": res,  # {"type":"rooms","items":[...],"meta":{...}}
    })

# OCR 방 후처리
async def run_stage_rooms_ocr(job: dict):
    # rooms.json 존재 확인
    rooms_path = job["results"].get("rooms_json")
    if not rooms_path or not Path(rooms_path).exists():
        job["log"].append("rooms.json not found; run rooms stage first.")
        return

    with open(rooms_path, "r", encoding="utf-8") as f:
        rooms_json = json.load(f)

    res = annotate_rooms_with_ocr(job["image_path"], rooms_json)  # dict
    out_path = _save_json(job["job_id"], "rooms_ocr.json", res)
    job["results"]["rooms_ocr_json"] = out_path
    job["stage"] = "ROOMS_OCR_DONE"


# 가구 탐지
async def run_stage_furniture(job: dict):

    detector = load_furniture_segmentor(WEIGHTS_FURNITURE, DEVICE)
    
    res = detector.predict(job["image_path"], conf=0.25, iou=0.45)
    
    out_path = _save_json(job["job_id"], "furniture.json", res)
    job["results"]["furniture_json"] = out_path
    
    job["stage"] = "FURNITURE_SENT"
    if not UNREAL_WEBHOOK:
            job["log"].append("UNREAL_WEBHOOK not set; skip POST.")
            return
    await post_json(UNREAL_WEBHOOK, {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "payload": res,  # {"type":"furniture","items":[...],"meta":{...}}
    })            

async def proceed_next_after_callback(job: dict, confirmed_stage: str):
    if confirmed_stage == "WALLS_SENT":
        job["stage"] = "WALLS_CONFIRMED"
        # TODO: 다음 단계 run_stage_openings(job) 이어붙이기
