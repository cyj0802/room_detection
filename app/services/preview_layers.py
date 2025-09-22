# scripts/preview_layers.py
import json
import cv2
import numpy as np
import random
from pathlib import Path
import argparse
import sys

def load_json(p: Path | None):
    if not p or not p.exists():
        return {"items": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def rand_color(seed_key):
    random.seed(str(seed_key))
    return (random.randint(50,255), random.randint(50,255), random.randint(50,255))

def fill_poly_alpha(dst, pts, color, alpha=0.35):
    overlay = dst.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0, dst)

def fill_rect_alpha(dst, x1, y1, x2, y2, color, alpha=0.45, radius=0):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    overlay = dst.copy()
    if radius <= 0:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    else:
        w, h = x2-x1, y2-y1
        r = min(radius, w//2, h//2)
        cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(overlay, (x1+r, y1+r), r, color, -1)
        cv2.circle(overlay, (x2-r, y1+r), r, color, -1)
        cv2.circle(overlay, (x1+r, y2-r), r, color, -1)
        cv2.circle(overlay, (x2-r, y2-r), r, color, -1)
    cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0, dst)

def make_preview(walls: dict, openings: dict, rooms: dict, furniture: dict) -> np.ndarray:
    # ----- 캔버스 크기 산정 -----
    xs, ys = [], []
    for it in walls.get("items", []):
        xs.append(int(it.get("x", 0))); ys.append(int(it.get("y", 0)))
    for src in (openings, furniture):
        for it in src.get("items", []):
            x1, y1, x2, y2 = map(int, it.get("bbox", [0,0,0,0]))
            xs += [x1, x2]; ys += [y1, y2]
    for it in rooms.get("items", []):
        pts = it.get("poly") or it.get("polygon") or []
        for x, y in pts:
            xs.append(int(x)); ys.append(int(y))
    max_x = max(xs) + 20 if xs else 1024
    max_y = max(ys) + 20 if ys else 1024
    canvas = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255

    # ----- 색상 매핑 -----
    ROOM_COLOR = {}
    for it in rooms.get("items", []):
        cls = it.get("class") or it.get("class_name") or "room"
        if cls not in ROOM_COLOR:
            ROOM_COLOR[cls] = rand_color(("r", cls))
    OPENING_COLOR = {"door": (0, 0, 255), "window": (255, 0, 0)}
    FURN_COLOR = {}
    for it in furniture.get("items", []):
        cls = it.get("class") or it.get("class_name") or "furniture"
        if cls not in FURN_COLOR:
            FURN_COLOR[cls] = rand_color(("f", cls))

    # ----- 레이어 그리기 (아래→위) -----
    # 1) corners
    for it in walls.get("items", []):
        x, y = int(it.get("x", 0)), int(it.get("y", 0))
        cls = it.get("class") or it.get("class_name") or "corner"
        cv2.circle(canvas, (x, y), 3, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, cls, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

    # 2) openings
    for it in openings.get("items", []):
        cls = (it.get("class") or it.get("class_name") or "opening").lower()
        x1, y1, x2, y2 = map(int, it.get("bbox", [0,0,0,0]))
        color = OPENING_COLOR.get(cls, rand_color(("o", cls)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, cls, (x1, max(y1-4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 3) rooms
    for it in rooms.get("items", []):
        cls = it.get("class") 
        pts_list = it.get("poly") 
        if not pts_list:
            continue
        pts = np.array(pts_list, dtype=np.int32).reshape((-1,1,2))
        color = ROOM_COLOR.get(cls, rand_color(("r", cls)))
        fill_poly_alpha(canvas, pts, color, alpha=0.30)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(canvas, cls, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, cv2.LINE_AA)

    # 4) furniture
    for it in furniture.get("items", []):
        cls = it.get("class")
        pts_list = it.get("poly") 
        if pts_list and isinstance(pts_list[0], (list, tuple)):
            pts = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))
            color = FURN_COLOR.get(cls, rand_color(("f", cls)))
            # 방처럼 내부 채우고 외곽선 그리고 중심에 라벨
            fill_poly_alpha(canvas, pts, color, alpha=0.45)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(canvas, cls, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20,20,20), 1, cv2.LINE_AA)

    return canvas