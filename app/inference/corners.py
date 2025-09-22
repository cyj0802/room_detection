# app/inference/corners.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from app.inference.yolo_utils import predict_consistent_model, select_device

# 모서리 탐지
class CornersDetector:
    def __init__(self, weights: str | Path, device: Optional[str] = None):
        self.weights = str(weights)
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`.")
        self.model = YOLO(self.weights)
        self.device = select_device(device)

    # 추론    
    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,       
        half: bool = False,
        agnostic_nms: bool = True,
        max_det: int = 1000,
    ) -> Dict[str, Any]:
        image_path = str(image_path)

        results = predict_consistent_model(
            self.model, image_path,
            device=self.device, imgsz=imgsz,
            conf=conf, iou=iou, half=half,
            agnostic_nms=agnostic_nms, retina_masks=False,
            max_det=max_det, augment=False, verbose=False,
        )

        r = results[0]
        names = r.names
        items: List[Dict[str, Any]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().tolist()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cid = int(cls[i])
                label = names.get(cid, str(cid)) if isinstance(names, dict) else str(names[cid])
                items.append({"class": label, "x": cx, "y": cy})

        return {"type": "corners", "items": items, "meta": {"image": Path(image_path).name}}

_detector_singleton: Optional[CornersDetector] = None
def load_corners_detector(weights: str | Path, device: Optional[str] = None) -> CornersDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = CornersDetector(weights, device)
    return _detector_singleton
