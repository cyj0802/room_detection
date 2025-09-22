# app/inference/furniture.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from app.inference.yolo_utils import predict_consistent_model, select_device

class FurnitureDetector:
    def __init__(self, weights: str | Path, device: Optional[str] = None):
        self.weights = str(weights)
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`.")
        self.model = YOLO(self.weights)
        self.device = select_device(device)

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
            agnostic_nms=agnostic_nms, retina_masks=True, 
            max_det=max_det, augment=False, verbose=False,
        )

        r = results[0]
        names = r.names
        items: List[Dict[str, Any]] = []

        if getattr(r, "masks", None) is not None and r.masks is not None:
            polys_list = r.masks.xy  # list[np.ndarray of shape (N_i, 2)]
            cls_list = r.boxes.cls.int().cpu().numpy().tolist() if r.boxes is not None else [0] * len(polys_list)
            for poly, cid in zip(polys_list, cls_list):
                label = names.get(cid, str(cid)) if isinstance(names, dict) else str(names[cid])
                pts = [[int(x), int(y)] for x, y in poly.tolist()]
                items.append({"class": label, "poly": pts})

        elif r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.int().cpu().numpy().tolist()
            cls = r.boxes.cls.int().cpu().numpy().tolist()
            for (x1, y1, x2, y2), cid in zip(xyxy, cls):
                label = names.get(cid, str(cid)) if isinstance(names, dict) else str(names[cid])
                rect = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                items.append({"class": label, "poly": rect})

        return {"type": "furniture", "items": items, "meta": {"image": Path(image_path).name}}

_furniture_singleton: Optional[FurnitureDetector] = None
def load_furniture_segmentor(weights: str | Path, device: Optional[str] = None) -> FurnitureDetector:
    global _furniture_singleton
    if _furniture_singleton is None:
        _furniture_singleton = FurnitureDetector(weights, device)
    return _furniture_singleton
