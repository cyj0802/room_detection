# app/inference/openings.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class OpeningsDetector:
    def __init__(self, weights: str | Path, device: Optional[str] = None):
        self.weights = str(weights)
        wants_cuda = (device or "").startswith("cuda")
        has_cuda = torch.cuda.is_available()
        self.device = "cuda" if (wants_cuda and has_cuda) else "cpu"

        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`.")
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"openings weights not found: {self.weights}")

        self.model = YOLO(self.weights)
        if self.device == "cuda":
            self.model.to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Output schema:
        {
          "type": "openings",
          "items": [{"class": str, "bbox": [x1,y1,x2,y2]}],
          "meta": {"image": "..."}
        }
        """
        image_path = str(image_path)
        results = self.model.predict(
            image_path, conf=conf, iou=iou, verbose=False, device=self.device
        )
        r = results[0]
        names = r.names  # dict(int->str) or list
        items: List[Dict[str, Any]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().tolist()

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                cid = int(cls[i])
                # 안전하게 클래스 라벨 얻기
                if isinstance(names, dict):
                    label = str(names.get(cid, cid))
                else:
                    try:
                        label = str(names[cid])
                    except Exception:
                        label = str(cid)
                items.append({"class": label, "bbox": [int(x1), int(y1), int(x2), int(y2)]})

        return {"type": "openings", "items": items, "meta": {"image": Path(image_path).name}}

_detector_singleton: Optional[OpeningsDetector] = None

def load_openings_detector(weights: str | Path, device: Optional[str] = None) -> OpeningsDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = OpeningsDetector(weights, device)
    return _detector_singleton
