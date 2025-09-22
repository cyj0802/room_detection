from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class FurnitureDetector:
    def __init__(self, weights: str | Path, device: Optional[str] = None):
        self.weights = str(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`.")
        self.model = YOLO(self.weights)
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Return furniture as bounding boxes.

        Output schema:
        {
          "type": "furniture",
          "items": [{"class": str, "poly": [x,y,w,h]}],
          "meta": {"image": "..."}
        }
        """
        image_path = str(image_path)
        results = self.model.predict(
            image_path, conf=conf, iou=iou, verbose=False, device=self.device
        )
        r = results[0]
        names = r.names
        items: List[Dict[str, Any]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().tolist()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                w = int(x2 - x1)
                h = int(y2 - y1)
                items.append(
                    {
                        "class": names.get(int(cls[i]), str(int(cls[i]))),
                        "poly": [int(x1), int(y1), w, h],
                    }
                )

        return {"type": "furniture", "items": items, "meta": {"image": Path(image_path).name}}


# singleton loader
_furniture_singleton: Optional[FurnitureDetector] = None


def load_furniture_segmentor(weights: str | Path, device: Optional[str] = None) -> FurnitureDetector:
    global _furniture_singleton
    if _furniture_singleton is None:
        _furniture_singleton = FurnitureDetector(weights, device)
    return _furniture_singleton