from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class RoomsSegmentor:
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
        Return rooms as polygons when seg is available; otherwise box->rect polygon.

        Output schema:
        {
          "type": "rooms",
          "items": [{"class": str, "poly": [[x,y],...]}],
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

        # Prefer segmentation masks if available
        if getattr(r, "masks", None) is not None and r.masks is not None:
            polys_list = r.masks.xy  # list[np.ndarray Nx2]
            cls_list = r.boxes.cls.cpu().numpy().tolist() if r.boxes is not None else [0] * len(polys_list)
            confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else [1.0] * len(polys_list)
            for i, poly in enumerate(polys_list):
                name = names.get(int(cls_list[i]), str(int(cls_list[i])))
                pts = poly.tolist()
                items.append(
                    {
                        "class": name,
                        "poly": [[int(x), int(y)] for x, y in pts],
                    }
                )
        else:
            # Fallback: boxes to rectangle polygons
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                cls = r.boxes.cls.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    name = names.get(int(cls[i]), str(int(cls[i])))
                    rect = [
                        [int(x1), int(y1)],
                        [int(x2), int(y1)],
                        [int(x2), int(y2)],
                        [int(x1), int(y2)],
                    ]
                    items.append(
                        {
                            "name": name,
                            "poly": rect,
                        }
                    )

        return {"type": "rooms", "items": items, "meta": {"image": Path(image_path).name}}


# singleton loader
_segmentor_singleton: Optional[RoomsSegmentor] = None


def load_rooms_segmentor(weights: str | Path, device: Optional[str] = None) -> RoomsSegmentor:
    global _segmentor_singleton
    if _segmentor_singleton is None:
        _segmentor_singleton = RoomsSegmentor(weights, device)
    return _segmentor_singleton