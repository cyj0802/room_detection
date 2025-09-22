# inference/corners.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

try:
    from ultralytics import YOLO
except Exception:  # Ultralytics 미설치 시 명확한 에러 처리
    YOLO = None

# corner 탐지 모델을 메모리에 로드하는 부분
class CornersDetector:
    def __init__(self, weights: str | Path, device: Optional[str] = None):
        self.weights = str(weights)
        wants_cuda = (device or "").startswith("cuda")
        has_cuda = torch.cuda.is_available()
        # 요청이 cuda여도 현재 torch가 cuda 미지원이면 강제로 cpu
        self.device = "cuda" if (wants_cuda and has_cuda) else "cpu"

        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`.")
        self.model = YOLO(self.weights)
        # .to()는 CUDA 미지원 빌드에서 실패할 수 있으니 조건부 적용
        if self.device == "cuda":
            self.model.to(self.device)

    # 추론모드  
    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Return corners as point-like detections.

        Output schema:
        {
          "type": "corners",
          "items": [{"class": str, "x": int, "y": int}],
          "meta": {"image": "..."}
        }
        """
        image_path = str(image_path)
        results = self.model.predict(image_path, conf=conf, iou=iou, verbose=False, device=self.device)

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
                items.append(
                    {
                        "class": names.get(cid, str(cid)),
                        "x": cx,
                        "y": cy,
                    }
                )

        return {"type": "corners", "items": items, "meta": {"image": Path(image_path).name}}


# convenience singleton loader
_detector_singleton: Optional[CornersDetector] = None

def load_corners_detector(weights: str | Path, device: Optional[str] = None) -> CornersDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = CornersDetector(weights, device)
    return _detector_singleton
