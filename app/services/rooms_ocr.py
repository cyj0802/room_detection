# app/services/rooms_ocr.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re, unicodedata, json
import numpy as np
import cv2
import torch

try:
    import easyocr
except Exception as e:
    raise RuntimeError("easyocr가 필요합니다. pip install easyocr") from e

try:
    from rapidfuzz import fuzz, process
except Exception as e:
    raise RuntimeError("rapidfuzz가 필요합니다. pip install rapidfuzz") from e

# --------- 설정 ---------
LANGS = ['ko', 'en']  # 한국어 + 영어
HAS_CUDA = torch.cuda.is_available()
_READER_SINGLETON: Optional[easyocr.Reader] = None

def get_reader() -> easyocr.Reader:
    global _READER_SINGLETON
    if _READER_SINGLETON is None:
        # GPU가 없는 환경이면 자동으로 CPU로
        _READER_SINGLETON = easyocr.Reader(LANGS, gpu=HAS_CUDA)
    return _READER_SINGLETON

# r7/r8/r10 서브클래스 매핑
SUB_MAP = {
    "창고": "r7-1",
    "다용도실": "r7-2",
    "실외기": "r7-3",
    "테라스": "r8",
    "개방형 발코니": "r8-1",
    "드레스룸": "r10",
    "파우더룸": "r10-1",
}
CANDIDATES = list(SUB_MAP.keys())

def normalize_korean(text: str) -> str:
    return unicodedata.normalize('NFC', text).strip()

def extract_texts_from_easyocr(ocr_items, conf_th=0.45):
    texts = []
    for item in ocr_items or []:
        text, conf = None, 1.0
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                _, text, conf = item
            elif len(item) == 2:
                _, text = item
            elif len(item) == 1:
                text = item[0]
        else:
            text = item
        if text is None:
            continue
        if conf is None:
            conf = 1.0
        if conf >= conf_th:
            texts.append(normalize_korean(text))
    return texts

def join_ocr_texts(ocr_items, conf_th=0.45):
    texts = extract_texts_from_easyocr(ocr_items, conf_th=conf_th)
    if not texts:
        return "", ""
    joined     = re.sub(r"\s+", " ", " ".join(texts)).strip()
    joined_nos = re.sub(r"\s+", "", " ".join(texts))
    return joined, joined_nos

def heuristic_fix(s: str) -> str:
    s = s.replace("도레", "드레")
    s = s.replace("스움", "스룸")
    s = s.replace("싴외기", "실외기").replace("실외긔", "실외기")
    s = s.replace("다용 도심", "다용도실")
    s = s.replace("도레 스톱", "드레스룸")
    return s

def regex_pick(joined: str, nospace: str):
    j = joined or ""
    n = nospace or ""
    # r10-1 파우더룸
    if re.search(r"파\s*우\s*더\s*룸|파우더룸|POWDER\s*ROOM|POWDERROOM|파우더", j, re.I) or re.search(r"(파우더룸|POWDERROOM|파우더)", n, re.I):
        return "파우더룸", "r10-1"
    # r10 드레스룸
    if re.search(r"드레\s*스\s*룸|드레스\s*룸|드레스룸|DRESS\s*ROOM|DRESSROOM|WIC|WALK\s*IN\s*CLOSET|WALKINCLOSET", j, re.I) \
       or re.search(r"(드레스룸|DRESSROOM|WIC|WALKINCLOSET)", n, re.I):
        return "드레스룸", "r10"
    # r8-1 개방형 발코니
    if re.search(r"개방형\s*발코니|오픈\s*발코니|OPEN\s*BALCONY|OPENBALCONY", j, re.I) \
       or re.search(r"(개방형발코니|오픈발코니|OPENBALCONY)", n, re.I):
        return "개방형 발코니", "r8-1"
    # r8 테라스
    if re.search(r"테라스|TERRACE", j, re.I) or re.search(r"(테라스|TERRACE)", n, re.I):
        return "테라스", "r8"
    # r7 서브
    if re.search(r"창\s*고|수납\s*실|저장\s*고|창고", j, re.I) or re.search(r"(창고|수납실|저장고)", n, re.I):
        return "창고", "r7-1"
    if re.search(r"다\s*용\s*도\s*실|다용도\s*실|다목적\s*실|다목적", j, re.I) or re.search(r"(다용도실|다목적실|다목적)", n, re.I):
        return "다용도실", "r7-2"
    if re.search(r"실\s*외\s*기|실외기|에어컨\s*실외기", j, re.I) or re.search(r"(실외기|에어컨실외기)", n, re.I):
        return "실외기", "r7-3"
    return None, None

def fuzzy_pick(joined: str, nospace: str, yolo_label: str, score_cutoff=78):
    # 1) 정규식 우선
    name, code = regex_pick(joined, nospace)
    if code:
        return name, code
    # 2) 퍼지 백업
    raw = heuristic_fix((joined or "") + " " + (nospace or ""))
    best = process.extractOne(raw, CANDIDATES, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
    if best:
        name = best[0]
        return name, SUB_MAP[name]
    # 3) 실패 → YOLO 라벨 유지
    return yolo_label, yolo_label

def crop_roi_by_poly(image: np.ndarray, poly: List[List[float]], pad: int = 6) -> np.ndarray:
    h, w = image.shape[:2]
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    x, y, ww, hh = cv2.boundingRect(pts)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
    roi = cv2.bitwise_and(image[y0:y1, x0:x1], image[y0:y1, x0:x1], mask=mask[y0:y1, x0:x1])
    return roi

def preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    binm = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,5)
    return binm

# --------- 공개 함수: rooms JSON + 이미지 → rooms_ocr JSON ---------
def annotate_rooms_with_ocr(image_path: str, rooms_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    입력:
      - image_path: 원본 도면 이미지 경로
      - rooms_json: {"type":"rooms","items":[{"class":str,"poly":[[x,y],...]}], "meta":{...}}

    출력:
      - 같은 스키마 + 각 item에 "subclass_name", "subclass_code" 필드 추가
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")

    reader = get_reader()
    items_out: List[Dict[str, Any]] = []

    for item in rooms_json.get("items", []):
        yolo_label = (item.get("class") or item.get("class_name") or "").strip().lower()
        poly = item.get("poly") or item.get("polygon")

        # 폴리곤 없으면 스킵
        if not poly:
            items_out.append(item)
            continue

        # r7/r8/r10 만 OCR 후처리 (다른 방은 통과)
        if yolo_label not in {"r7", "r8", "r10"}:
            items_out.append(item)
            continue

        roi = crop_roi_by_poly(img, poly, pad=6)
        binm = preprocess_for_ocr(roi)
        ocr = reader.readtext(binm, detail=1, paragraph=True)
        joined, nospace = join_ocr_texts(ocr, conf_th=0.45)

        name, code = fuzzy_pick(joined, nospace, yolo_label, score_cutoff=78)

        new_item = dict(item)
        new_item["subclass_name"] = name
        new_item["subclass_code"] = code
        items_out.append(new_item)

    return {
        "type": rooms_json.get("type", "rooms"),
        "items": items_out,
        "meta": rooms_json.get("meta", {}),
    }
