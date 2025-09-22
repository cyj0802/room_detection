from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import re, unicodedata, json
import cv2, numpy as np, torch

# ---------- 외부 라이브러리 ----------
try:
    import easyocr
except Exception as e:
    raise RuntimeError("pip install easyocr") from e

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = process = None  # 설치 안 되어도 초성/정규식으로만 동작

# ---------- EasyOCR 설정 ----------
LANGS = ['ko']
_READER: Optional[easyocr.Reader] = None

def get_reader(force_gpu: Optional[bool] = True) -> easyocr.Reader:
    """EasyOCR Reader 싱글톤"""
    global _READER
    if _READER is None:
        use_gpu = torch.cuda.is_available() if force_gpu is None else (force_gpu and torch.cuda.is_available())
        _READER = easyocr.Reader(LANGS, gpu=use_gpu)
    return _READER

# ---------- 라벨 사전 ----------
SYNONYMS: Dict[str, List[str]] = {
    "r1":  ["침실", "가족실"],
    "r2":  ["화장실"],
    "r3":  ["샤워부스"],
    "r4":  ["발코니"],
    "r5":  ["duck"], 
    "r6":  ["현관"],
    "r7":  ["closet", "드레스룸", "창고", "다용도실", "실외기"],
    "r8":  ["테라스", "야외테라스", "개방형 발코니"],
    "r9":  ["주방 및 식당", "주방"],
    "r10": ["드레스룸, 파우더룸"],
    "r11": ["계단실"],
}

SUB_MAP: Dict[str, str] = {
    "발코니": "r4", 
    "창고":       "r7-1",
    "다용도실":   "r7-2",
    "실외기":     "r7-3",
    "테라스":     "r8",
    "개방형 발코니": "r8-1",
    "드레스룸":   "r10",
    "파우더룸":   "r10-1",
}

CANDIDATES = list(SUB_MAP.keys())

# 초성
_CHOSEONG_LIST = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

def normalize_korean(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.strip()
    s = re.sub(r"[\"'`·▪️•\-\_\.\,\:\;\!\?\(\)\[\]\{\}]", "", s)
    s = re.sub(r"\s+", "", s)
    return s

def _choseong(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFC", s or ""):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:  # 한글 음절
            base = code - 0xAC00
            cho = base // (21 * 28)
            out.append(_CHOSEONG_LIST[cho])
        elif "\u3131" <= ch <= "\u3163":  # 자모 그대로
            out.append(ch)
    return "".join(out)

def _fuzzy_one(target: str, candidates: List[str], cutoff=80) -> Optional[str]:
    if not target or not candidates or process is None:
        return None
    best = process.extractOne(target, candidates, scorer=fuzz.partial_ratio)
    return best[0] if (best and best[1] >= cutoff) else None

def debug_print_ocr(ocr_items, *, header: str = "", conf_th: float = None):
    if header:
        print(header)
    if not ocr_items:
        print("  (no ocr items)")
        return
    for idx, item in enumerate(ocr_items):
        bbox, text, conf = None, None, None
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                bbox, text, conf = item
            elif len(item) == 2:
                bbox, text = item
                conf = None
            elif len(item) == 1:
                text = item[0]
        else:
            text = item
        try:
            bb = np.array(bbox).astype(int).tolist() if bbox is not None else None
        except Exception:
            bb = bbox
        print(f"  - idx={idx} | text={text!r} | conf={conf if conf is not None else 'NA'} | bbox={bb}")
    if conf_th is not None:
        print(f"  ↳ (conf_th={conf_th}) 이상만 최종 join에 사용")

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

def join_ocr_texts(ocr_items, conf_th=0.45) -> Tuple[str, str]:
    texts = extract_texts_from_easyocr(ocr_items, conf_th=conf_th)
    if not texts:
        return "", ""  # 빈 문자열 → 이후 fallback
    joined     = re.sub(r"\s+", " ", " ".join(texts)).strip()
    joined_nos = re.sub(r"\s+", "", " ".join(texts))
    return joined, joined_nos

# ---------- 간단한 오타 보정 ----------
def heuristic_fix(s: str) -> str:
    # 대표 오타/분절 보정 (필요시 추가)
    s = s.replace("도레", "드레")
    s = s.replace("스움", "스룸")
    s = s.replace("싴외기", "실외기").replace("실외긔", "실외기")
    s = s.replace("다용 도심", "다용도실")
    s = s.replace("도레 스톱", "드레스룸")
    s = s.replace("드레 스톱", "드레스룸")
    return s

# ---------- 정규식 우선 매칭 ----------
def regex_pick(joined: str, nospace: str) -> Tuple[Optional[str], Optional[str]]:
    j = joined or ""
    n = nospace or ""

    if re.search(r"(발|바)\s*코\s*(?:니)?|BALCONY", j, re.I) \
       or re.search(r"((발|바)코니?|BALCONY)", n, re.I):
        return "발코니", "r4"

    # r10-1 (파우더룸) 우선
    if re.search(r"파\s*우\s*더\s*룸|파우더룸|POWDER\s*ROOM|POWDERROOM|파우더", j, re.I) \
       or re.search(r"(파우더룸|POWDERROOM|POWDERROOM|파우더)", n, re.I):
        return "파우더룸", "r10-1"

    # r10 (드레스룸)
    if re.search(r"드레\s*스\s*룸|드레스\s*룸|드레스룸|DRESS\s*ROOM|DRESSROOM|WIC|WALK\s*IN\s*CLOSET|WALKINCLOSET", j, re.I) \
       or re.search(r"(드레스룸|DRESSROOM|WIC|WALKINCLOSET)", n, re.I):
        return "드레스룸", "r10"

    # r8-1 (개방형 발코니)
    if re.search(r"개방형\s*발코니|오픈\s*발코니|OPEN\s*BALCONY|OPENBALCONY", j, re.I) \
       or re.search(r"(개방형발코니|오픈발코니|OPENBALCONY)", n, re.I):
        return "개방형 발코니", "r8-1"

    # r8 (테라스)
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

# ---------- SYNONYMS 전개 & 매퍼 ----------
def _expand_synonyms(raw: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """'드레스룸, 파우더룸' → ['드레스룸','파우더룸']로 분해"""
    exp: Dict[str, List[str]] = {}
    for k, arr in raw.items():
        acc = []
        for a in arr:
            parts = re.split(r"[\,/·ㆍ]", a)
            for p in parts:
                p = p.strip()
                if p:
                    acc.append(p)
        exp[k] = acc
    return exp

EXP_SYNS = _expand_synonyms(SYNONYMS)

# 초성 힌트(간단)
CHO_HINTS: Dict[str, List[str]] = {
    "r1":  ["ㅊㅅ","ㄱㅈㅅ"],              # 침실/가족실/방
    "r2":  ["ㅎㅈㅅ"],                          # 화장실
    "r3":  ["ㅅㅇㅂㅅ"],                        # 샤워부스
    "r4":  ["ㅂㅋㄴ"],                          # 발코니
    "r6":  ["ㅎㄱ"],                            # 현관
    "r7":  ["ㄷㄹㅅㄹ","ㅊㄱ","ㄷㅇㄷㅅ","ㅅㅇㄱ"],  # 드레스룸/창고/다용도실/실외기
    "r8":  ["ㅌㄹㅅ","ㅇㅇㅌㄹㅅ","ㄱㅂㅎㅂㅋㄴ"],   # 테라스/야외테라스/개방형발코니
    "r9":  ["ㅈㅂ","ㅈㅂㅁㅅㄷ"],              # 주방/주방및식당
    "r10": ["ㄷㄹㅅㄹ","ㅍㅇㄷㄹ"],            # 드레스룸/파우더룸
    "r11": ["ㄱㄷㅅ"],                          # 계단실
}

def map_text_to_label(yolo_label: str, raw_text: str) -> Optional[str]:
    """
    OCR 텍스트로 먼저 세부라벨(SUB_MAP) → 실패 시 기본라벨(SYNONYMS) → 실패 시 초성 → None
    반환: 'r7-1', 'r8-1' 같은 세부코드 또는 'r1'~'r11' 기본코드
    """
    s = normalize_korean(raw_text)
    if not s:
        return None

    # 0) 세부라벨: 부분문자열 우선
    for key, code in SUB_MAP.items():
        if normalize_korean(key) in s:
            return code

    # 1) 세부라벨 퍼지
    hit = _fuzzy_one(s, list(SUB_MAP.keys()), cutoff=85)
    if hit:
        return SUB_MAP[hit]

    # 2) 기본라벨: 정확 포함
    for rid, words in EXP_SYNS.items():
        for w in words:
            nw = normalize_korean(w)
            if nw and nw in s:
                return rid

    # 3) 기본라벨 퍼지
    if process is not None:
        all_words = [(rid, w) for rid, arr in EXP_SYNS.items() for w in arr if w]
        cand_words = [w for _, w in all_words]
        wbest = _fuzzy_one(s, cand_words, cutoff=82)
        if wbest:
            for rid, w in all_words:
                if w == wbest:
                    return rid

    # 4) 초성 힌트
    ch = _choseong(s)
    for rid, hints in CHO_HINTS.items():
        if any(h in ch or ch.startswith(h) for h in hints):
            return rid

    # 5) 따옴표 제거 후 재시도
    s2 = normalize_korean(raw_text.replace("'", "").replace("\"", ""))
    if s2 and s2 != s:
        return map_text_to_label(yolo_label, s2)

    return None

# ---------- 퍼지 폴백 (이전 호환) ----------
def fuzzy_pick(joined: str, nospace: str, yolo_label: str, score_cutoff=78) -> Tuple[str, str, str]:
    """
    return: (name, code, reason)
      - reason ∈ {'regex', 'map_text', 'fuzzy', 'fallback_yolo'}
    1) 정규식 강매칭 → 세부코드
    2) SUB/SYN 매핑(map_text_to_label)
    3) 퍼지(CANDIDATES)
    4) 실패 시 YOLO 라벨 유지
    """
    # 1) 정규식 강매칭
    name, code = regex_pick(joined, nospace)
    if code:
        return name, code, "regex"

    # 2) 세부/기본 매핑
    mapped = map_text_to_label(yolo_label, (nospace or joined))
    if mapped:
        return mapped, mapped, "map_text"

    # 3) 세부라벨에 대한 퍼지(토큰셋)
    raw = heuristic_fix((joined or "") + " " + (nospace or ""))
    if process is not None:
        best = process.extractOne(raw, CANDIDATES, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
    else:
        best = None
    if best:
        name = best[0]
        return name, SUB_MAP[name], f"fuzzy(score≥{score_cutoff})"

    # 4) 전부 실패 → YOLO 라벨 유지
    return yolo_label, yolo_label, "fallback_yolo"

# ---------- ROI 추출 & 전처리 ----------
def crop_roi_by_poly(image, poly, pad=16):  
    H, W = image.shape[:2]
    pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
    mask = np.zeros((H,W), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    roi = cv2.bitwise_and(image[y0:y1, x0:x1], image[y0:y1, x0:x1], mask=mask[y0:y1, x0:x1])
    return roi

def preprocess_for_ocr(roi_bgr):
    # 1) 업스케일 (작은 글자 복원에 효과)
    H, W = roi_bgr.shape[:2]
    scale = 2 if max(H, W) < 120 else 1  # 너무 크면 과적합이니 작은 경우만
    if scale > 1:
        roi_bgr = cv2.resize(roi_bgr, (W*scale, H*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 2) 대비 향상 + 노이즈 정리
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    # 작은 구멍/틈 메우기 (글자 끊김 방지)
    gray = cv2.medianBlur(gray, 3)

    # 3) 이진화 멀티패스: adaptive vs Otsu 둘 다 시도하도록, 여기선 adaptive만 반환
    binm = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)
    return binm

def _read_with_variants(reader, img_bin_or_gray):
    # EasyOCR 파라미터 후보들
    param_sets = [
        dict(detail=1, paragraph=True,  contrast_ths=0.05, adjust_contrast=0.7,
             text_threshold=0.3, low_text=0.2, link_threshold=0.2,
             decoder='beamsearch', allowlist='가-힣 ·'),
        dict(detail=1, paragraph=True,  contrast_ths=0.1,  adjust_contrast=0.5,
             text_threshold=0.4, low_text=0.3, link_threshold=0.3,
             decoder='greedy',     allowlist='가-힣 ·'),
    ]
    outs = []
    for ps in param_sets:
        outs.append(reader.readtext(img_bin_or_gray, **ps))
    return outs

def join_ocr_texts_best(reader, roi):
    # 멀티 전처리: binary(adaptive), binary(otsu), invert 두 가지
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()

    # adaptive
    ada = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,31,5)
    # otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # invert 후보
    ada_inv = 255 - ada
    otsu_inv = 255 - otsu

    candidates = []
    for variant in (ada, otsu, ada_inv, otsu_inv):
        all_ocr = _read_with_variants(reader, variant)
        for ocr_items in all_ocr:
            joined, nospace = join_ocr_texts(ocr_items, conf_th=0.45)
            # 점수: 한글 비율↑, 길이↑ 를 선호
            ko_len = sum(1 for ch in nospace if '가' <= ch <= '힣')
            score = ko_len * 2 + len(nospace)  # 가중치 간단
            candidates.append((score, joined, nospace))

    # 가장 점수 높은 텍스트 채택
    candidates.sort(key=lambda x: x[0], reverse=True)
    if candidates:
        return candidates[0][1], candidates[0][2]
    return "", ""


# ---------- 파일 I/O 버전 ----------
def run_file(IMAGE_PATH: str, IN_JSON: str, OUT_JSON: str,
             *, mode: str = "code", force_gpu: bool = True, verbose: bool = False) -> int:
    """
    - 이미지/JSON 읽음
    - 각 폴리곤 ROI에 대해 OCR 수행
    - (정규식/세부·기본 매핑/퍼지)로 최종 라벨 결정
    - class 혹은 class_name을 덮어쓰고 저장
    - 변경 건수 반환
    """
    reader = get_reader(force_gpu)

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 열기 실패: {IMAGE_PATH}")

    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("items", [])
    changed = 0

    for idx, item in enumerate(items):
        yolo_label = (item.get("class_name") or item.get("class") or "").strip().lower()
        poly = item.get("poly") or item.get("polygon")
        if not poly:
            continue

        roi = crop_roi_by_poly(img, poly, pad=6)
        binm = preprocess_for_ocr(roi)

        if verbose:
            print(f"\n[ROI #{idx}] yolo_label={yolo_label}")
            debug_print_ocr(ocr, header="  · Raw OCR items:", conf_th=0.45)

        joined, nospace = join_ocr_texts_best(reader, roi)  # ← 멀티패스 결과 채택

        if not (joined or nospace):
            ocr = reader.readtext(binm, detail=1, paragraph=True, allowlist='가-힣 ·')
            joined, nospace = join_ocr_texts(ocr, conf_th=0.45)
        
        if verbose:
            print(f"  · joined    : {joined!r}")
            print(f"  · nospace   : {nospace!r}")

        # 통합 결정 (regex → map → fuzzy → yolo)
        name, code, reason = fuzzy_pick(joined, nospace, yolo_label, score_cutoff=78)
        new_cls = code if mode == "code" else name

        # 덮어쓰기
        orig = item.get("class_name") if "class_name" in item else item.get("class")
        if "class_name" in item:
            item["class_name"] = new_cls
        if "class" in item:
            item["class"] = new_cls
        if "class" not in item and "class_name" not in item:
            item["class"] = new_cls

        if verbose:
            print(f"  · decision  : {name} → {new_cls}  (reason={reason})")

        if (orig or "").lower() != (new_cls or "").lower():
            changed += 1
            print(f"UPDATE: {orig}  →  {new_cls}  (ocr='{joined}')")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            data["items"] = items
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUT_JSON} (changed={changed})")
    return changed

# ---------- 모듈 API 버전 ----------
def annotate_rooms_with_ocr(
    image_path: str,
    rooms_json: Dict[str, Any],
    *,
    overwrite: bool = True,
    mode: str = "code",
    force_gpu: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    파이프라인(dict → dict): 각 폴리곤 ROI에 OCR → 라벨 교체
    """
    reader = get_reader(force_gpu)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 열기 실패: {image_path}")

    out_items: List[Dict[str, Any]] = []
    changed = 0

    for idx, item in enumerate(rooms_json.get("items", [])):
        poly = item.get("poly") or item.get("polygon")
        if not poly:
            out_items.append(item)
            continue

        # 기존 라벨
        yolo_label = (item.get("class") or item.get("class_name") or "").strip().lower()

        # 폴리곤 ROI OCR
        roi  = crop_roi_by_poly(img, poly, pad=6)
        binm = preprocess_for_ocr(roi)
        ocr  = reader.readtext(binm, detail=1, paragraph=True)

        if verbose:
            print(f"\n[annotate ROI #{idx}] yolo_label={yolo_label}")
            debug_print_ocr(ocr, header="  · Raw OCR items:", conf_th=0.45)

        joined, nospace = join_ocr_texts(ocr, conf_th=0.45)
        if verbose:
            print(f"  · joined    : {joined!r}")
            print(f"  · nospace   : {nospace!r}")

        # 통합 결정
        name, code, reason = fuzzy_pick(joined, nospace, yolo_label, score_cutoff=78)

        new_item = dict(item)
        if overwrite:
            new_cls = code if mode == "code" else name
            orig = new_item.get("class_name") if "class_name" in new_item else new_item.get("class")

            # 교체
            if "class_name" in new_item:
                new_item["class_name"] = new_cls
            if "class" in new_item:
                new_item["class"] = new_cls
            if "class" not in new_item and "class_name" not in new_item:
                new_item["class"] = new_cls

            # 변경 로그 출력
            if (orig or "").lower() != (new_cls or "").lower():
                changed += 1
                if verbose:
                    print(f"  · decision  : {name} → {new_cls}  (reason={reason})")
                print(f"UPDATE: {orig or yolo_label}  →  {new_cls}  (ocr='{joined}')")

        out_items.append(new_item)

    if verbose:
        print(f"[rooms_ocr] total updates: {changed}")

    return {
        "type": rooms_json.get("type", "rooms"),
        "items": out_items,
        "meta": rooms_json.get("meta", {}),
    }
