import os, random, numpy as np, torch

# 모든 랜덤 요소를 고정해 결과가 매번 동일하게 나오도록 하는 함수
def set_global_determinism(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# gpu 없으면 cpu
def select_device(device_hint=None):
    return 0 if torch.cuda.is_available() else "cpu"

# 같은 이미지 넣으면 항상 같은 결과가 나오도록 보장
def predict_consistent_model(
    model, source, *,
    device=None, imgsz=1024, conf=0.25, iou=0.45,
    half=False, agnostic_nms=True, retina_masks=False,
    max_det=1000, augment=False, verbose=False,
):
    set_global_determinism(0) 
    if device is None:
        device = select_device()
    return model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,      
        half=half,
        agnostic_nms=agnostic_nms,
        retina_masks=retina_masks,
        max_det=max_det,
        augment=augment,
        verbose=verbose,
        save=False,
    )
