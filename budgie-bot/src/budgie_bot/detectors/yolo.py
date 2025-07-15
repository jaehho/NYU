from ultralytics import YOLO
BIRD_CLS = 14
_model = YOLO("yolov8n.pt")
_model.to("cpu")  # <-- force CPU

def yolo(event: str, frame, *_, state=None):
    if event in ("init", "reset", "set_bg"):
        return {}
    if event == "detect":
        res = _model(frame, conf=0.25, iou=0.5, verbose=False, device="cpu")[0]
        motion = any(b.cls == BIRD_CLS for b in res.boxes)
        return motion, {}
    return {}
