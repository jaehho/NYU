import time, threading, queue, cv2, sys, configparser, pathlib
from datetime import datetime
from ultralytics import YOLO  # CPU inference supported
import numpy as np

# ---------- config ----------
cfg = configparser.ConfigParser()
cfg.read(pathlib.Path(__file__).parent.parent / "config.ini")
CAMERAS       = [c.strip() for c in cfg["general"]["camera_ids"].split(",")]
INF_FPS       = float(cfg["general"]["inference_fps"])
LABEL_PARAMS  = cfg["label"]
LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX
# -----------------------------

model = YOLO("yolov8n.pt").to("cpu")   # force CPU
BIRD_CLASS_ID = 14                     # COCO class index for "bird"

def capture_loop(cam_id, frame_q):
    cap = cv2.VideoCapture(int(cam_id)) if cam_id.isdigit() else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}", file=sys.stderr); return

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = resize_frame(frame, width=416)  # shrink to help CPU inference
        if not frame_q.full():
            frame_q.put(frame)
        else:
            try: frame_q.get_nowait()
            except queue.Empty: pass
            frame_q.put(frame)
    cap.release()

def inference_loop(cam_id, frame_q):
    period = 1.0 / INF_FPS
    last_t = 0
    while True:
        try:
            frame = frame_q.get(timeout=1)
        except queue.Empty:
            continue

        now = time.time()
        if now - last_t < period:
            show(frame, False, cam_id)
            continue

        last_t = now
        bird_present = detect_bird(frame)
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp},{cam_id},{int(bird_present)}")
        show(frame, bird_present, cam_id)

def detect_bird(frame):
    results = model(frame, conf=0.25, iou=0.5, verbose=False)[0]
    return any(det.cls == BIRD_CLASS_ID for det in results.boxes)

def resize_frame(frame, width=416):
    h, w = frame.shape[:2]
    ratio = width / float(w)
    new_dim = (width, int(h * ratio))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)

def show(frame, bird, cam_id):
    label = "BIRD" if bird else "NO BIRD"
    color = (0,255,0) if bird else (0,0,255)
    cv2.putText(frame, label,
                (10, 30),
                LABEL_FONT,
                float(LABEL_PARAMS.get("font_scale", 0.7)),
                color,
                int(LABEL_PARAMS.get("thickness", 2)),
                lineType=cv2.LINE_AA)
    cv2.imshow(f"Camera {cam_id}", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        sys.exit()

if __name__ == "__main__":
    queues = []
    for cam in CAMERAS:
        q = queue.Queue(maxsize=1)
        queues.append(q)
        threading.Thread(target=capture_loop,   args=(cam,q), daemon=True).start()
        threading.Thread(target=inference_loop, args=(cam,q), daemon=True).start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting…")
