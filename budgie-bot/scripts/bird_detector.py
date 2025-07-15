import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import configparser
import importlib

import cv2

# Configuration
cfg = configparser.ConfigParser()
config_path = Path.cwd() / "config.ini"
if not cfg.read(config_path):
    print(f"[WARN] config.ini not found at {config_path}. Using defaults.", file=sys.stderr)

CAMERAS  = [c.strip() for c in cfg.get("general", "camera_ids",  fallback="0").split(",")]
FPS      = float(cfg.get("general", "inference_fps",   fallback="5"))
MIN_AREA = int(  cfg.get("general", "min_motion_area", fallback="500"))
MODE     = cfg.get("general", "detection_mode", fallback="snapshot").lower()

LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = float(cfg.get("label", "font_scale", fallback="0.7"))
LABEL_THK   = int(  cfg.get("label", "thickness",  fallback="2"))

# Dynamic import: look for detectors.<mode> module exposing a function of the same name
try:
    det_module = importlib.import_module(f"budgie_bot.detectors.{MODE}")
    detector   = getattr(det_module, MODE)
except (ModuleNotFoundError, AttributeError) as e:
    print(f"[ERROR] detector '{MODE}' not found: {e}", file=sys.stderr)
    sys.exit(1)

def camera_worker(cam_id: str):
    cap = cv2.VideoCapture(int(cam_id)) if cam_id.isdigit() else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}", file=sys.stderr)
        return

    detector_state = detector("init", cap.read()[1], MIN_AREA)

    last_result, last_inf = None, 0.0
    period = 1.0 / FPS
    win_title = f"Cam {cam_id}"

    while True:
        ok, frame = cap.read()
        if not ok: break

        key = cv2.pollKey() & 0xFF
        if key == ord('b'):
            detector_state = detector("set_bg", frame, MIN_AREA, detector_state)
        elif key == ord('r'):
            detector_state = detector("reset", frame, MIN_AREA, detector_state)
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); sys.exit()

        now = time.time()
        if now - last_inf >= period:
            last_inf = now
            motion, detector_state = detector("detect", frame, MIN_AREA, detector_state)
            last_result = int(motion)
            print(f"{datetime.now().isoformat(timespec='seconds')},{cam_id},{last_result}")

        # choose label
        if MODE in ("snapshot", "background") and detector_state.get("bg") is None:
            label, ok_flag = "PRESS 'b' TO SET BG", False
        else:
            ok_flag = bool(last_result)
            label = "BIRD" if ok_flag else "NO BIRD"


        color = (0, 255, 0) if ok_flag else (0, 0, 255)
        cv2.putText(frame, label, (10, 30),
                    LABEL_FONT, LABEL_SCALE, color, LABEL_THK, cv2.LINE_AA)
        cv2.imshow(win_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyWindow(win_title)

if __name__ == "__main__":
    workers = [threading.Thread(target=camera_worker, args=(cid,), daemon=True)
               for cid in CAMERAS]
    for t in workers: t.start()

    try:
        while any(t.is_alive() for t in workers):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()