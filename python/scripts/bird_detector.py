import sys, time, threading, queue, signal
from datetime import datetime
from pathlib import Path
import importlib
import configparser
import cv2

# Configuration
cfg = configparser.ConfigParser()
config_path = Path.cwd() / "config.ini"
if not cfg.read(config_path):
    print(f"[WARN] config.ini not found at {config_path}. Using defaults.", file=sys.stderr)

CAMERAS = [c.strip() for c in cfg.get("general", "camera_ids", fallback="0").split(",")]
FPS = float(cfg.get("general", "inference_fps", fallback="5"))
MIN_AREA = int(cfg.get("general", "min_motion_area", fallback="500"))
MODE = cfg.get("general", "detection_mode", fallback="bg_subtract").lower()

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = float(cfg.get("label", "font_scale", fallback="0.7"))
FONT_THK = int(cfg.get("label", "thickness", fallback="2"))

# Dynamic detector import
try:
    det_module = importlib.import_module(f"budgie_bot.detectors.{MODE}")
    detector   = getattr(det_module, MODE)
except (ModuleNotFoundError, AttributeError) as e:
    print(f"[ERROR] detector '{MODE}' not found: {e}", file=sys.stderr)
    sys.exit(1)

# Worker thread
def camera_worker(cam_id: str,
                  frame_q: queue.Queue[tuple[str, cv2.Mat]],
                  ctrl_q:  queue.Queue[str]):
    cap = cv2.VideoCapture(int(cam_id)) if cam_id.isdigit() else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}", file=sys.stderr)
        return

    ok, first_frame = cap.read()
    if not ok:
        print(f"[ERROR] Camera {cam_id} returned no frames.", file=sys.stderr)
        cap.release()
        return

    state       = detector("init", first_frame, MIN_AREA)
    period      = 1.0 / FPS
    last_inf_t  = 0.0
    last_motion = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last_inf_t >= period:
            last_inf_t = now
            motion, state = detector("detect", frame, MIN_AREA, state)
            last_motion   = bool(motion)
            print(f"{datetime.now().isoformat(timespec='seconds')},{cam_id},{int(motion)}")

        # non‑blocking control messages
        try:
            msg = ctrl_q.get_nowait()
            if msg == "set_bg":
                state = detector("set_bg", frame, MIN_AREA, state)
        except queue.Empty:
            pass

        # annotate main label
        label, ok_flag = ("BIRD", True) if last_motion else ("NO BIRD", False)
        color = (0, 255, 0) if ok_flag else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), FONT, FONT_SCALE, color, FONT_THK, cv2.LINE_AA)

        # draw background thumbnail inset (if available)
        bg_img = state.get("bg") if isinstance(state, dict) else None
        if bg_img is not None:
            H, W = frame.shape[:2]
            inset_h, inset_w = H // 4, W // 4           # ¼‑size thumbnail
            bg_thumb = cv2.resize(bg_img, (inset_w, inset_h))

            # destination ROI (10‑px margin from bottom‑right)
            y1, y2 = H - inset_h - 10, H - 10
            x1, x2 = W - inset_w - 10, W - 10

            # blend thumbnail over ROI for transparency effect
            roi = frame[y1:y2, x1:x2]
            cv2.addWeighted(bg_thumb, 0.4, roi, 0.6, 0, dst=roi)

            # thin white border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, "BG", (x1 + 4, y1 + 14),
                        FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # push latest frame (overwrite queue if full)
        try: frame_q.get_nowait()
        except queue.Empty: pass
        frame_q.put_nowait((cam_id, frame))

    cap.release()

if __name__ == "__main__":
    frame_queues = [queue.Queue(maxsize=1) for _ in CAMERAS]  # images
    ctrl_queues  = [queue.Queue(maxsize=5) for _ in CAMERAS]  # commands

    threads = [threading.Thread(target=camera_worker,
                                args=(cid, fq, cq),
                                daemon=True)
               for cid, fq, cq in zip(CAMERAS, frame_queues, ctrl_queues)]
    for t in threads:
        t.start()

    # tidy exit on Ctrl‑C in terminal
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    key2idx = {ord(str(i)): i for i in range(10)}   # digit key → index

    try:
        while True:
            any_alive = False

            # Display most‑recent frames
            for cid, fq in zip(CAMERAS, frame_queues):
                try:
                    cam_id, frame = fq.get_nowait()
                    cv2.imshow(f"Cam {cam_id}", frame)
                except queue.Empty:
                    pass

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                    # Esc -> quit
                break
            elif key in key2idx:             # digit -> reset BG for that cam
                idx = key2idx[key]
                if idx < len(ctrl_queues):
                    try:
                        ctrl_queues[idx].put_nowait("set_bg")
                        print(f"[INFO] BG reset requested for camera {CAMERAS[idx]}")
                    except queue.Full:
                        pass

            # keep GUI responsive / detect thread end
            for t in threads:
                any_alive |= t.is_alive()
            if not any_alive:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted – shutting down…")

    cv2.destroyAllWindows()
