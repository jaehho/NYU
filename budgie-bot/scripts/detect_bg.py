import cv2
import time
import configparser, pathlib
import threading, sys
from datetime import datetime
import numpy as np


# Load Configuration
cfg = configparser.ConfigParser()
cfg.read(pathlib.Path(__file__).parent.parent / "config.ini")

CAMERAS     = [c.strip() for c in cfg["general"]["camera_ids"].split(",")]
FPS         = float(cfg["general"]["inference_fps"])
MIN_AREA    = int(cfg["general"]["min_motion_area"])

LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = float(cfg["label"]["font_scale"])
LABEL_THK   = int(cfg["label"]["thickness"])

def annotate(frame, text, ok):
    """Overlay label on frame with green (ok) or red (not ok) text."""
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(frame, text, (10, 30),
                LABEL_FONT, LABEL_SCALE, color, LABEL_THK, cv2.LINE_AA)
    return frame


def detect_motion(bg_img, frame, min_area=500):
    """
    Compares current frame to stored background image.
    Returns: motion_detected (bool)
    """
    # Compute absolute pixel difference
    diff = cv2.absdiff(bg_img, frame)

    # Convert to grayscale → blur → threshold to isolate changes
    gray   = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th  = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

    # Expand white blobs to consolidate motion regions
    th = cv2.dilate(th, None, iterations=2)

    # Find blobs (contours) in the motion mask
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour is large enough to count as motion
    return any(cv2.contourArea(c) > min_area for c in contours)


def camera_worker(cam_id: str):
    """
    Main camera thread: 
    - Reads frames
    - Waits for 'b' to set snapshot background
    - Runs motion detection at given FPS
    - Displays label and logs result
    """
    cap = cv2.VideoCapture(int(cam_id)) if cam_id.isdigit() else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}", file=sys.stderr)
        return

    bg_img       = None           # Frozen background frame
    last_motion  = None           # 0 or 1, updated every detection
    last_inf     = 0.0            # Last time detection ran
    period       = 1.0 / FPS
    win          = f"Cam {cam_id}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Handle Key Events 
        key = cv2.pollKey() & 0xFF
        if key == ord('b'):  # Set background
            bg_img = frame.copy()
            last_motion = 0
            print(f"[INFO] Background set for cam {cam_id}")
        elif key == ord('r'):  # Reset background
            bg_img = None
            last_motion = None
            print(f"[INFO] Background cleared for cam {cam_id}")
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        # Run Detection at Limited FPS 
        now = time.time()
        if bg_img is not None and (now - last_inf) >= period:
            last_inf = now
            motion_detected = detect_motion(bg_img, frame, MIN_AREA)
            last_motion = int(motion_detected)

            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"{timestamp},{cam_id},{last_motion}")

        #  Display Overlay 
        if bg_img is None:
            label, ok = "PRESS 'b' TO SET BG", False
        elif last_motion is None:
            label, ok = "WAITING…", False
        else:
            label, ok = ("BIRD", True) if last_motion else ("NO BIRD", False)

        show_frame = annotate(frame.copy(), label, ok)
        cv2.imshow(win, show_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow(win)


#  Main Program 
if __name__ == "__main__":
    threads = [
        threading.Thread(target=camera_worker, args=(cid,), daemon=True)
        for cid in CAMERAS
    ]
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()