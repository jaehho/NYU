import cv2, numpy as np

def snapshot(event: str, frame, min_area=500, state=None):
    if state is None: state = {"bg": None}
    if event == "init":
        return state
    if event == "set_bg":
        state["bg"] = frame.copy(); return state
    if event == "reset":
        state["bg"] = None; return state
    if event == "detect":
        bg = state.get("bg");
        if bg is None: return False, state
        diff = cv2.absdiff(bg, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > min_area for c in contours)
        return motion, state
    return state