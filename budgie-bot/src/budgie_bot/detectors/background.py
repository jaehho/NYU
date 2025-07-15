import cv2

def background(event: str, frame, min_area=500, state=None):
    if state is None or event in ("init", "reset"):
        state = {"mog2": cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)}
        return state if event != "detect" else (False, state)
    if event == "detect":
        mask = state["mog2"].apply(frame, learningRate=0.001)
        _, th = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > min_area for c in contours)
        return motion, state
    return state