import sys, os
import cv2
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (e.g., libjpeg warnings)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Use it for camera capture
with suppress_stderr():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("❌ Could not open the camera.")
    exit()

print("🎥 Starting live video. Press 'q' to quit.")

while True:
    with suppress_stderr():  # suppress decoding warnings
        ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("⌨️  'q' pressed. Exiting.")
        break

    if cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
        print("🪟 Window closed. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera released and window closed.")
