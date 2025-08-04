import sounddevice as sd
import numpy as np
import time

last_print_time = 0
print_interval = 0.2  # seconds

def audio_callback(indata, frames, time_info, status):
    global last_print_time
    if status:
        print(status)
    current_time = time.time()
    if current_time - last_print_time >= print_interval:
        volume_norm = np.linalg.norm(indata) * 10
        print(f"🔊 Mic level: {volume_norm:.2f}")
        last_print_time = current_time

print("🎛️ Real-time mic level monitoring... Press Ctrl+C to stop.")
try:
    with sd.InputStream(callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("🛑 Monitoring stopped.")
