from budgie_bot.audio_visualizer import AudioVisualizer
import time

# Optional: define a callback to handle real-time amplitude updates
def on_amplitude(device_id: int, rms: float):
    print(f"[Callback] Device {device_id} RMS: {rms:.4f}")

# Create the visualizer
vis = AudioVisualizer(
    devices=[3, 4],                  # Change to match your input device IDs
    amplitude_threshold=0.04,       # Trigger threshold
    amplitude_callback=on_amplitude
)

# Start the visualizer (non-blocking: figure stays open while script continues)
vis.start(block=True)

try:
    # Main application loop (10 seconds in this example)
    for _ in range(100):
        time.sleep(0.1)
        for dev, val in vis.latest_rms.items():
            print(f"[MainLoop] Device {dev} RMS: {val:.4f}")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    vis.stop()
