import sounddevice as sd
import numpy as np

def test_device(device_index):
    try:
        # Try opening the stream for the given device
        with sd.InputStream(device=device_index, channels=1, samplerate=44100, blocksize=1024):
            print(f"✅ Working input device: {device_index} - {sd.query_devices(device_index)['name']}")
            return True
    except Exception as e:
        return False

def main():
    print("🔍 Testing all available input audio devices...\n")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            test_device(idx)

if __name__ == "__main__":
    main()
