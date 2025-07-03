# audio_stream.py
import numpy as np
import sounddevice as sd
from typing import List, Callable

class AudioCapture:
    """
    Captures audio from multiple input devices into rolling buffers,
    and allows registering processing callbacks.
    """

    def __init__(
        self,
        devices: List[int],
        samplerate: int = 44100,
        block_duration: float = 0.1,
        buffer_duration: float = 5.0,
    ):
        self.devices = devices
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.buffer_duration = buffer_duration

        self.block_size = int(self.samplerate * self.block_duration)
        self.buffer_size = int(self.samplerate * self.buffer_duration)

        # One rolling buffer per device
        self.buffers: List[np.ndarray] = [
            np.full(self.buffer_size, 1e-6, dtype=float) for _ in devices
        ]

        # User‐provided processing callbacks:
        # fn(block: np.ndarray, device_id: int) -> None
        self._callbacks: List[Callable[[np.ndarray, int], None]] = []

        self._streams: List[sd.InputStream] = []

    def register_callback(self, fn: Callable[[np.ndarray, int], None]):
        """Register a function to be called on each incoming audio block."""
        self._callbacks.append(fn)

    def _make_callback(self, idx: int):
        def _cb(indata, frames, time, status):
            if status:
                print(f"[AudioStreamer] Device {self.devices[idx]} status: {status}")
            buf = self.buffers[idx]
            buf[:] = np.roll(buf, -frames)
            buf[-frames:] = indata[:, 0] + 1e-6

            for fn in self._callbacks:
                fn(indata[:, 0].copy(), self.devices[idx])

        return _cb

    def start(self):
        """Open one InputStream per device and begin capturing."""
        for idx, dev in enumerate(self.devices):
            stream = sd.InputStream(
                device=dev,
                samplerate=self.samplerate,
                blocksize=self.block_size,
                channels=1,
                callback=self._make_callback(idx),
            )
            stream.start()
            self._streams.append(stream)

    def stop(self):
        """Stop and close all streams."""
        for stream in self._streams:
            stream.stop()
            stream.close()
        self._streams.clear()
