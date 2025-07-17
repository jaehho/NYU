import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

__all__ = [
    "AudioVisualizer",
]

# ---------------------------------------------------------------------------
#  Low‑level helpers (kept private – do not export)
# ---------------------------------------------------------------------------
class _AmplitudeVisualizer:
    """Live line plot of block‑wise RMS amplitude."""

    def __init__(self, buffer_length: int, block_duration: float, ax):
        self.block_duration = block_duration
        self.data: Deque[float] = deque(maxlen=buffer_length)
        self.ax = ax
        self.line, = ax.plot([], [], "-o")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Avg amplitude")
        ax.set_xlim(0, buffer_length * block_duration)

    # --- public API used by *AudioVisualizer* ---------------------------------
    def add(self, amp: float):
        self.data.append(amp)

    def update(self, _frame):
        xs = [i * self.block_duration for i in range(len(self.data))]
        self.line.set_data(xs, list(self.data))
        if xs:
            self.ax.set_xlim(xs[0], xs[-1] + self.block_duration)
        return (self.line,)


class _SpectrogramVisualizer:
    """Live spectrogram view for a rolling mono buffer."""

    def __init__(
        self,
        buffer: np.ndarray,
        device_id: int,
        samplerate: int,
        block_duration: float,
        nfft: int,
        noverlap: int,
        cmap: str,
        vmin: float,
        vmax: float,
        ax,
    ):
        self.buffer = buffer
        self.device_id = device_id
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.nfft = nfft
        self.noverlap = noverlap
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.ax = ax

    def update(self, _frame):
        self.ax.clear()
        with np.errstate(divide="ignore"):
            _spectrum, _freqs, _t, im = self.ax.specgram(
                self.buffer,
                NFFT=self.nfft,
                Fs=self.samplerate,
                noverlap=self.noverlap,
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )
        self.ax.set_ylim(0, self.samplerate / 2)
        self.ax.set_title(f"Device {self.device_id} Spectrogram")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Frequency [Hz]")
        return (im,)



class _AudioCapture:
    """Multi‑device audio capture into rolling numpy buffers."""

    def __init__(
        self,
        devices: List[int],
        samplerate: int,
        block_duration: float,
        buffer_duration: float,
    ):
        self.devices = devices
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.buffer_duration = buffer_duration
        self.block_size = int(samplerate * block_duration)
        self.buffer_size = int(samplerate * buffer_duration)
        self.buffers = [np.full(self.buffer_size, 1e-6, float) for _ in devices]
        self._callbacks: List[Callable[[np.ndarray, int], None]] = []
        self._streams: List[sd.InputStream] = []

    # ---- public helpers ------------------------------------------------------
    def register_callback(self, fn: Callable[[np.ndarray, int], None]):
        self._callbacks.append(fn)

    def start(self):
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
        for s in self._streams:
            s.stop(); s.close()
        self._streams.clear()

    # ---- private -------------------------------------------------------------
    def _make_callback(self, idx: int):
        def _cb(indata, frames, _time, status):
            if status:
                print(f"[AudioCapture] Dev {self.devices[idx]}: {status}")
            buf = self.buffers[idx]
            buf[:] = np.roll(buf, -frames)
            buf[-frames:] = indata[:, 0] + 1e-6
            for fn in self._callbacks:
                fn(indata[:, 0].copy(), self.devices[idx])
        return _cb


# ---------------------------------------------------------------------------
#  Public facing class
# ---------------------------------------------------------------------------
class AudioVisualizer:
    """High‑level orchestrator: capture, plot, trigger and amplitude feed."""

    def __init__(
        self,
        devices: List[int],
        *,
        samplerate: int = 48_000,
        block_duration: float = 0.1,
        buffer_duration: float = 5.0,
        nfft: int = 1024,
        noverlap: int = 512,
        cmap: str = "viridis",
        vmin: float = -200.0,
        vmax: float = -50.0,
        amplitude_threshold: float = 0.04,
        amplitude_callback: Optional[Callable[[int, float], None]] = None,
    ):
        self.devices = devices
        self.latest_rms: Dict[int, float] = {d: 0.0 for d in devices}
        self._prev_rms = self.latest_rms.copy()
        self._amp_cb = amplitude_callback
        # Capture
        self._cap = _AudioCapture(devices, samplerate, block_duration, buffer_duration)
        # Figure / axes layout
        num_dev = len(devices)
        self.fig, axes = plt.subplots(num_dev, 2, figsize=(12, 5 * num_dev))
        self.fig.suptitle("Multi‑Device Audio Analysis", fontsize=16)
        if num_dev == 1:
            axes = axes.reshape(1, -1)
        # Helpers
        blen = int(buffer_duration / block_duration)
        self._amp_vis = {}
        self._spec_vis = {}
        for idx, dev in enumerate(devices):
            ax_spec, ax_amp = axes[idx]
            self._amp_vis[dev] = _AmplitudeVisualizer(blen, block_duration, ax_amp)
            ax_amp.set_title(f"Device {dev} – Amplitude")
            self._spec_vis[dev] = _SpectrogramVisualizer(
                buffer=self._cap.buffers[idx],
                device_id=dev,
                samplerate=samplerate,
                block_duration=block_duration,
                nfft=nfft,
                noverlap=noverlap,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax_spec,
            )
        # Register RMS callback
        self._cap.register_callback(self._make_rms_callback(amplitude_threshold))
        # Animation
        self._ani: Optional[FuncAnimation] = None

    # ---------------------------------------------------------------------
    #  Public control methods
    # ---------------------------------------------------------------------
    def start(self, block: bool = False):
        """Start streams and launch Matplotlib animation.

        If *block* is False (default) this returns immediately; the GUI event‑loop
        keeps running in the background.  If True, the method only returns once
        the user closes the Matplotlib window.
        """
        self._cap.start()
        self._ani = FuncAnimation(
            self.fig,
            self._update_all,
            interval=self._cap.block_duration * 1000,
            cache_frame_data=False,
        )
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.ion()
            self.fig.show()  # non‑blocking, ensures the canvas is realised
            plt.pause(0.001)  # forces an initial draw

    def stop(self):
        """Stop audio capture and close the figure (if still open)."""
        self._cap.stop()
        plt.close(self.fig)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------
    def _make_rms_callback(self, threshold: float):
        def _cb(block: np.ndarray, device_id: int):
            rms = float(np.sqrt((block ** 2).mean()))
            self.latest_rms[device_id] = rms
            self._amp_vis[device_id].add(rms)
            # rising‑edge trigger
            if self._prev_rms[device_id] < threshold <= rms:
                print(
                    f"[Trigger] Dev {device_id}: {self._prev_rms[device_id]:.4f} \u2192 {rms:.4f} (>{threshold})"
                )
            self._prev_rms[device_id] = rms
            if self._amp_cb:
                self._amp_cb(device_id, rms)
        return _cb

    def _update_all(self, frame):
        artists = []
        for dev in self.devices:
            artists += self._spec_vis[dev].update(frame)
            artists += self._amp_vis[dev].update(frame)
        return artists
