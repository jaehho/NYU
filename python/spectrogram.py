# spectrogram.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import numpy as np

class SpectrogramVisualizer:
    """
    Displays real-time spectrogram for a single device buffer.
    """

    def __init__(
        self,
        buffer: np.ndarray,
        device_id: int,
        samplerate: int = 44100,
        block_duration: float = 0.1,
        nfft: int = 1024,
        noverlap: int = 512,
        cmap: str = "viridis",
        vmin: float = -200.0,
        vmax: float = -50.0,
        ax=None,
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

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 4))
            self.owns_fig = True
        else:
            self.ax = ax
            self.fig = ax.figure
            self.owns_fig = False

    def _update(self, frame):
        self.ax.clear()
        with np.errstate(divide="ignore"):
            self.ax.specgram(
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

    def start(self):
        """Set up the animation."""
        max_frames = int(self.buffer.size / (self.samplerate * self.block_duration))
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            interval=self.block_duration * 1000,
            save_count=max_frames,
        )
        if self.owns_fig:
            plt.tight_layout()
