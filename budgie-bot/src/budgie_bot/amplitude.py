# amplitude.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from typing import Deque

class AmplitudeVisualizer:
    """
    Displays a live feed of average block amplitudes.
    """

    def __init__(self, buffer_length: int, block_duration: float, ax=None):
        self.block_duration = block_duration
        self.data: Deque[float] = deque(maxlen=buffer_length)

        if ax is None:
            self.fig, self.ax = plt.subplots()
            self.owns_fig = True
        else:
            self.ax = ax
            self.fig = ax.figure
            self.owns_fig = False
            
        self.line, = self.ax.plot([], [], '-o')
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Avg amplitude")
        # self.ax.set_ylim(0, 1.0)
        self.ax.set_xlim(0, buffer_length * block_duration)

    def add_amplitude(self, amp: float):
        """Call this every time you compute a new block’s average amplitude."""
        self.data.append(amp)

    def _update(self, frame):
        xs = [i * self.block_duration for i in range(len(self.data))]
        self.line.set_data(xs, list(self.data))
        if xs:
            self.ax.set_xlim(xs[0], xs[-1] + self.block_duration)
        return (self.line,)

    def start(self):
        """Set up the animation with an explicit save_count to suppress warnings."""
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            interval=self.block_duration * 1000,
            save_count=self.data.maxlen,
            cache_frame_data=False
        )
        if self.owns_fig:
            plt.tight_layout()
