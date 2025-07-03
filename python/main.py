# main.py
from audio_stream import AudioCapture
from spectrogram import SpectrogramVisualizer, SingleDeviceSpectrogramVisualizer
from amplitude import AmplitudeVisualizer
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────
    devices = [3, 4]
    samplerate = 48000
    block_duration = 0.1
    buffer_duration = 5.0
    nfft = 1024
    noverlap = 512
    cmap = "viridis"
    vmin = -200.0  # dBFS
    vmax = -50.0   # dBFS

    # ── Create figures with subplots for each device ──────────────
    amp_buffer_length = int(buffer_duration / block_duration)
    device_figures = {}
    amp_vis_devices = {}
    spec_vis_devices = {}
    
    for device_id in devices:
        # Create figure with 2 subplots for each device
        fig, (ax_spec, ax_amp) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Device {device_id} - Audio Analysis")
        device_figures[device_id] = fig
        
        # Create amplitude visualizer with the right subplot
        amp_vis_devices[device_id] = AmplitudeVisualizer(
            buffer_length=amp_buffer_length,
            block_duration=block_duration,
            ax=ax_amp
        )
        ax_amp.set_title("Amplitude")
        
        # We'll create the spectrogram visualizer after audio capture starts
        # since it needs access to the buffer

    # ── Audio capture & processing ────────────────────────────
    streamer = AudioCapture(
        devices=devices,
        samplerate=samplerate,
        block_duration=block_duration,
        buffer_duration=buffer_duration,
    )

    # processing callback: compute RMS and feed to respective amp_vis
    def rms_callback(block: np.ndarray, device_id: int):
        rms = np.sqrt((block**2).mean())
        amp_vis_devices[device_id].add_amplitude(rms)

    # register and start
    streamer.register_callback(rms_callback)
    streamer.start()

    # now that streamer.buffers exist, create spectrogram visualizers
    for idx, device_id in enumerate(devices):
        fig = device_figures[device_id]
        ax_spec = fig.axes[0]  # Left subplot for spectrogram
        
        spec_vis_devices[device_id] = SingleDeviceSpectrogramVisualizer(
            buffer=streamer.buffers[idx],
            device_id=device_id,
            samplerate=samplerate,
            block_duration=block_duration,
            nfft=nfft,
            noverlap=noverlap,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax_spec,
        )

    # ── Launch all animations, then block on show ──────────────
    for device_id in devices:
        spec_vis_devices[device_id].start()
        amp_vis_devices[device_id].start()
        plt.figure(device_figures[device_id].number)  # Bring figure to front
        plt.tight_layout()
    
    plt.show()

    # ── Cleanup on exit ───────────────────────────────────────
    streamer.stop()
