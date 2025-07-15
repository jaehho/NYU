# main.py
from audio_stream import AudioCapture
from spectrogram import SpectrogramVisualizer
from amplitude import AmplitudeVisualizer
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Configuration
    devices = [3, 4]
    samplerate = 48000
    block_duration = 0.1
    buffer_duration = 5.0
    nfft = 1024
    noverlap = 512
    cmap = "viridis"
    vmin = -200.0  # dBFS
    vmax = -50.0   # dBFS
    
    # Trigger Configuration
    amplitude_threshold = 0.04  # Adjust this value based on your audio levels
    previous_amplitudes = {device_id: 0.0 for device_id in devices}  # Track previous RMS values

    # Audio capture & processing
    streamer = AudioCapture(
        devices=devices,
        samplerate=samplerate,
        block_duration=block_duration,
        buffer_duration=buffer_duration,
    )

    # Start audio capture first to initialize buffers
    streamer.start()

    # Create single figure with subplots for all devices
    amp_buffer_length = int(buffer_duration / block_duration)
    num_devices = len(devices)
    fig, axes = plt.subplots(num_devices, 2, figsize=(12, 5 * num_devices))
    fig.suptitle("Multi-Device Audio Analysis", fontsize=16)
    
    # Ensure axes is always 2D array for consistent indexing
    if num_devices == 1:
        axes = axes.reshape(1, -1)

    # Create separate visualizers using shared figure axes
    amp_vis_devices = {}
    spec_vis_devices = {}
    
    for idx, device_id in enumerate(devices):
        ax_spec = axes[idx, 0]  # Left column for spectrogram
        ax_amp = axes[idx, 1]   # Right column for amplitude
        
        # Create amplitude visualizer with the right subplot
        amp_vis_devices[device_id] = AmplitudeVisualizer(
            buffer_length=amp_buffer_length,
            block_duration=block_duration,
            ax=ax_amp
        )
        ax_amp.set_title(f"Device {device_id} - Amplitude")
        
        # Create spectrogram visualizer with the left subplot
        spec_vis_devices[device_id] = SpectrogramVisualizer(
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

    # processing callback: compute RMS and feed to respective amp_vis
    def rms_callback(block: np.ndarray, device_id: int):
        rms = np.sqrt((block**2).mean())
        amp_vis_devices[device_id].add_amplitude(rms)

        # Trigger Detection
        prev_amp = previous_amplitudes[device_id]
        
        # Check for rising edge above threshold
        if prev_amp < amplitude_threshold and rms >= amplitude_threshold:
            print(f"TRIGGER: Device {device_id} amplitude rising edge detected "
                  f"({prev_amp:.4f} -> {rms:.4f}, threshold: {amplitude_threshold})")
        
        # Update previous amplitude for next comparison
        previous_amplitudes[device_id] = rms

    # register callback
    streamer.register_callback(rms_callback)

    # Launch visualizations with single animation
    # Create a single animation that updates all visualizers
    from matplotlib.animation import FuncAnimation
    
    def update_all(frame):
        # Update all spectrogram visualizers manually
        for device_id in devices:
            spec_vis_devices[device_id]._update(frame)
            amp_vis_devices[device_id]._update(frame)
        return []
    
    # Create single animation for the shared figure
    max_frames = int(streamer.buffers[0].size / (samplerate * block_duration))
    ani = FuncAnimation(
        fig,
        update_all,
        interval=block_duration * 1000,
        save_count=max_frames,
        cache_frame_data=False,
    )
    
    plt.tight_layout()
    plt.show()

    # Cleanup on exit
    streamer.stop()
