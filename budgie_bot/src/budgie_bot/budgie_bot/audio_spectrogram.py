import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from collections import deque

class SpectrogramImageViewer(Node):
    def __init__(self):
        super().__init__('spectrogram_image_viewer')

        self.declare_parameter('mic_name', 'mic0')
        self.declare_parameter('samplerate', 16000)
        self.declare_parameter('fft_size', 512)
        self.declare_parameter('history_len', 300) # TODO: Doesn't seem to actually change width of spectrogram

        self.mic_name = self.get_parameter('mic_name').value
        self.samplerate = self.get_parameter('samplerate').value
        self.fft_size = self.get_parameter('fft_size').value
        self.history_len = self.get_parameter('history_len').value

        self.namespace = self.get_namespace().strip('/')
        self.topic_name = f'/{self.namespace}/audio_buffer'

        self.buffer_history = deque(maxlen=self.history_len)

        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.topic_name,
            self.buffer_callback,
            10
        )

        self.create_timer(0.1, self.display_spectrogram)

    def buffer_callback(self, msg):
        signal = np.array(msg.data)
        if len(signal) < self.fft_size:
            signal = np.pad(signal, (0, self.fft_size - len(signal)))
        spectrum = np.abs(np.fft.rfft(signal[:self.fft_size]))
        self.buffer_history.append(spectrum)

    def display_spectrogram(self):
        if len(self.buffer_history) < 2:
            return

        spec_matrix = np.stack(self.buffer_history, axis=1)
        spec_db = 20 * np.log10(spec_matrix + 1e-8)
        norm_spec = np.interp(spec_db, (spec_db.min(), spec_db.max()), (0, 255)).astype(np.uint8)

        color_img = cv2.applyColorMap(norm_spec, cv2.COLORMAP_TURBO)
        window_name = f"Spectrogram - {self.mic_name}"
        cv2.imshow(window_name, color_img)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SpectrogramImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
