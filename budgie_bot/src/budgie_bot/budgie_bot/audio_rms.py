import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import sounddevice as sd
import numpy as np


class MultiMicAmplitudePublisher(Node):
    def __init__(self):
        super().__init__('multi_mic_amplitude_publisher')

        # Declare parameters
        self.declare_parameter('device_index', 0)
        self.declare_parameter('mic_name', 'mic0')
        self.declare_parameter('samplerate', 16000)
        self.declare_parameter('blocksize', 1024)

        self.device_index = self.get_parameter('device_index').value
        self.mic_name = self.get_parameter('mic_name').value
        self.samplerate = self.get_parameter('samplerate').value
        self.blocksize = self.get_parameter('blocksize').value

        self.publisher_ = self.create_publisher(Float32, 'audio_amplitude', 10)

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            callback=self.audio_callback,
        )

        self.stream.start()
        resolved_topic = self.resolve_topic_name('audio_amplitude')
        self.get_logger().info(
            f"Started mic '{self.mic_name}' on device {self.device_index}, publishing to {resolved_topic}"
        )

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.get_logger().warn(f'Audio stream status: {status}')
        amplitude = float(np.linalg.norm(indata) / np.sqrt(len(indata)))
        msg = Float32()
        msg.data = amplitude
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.stream.stop()
        self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiMicAmplitudePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
