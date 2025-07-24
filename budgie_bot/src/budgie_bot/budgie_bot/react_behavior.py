import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import pyfirmata2
import sounddevice as sd
import numpy as np


class MotorTriggerArduinoNode(Node):
    def __init__(self):
        super().__init__('motor_trigger_arduino_node')

        # Declare parameters
        self.declare_parameter('amplitude_threshold', 0.05)
        self.declare_parameter('motor_pwm_pin', 9)
        self.declare_parameter('motor_on_duty', 1.0)
        self.declare_parameter('motor_off_duty', 0.0)
        self.declare_parameter('device_path', pyfirmata2.Arduino.AUTODETECT)
        self.declare_parameter('amplitude_topic', 'audio_amplitude')
        self.declare_parameter('chirp_samplerate', 44100)
        self.declare_parameter('chirp_duration', 0.2)       # seconds
        self.declare_parameter('chirp_freq_start', 500.0)   # Hz
        self.declare_parameter('chirp_freq_end', 2000.0)    # Hz
        self.declare_parameter('chirp_volume', 0.3)

        # Load parameters
        self.threshold = self.get_parameter('amplitude_threshold').value
        self.motor_pwm_pin = self.get_parameter('motor_pwm_pin').value
        self.motor_on_duty = self.get_parameter('motor_on_duty').value
        self.motor_off_duty = self.get_parameter('motor_off_duty').value
        self.device_path = self.get_parameter('device_path').value
        self.amplitude_topic = self.get_parameter('amplitude_topic').value

        # Chirp parameters
        self.fs = self.get_parameter('chirp_samplerate').value
        self.chirp_duration = self.get_parameter('chirp_duration').value
        self.chirp_f_start = self.get_parameter('chirp_freq_start').value
        self.chirp_f_end = self.get_parameter('chirp_freq_end').value
        self.chirp_volume = self.get_parameter('chirp_volume').value

        # Precompute chirp signal
        self.chirp_wave = self.generate_chirp()

        # Setup Arduino
        try:
            self.board = pyfirmata2.Arduino(self.device_path)
            self.pwm_pin = self.board.get_pin(f'd:{self.motor_pwm_pin}:p')
            self.get_logger().info(f"Arduino connected on {self.device_path}, using pin D{self.motor_pwm_pin}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Arduino: {e}")
            raise e

        self.motor_state = False
        self.subscription = self.create_subscription(
            Float32,
            self.amplitude_topic,
            self.amplitude_callback,
            10
        )

    def generate_chirp(self):
        t = np.linspace(0, self.chirp_duration, int(self.fs * self.chirp_duration), endpoint=False)
        freqs = np.linspace(self.chirp_f_start, self.chirp_f_end, len(t))
        waveform = np.sin(2 * np.pi * freqs * t) * self.chirp_volume
        return waveform.astype(np.float32)

    def play_chirp(self):
        try:
            sd.play(self.chirp_wave, samplerate=self.fs)
        except Exception as e:
            self.get_logger().warn(f"Chirp playback failed: {e}")

    def amplitude_callback(self, msg):
        amplitude = msg.data
        if amplitude > self.threshold and not self.motor_state:
            self.set_motor_pwm(self.motor_on_duty)
            self.play_chirp()
            self.motor_state = True
            self.get_logger().info(f"Amplitude {amplitude:.3f} > threshold; motor ON + chirp")
        elif amplitude <= self.threshold and self.motor_state:
            self.set_motor_pwm(self.motor_off_duty)
            self.motor_state = False
            self.get_logger().info(f"Amplitude {amplitude:.3f} <= threshold; motor OFF")

    def set_motor_pwm(self, duty):
        duty = max(0.0, min(1.0, duty))  # Clamp
        self.pwm_pin.write(duty)

    def destroy_node(self):
        self.set_motor_pwm(0.0)
        if hasattr(self, 'board'):
            self.board.exit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorTriggerArduinoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
