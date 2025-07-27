import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import pyfirmata2
import sounddevice as sd
import numpy as np


class ReactBehaviorNode(Node):
    def __init__(self):
        super().__init__('react_behavior')

        # Declare parameters with defaults
        self.declare_parameter('amplitude_threshold', 0.05)
        self.declare_parameter('device_path', 'None')  # Will be string 'None' if not set
        self.declare_parameter('motor_pwm_pin', 9)
        self.declare_parameter('motor_on_duty', 1.0)
        self.declare_parameter('motor_off_duty', 0.0)
        self.declare_parameter('amplitude_topic', '/mic0/audio_amplitude')

        self.declare_parameter('chirp_samplerate', 44100)
        self.declare_parameter('chirp_duration', 0.2)
        self.declare_parameter('chirp_freq_start', 500.0)
        self.declare_parameter('chirp_freq_end', 2000.0)
        self.declare_parameter('chirp_volume', 0.3)

        # Load parameter values
        self.threshold = self.get_parameter('amplitude_threshold').value
        self.device_path = self.get_parameter('device_path').value
        self.motor_pwm_pin = self.get_parameter('motor_pwm_pin').value
        self.motor_on_duty = self.get_parameter('motor_on_duty').value
        self.motor_off_duty = self.get_parameter('motor_off_duty').value
        self.amplitude_topic = self.get_parameter('amplitude_topic').value

        self.fs = self.get_parameter('chirp_samplerate').value
        self.chirp_duration = self.get_parameter('chirp_duration').value
        self.chirp_f_start = self.get_parameter('chirp_freq_start').value
        self.chirp_f_end = self.get_parameter('chirp_freq_end').value
        self.chirp_volume = self.get_parameter('chirp_volume').value

        # Generate chirp sound
        self.chirp_wave = self.generate_chirp()

        # Try to connect to Arduino
        self.board = None
        self.pwm_pin = None

        try:
            if self.device_path.lower() != 'none':
                self.board = pyfirmata2.Arduino(self.device_path)
                self.pwm_pin = self.board.get_pin(f'd:{self.motor_pwm_pin}:p')
                self.get_logger().info(f"Arduino connected on {self.device_path}, using pin D{self.motor_pwm_pin}")
            else:
                self.get_logger().warn("Parameter 'device_path' set to 'None'; skipping Arduino connection.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Arduino: {e}")
            self.get_logger().warn("Continuing without motor control. Only chirp will function.")

        # Motor state
        self.motor_state = False

        # ROS 2 subscription
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
            self.get_logger().info(f"Amplitude {amplitude:.3f} > threshold {self.threshold:.3f}; motor ON + chirp")
        elif amplitude <= self.threshold and self.motor_state:
            self.set_motor_pwm(self.motor_off_duty)
            self.motor_state = False
            self.get_logger().info(f"Amplitude {amplitude:.3f} <= threshold; motor OFF")

    def set_motor_pwm(self, duty):
        if self.pwm_pin is not None:
            duty = max(0.0, min(1.0, duty))
            self.pwm_pin.write(duty)
        else:
            self.get_logger().debug("Motor PWM not available; skipping motor write.")

    def destroy_node(self):
        self.set_motor_pwm(0.0)
        if self.board:
            self.board.exit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ReactBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
