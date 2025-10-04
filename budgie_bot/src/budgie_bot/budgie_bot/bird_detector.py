import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger
import cv2
import importlib
import time
from datetime import datetime
import numpy as np

class BirdDetectorNode(Node):
    def __init__(self):
        super().__init__('bird_detector')

        # Parameters
        self.declare_parameter('detector_name', 'bg_subtract')
        self.declare_parameter('min_motion_area', 500)
        self.declare_parameter('font_scale', 0.7)
        self.declare_parameter('font_thickness', 2)
        self.declare_parameter('detection_fps', 5.0)

        detector_name = self.get_parameter('detector_name').value
        self.min_area = self.get_parameter('min_motion_area').value
        self.font_scale = self.get_parameter('font_scale').value
        self.font_thickness = self.get_parameter('font_thickness').value
        self.fps = self.get_parameter('detection_fps').value

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Load detector dynamically
        try:
            det_module = importlib.import_module(f"budgie_bot.detectors.{detector_name}")
            self.detector_fn = getattr(det_module, detector_name)
            self.get_logger().info(f"Loaded detector: {detector_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to load detector '{detector_name}': {e}")
            rclpy.shutdown()
            return

        # Publishers
        self.motion_pub = self.create_publisher(String, 'motion_detected', 10)
        self.image_pub = self.create_publisher(CompressedImage, 'motion_frame/compressed', 10)

        # Subscriber to compressed camera feed
        self.sub = self.create_subscription(CompressedImage, 'camera/image_raw/compressed', self.callback, 1)

        # Background reset service
        self.create_service(Trigger, 'set_background', self.handle_set_background)

        # Internal state
        self.state = self.detector_fn("init", None, self.min_area)
        self.last_inf_t = time.time()
        self.bg_reset = False
        self.last_motion = False

    def callback(self, msg: CompressedImage):
        # Throttle FPS
        now = time.time()
        if now - self.last_inf_t < 1.0 / self.fps:
            return
        self.last_inf_t = now

        # Decode JPEG to OpenCV frame
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Background reset if requested
        if self.bg_reset:
            self.state = self.detector_fn("set_bg", frame, self.min_area, self.state)
            self.bg_reset = False
            self.get_logger().info("Background reset completed.")

        # Motion detection
        motion, self.state = self.detector_fn("detect", frame, self.min_area, self.state)
        self.last_motion = bool(motion)
        self.motion_pub.publish(String(
            data=f"{datetime.now().isoformat(timespec='seconds')},motion={int(motion)}"
        ))

        # Annotate frame
        label = "BIRD" if motion else "NO BIRD"
        color = (0, 255, 0) if motion else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), self.font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

        # Publish compressed annotated frame
        out_msg = CompressedImage()
        out_msg.format = "jpeg"
        out_msg.data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        self.image_pub.publish(out_msg)

    def handle_set_background(self, request, response):
        self.bg_reset = True
        response.success = True
        response.message = "Background reset requested"
        return response


def main():
    rclpy.init()
    node = BirdDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
