import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import importlib
import cv2
import time
from datetime import datetime


class SingleCameraNode(Node):
    def __init__(self):
        super().__init__('single_camera_node')

        self.declare_parameter('camera_id', '0')
        self.declare_parameter('inference_fps', 5.0)
        self.declare_parameter('min_motion_area', 500)
        self.declare_parameter('detection_mode', 'bg_subtract')
        self.declare_parameter('font_scale', 0.7)
        self.declare_parameter('font_thickness', 2)

        self.cam_id = self.get_parameter('camera_id').value
        self.fps = self.get_parameter('inference_fps').value
        self.min_area = self.get_parameter('min_motion_area').value
        self.mode = self.get_parameter('detection_mode').value
        self.font_scale = self.get_parameter('font_scale').value
        self.font_thickness = self.get_parameter('font_thickness').value

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bridge = CvBridge()

        self.motion_pub = self.create_publisher(String, 'motion_detected', 10)
        self.image_pub = self.create_publisher(Image, 'motion_frame', 10)
        
        # Load detector
        try:
            det_module = importlib.import_module(f"budgie_bot.detectors.{self.mode}")
            self.detector = getattr(det_module, self.mode)
        except Exception as e:
            self.get_logger().error(f"Detector '{self.mode}' not found: {e}")
            rclpy.shutdown()
            return

        self.cap = cv2.VideoCapture(int(self.cam_id)) if self.cam_id.isdigit() else cv2.VideoCapture(self.cam_id)
        
        # Set video codec for WSL2 compatibility (comment out if not needed)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera {self.cam_id}")
            rclpy.shutdown()
            return

        ok, first_frame = self.cap.read()
        if not ok:
            self.get_logger().error(f"Camera {self.cam_id} returned no frames")
            rclpy.shutdown()
            return

        self.state = self.detector("init", first_frame, self.min_area)
        self.last_inf_t = time.time()
        self.last_motion = False
        self.bg_reset = False

        self.create_timer(1.0 / self.fps, self.process_frame)
        self.create_service(Trigger, f'/cam/cam{self.cam_id}/set_background', self.handle_set_background)

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        now = time.time()
        if now - self.last_inf_t >= 1.0 / self.fps:
            self.last_inf_t = now
            motion, self.state = self.detector("detect", frame, self.min_area, self.state)
            self.last_motion = bool(motion)
            msg = f"{datetime.now().isoformat(timespec='seconds')},{self.cam_id},{int(motion)}"
            self.motion_pub.publish(String(data=msg))

        if self.bg_reset:
            self.state = self.detector("set_bg", frame, self.min_area, self.state)
            self.bg_reset = False

        label = "BIRD" if self.last_motion else "NO BIRD"
        color = (0, 255, 0) if self.last_motion else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), self.font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)
        
        # Draw BG thumbnail inset
        bg_img = self.state.get("bg") if isinstance(self.state, dict) else None
        if bg_img is not None:
            H, W = frame.shape[:2]
            inset_h, inset_w = H // 4, W // 4  # Quarter size thumbnail
            bg_thumb = cv2.resize(bg_img, (inset_w, inset_h))

            # Position: bottom right with 10px margin
            y1, y2 = H - inset_h - 10, H - 10
            x1, x2 = W - inset_w - 10, W - 10

            # Blend for transparency
            roi = frame[y1:y2, x1:x2]
            cv2.addWeighted(bg_thumb, 0.4, roi, 0.6, 0, dst=roi)

            # Optional border and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, "BG", (x1 + 4, y1 + 14),
                        self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(msg)
        cv2.imshow(f"Cam {self.cam_id}", frame)
        cv2.waitKey(1)

    def handle_set_background(self, request, response):
        self.bg_reset = True
        response.success = True
        response.message = f"Background reset for camera {self.cam_id}"
        return response


def main():
    rclpy.init()
    node = SingleCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
