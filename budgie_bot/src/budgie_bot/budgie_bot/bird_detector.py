import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from custom_interfaces.srv import SetBG

import cv2
import time
import threading
import queue
import importlib
from datetime import datetime


class BirdDetector(Node):
    def __init__(self):
        super().__init__('camera_motion_node')

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_ids', ['0']),
                ('inference_fps', 5.0),
                ('min_motion_area', 500),
                ('detection_mode', 'bg_subtract'),
                ('font_scale', 0.7),
                ('font_thickness', 2),
            ]
        )

        self.camera_ids = self.get_parameter('camera_ids').get_parameter_value().string_array_value
        self.fps = self.get_parameter('inference_fps').value
        self.min_area = self.get_parameter('min_motion_area').value
        self.mode = self.get_parameter('detection_mode').value
        self.font_scale = self.get_parameter('font_scale').value
        self.font_thickness = self.get_parameter('font_thickness').value

        self.bridge = CvBridge()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Import detector
        try:
            det_module = importlib.import_module(f"budgie_bot.detectors.{self.mode}")
            self.detector = getattr(det_module, self.mode)
        except Exception as e:
            self.get_logger().error(f"Detector '{self.mode}' not found: {e}")
            rclpy.shutdown()
            return

        # Queues and threads
        self.frame_queues = [queue.Queue(maxsize=1) for _ in self.camera_ids]
        self.ctrl_queues = [queue.Queue(maxsize=5) for _ in self.camera_ids]

        self.threads = []
        for cam_id, fq, cq in zip(self.camera_ids, self.frame_queues, self.ctrl_queues):
            t = threading.Thread(target=self.camera_worker, args=(cam_id, fq, cq), daemon=True)
            t.start()
            self.threads.append(t)

        # Publishers
        self.motion_pub = self.create_publisher(String, 'motion_detected', 10)
        self.image_pub = self.create_publisher(Image, 'motion_frame', 10)

        # Timer
        self.timer = self.create_timer(1.0 / self.fps, self.display_frames)

        # Service
        self.srv = self.create_service(SetBG, 'set_background', self.handle_set_background)

        self.get_logger().info("CameraMotionNode started.")

    def camera_worker(self, cam_id, frame_q, ctrl_q):
        cap = cv2.VideoCapture(int(cam_id)) if cam_id.isdigit() else cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            self.get_logger().error(f"Cannot open camera {cam_id}")
            return

        ok, first_frame = cap.read()
        if not ok:
            self.get_logger().error(f"Camera {cam_id} returned no frames.")
            return

        state = self.detector("init", first_frame, self.min_area)
        period = 1.0 / self.fps
        last_inf_t = 0.0
        last_motion = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_inf_t >= period:
                last_inf_t = now
                motion, state = self.detector("detect", frame, self.min_area, state)
                last_motion = bool(motion)
                msg = f"{datetime.now().isoformat(timespec='seconds')},{cam_id},{int(motion)}"
                self.motion_pub.publish(String(data=msg))

            try:
                msg = ctrl_q.get_nowait()
                if msg == "set_bg":
                    state = self.detector("set_bg", frame, self.min_area, state)
            except queue.Empty:
                pass

            # Annotate
            label = "BIRD" if last_motion else "NO BIRD"
            color = (0, 255, 0) if last_motion else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), self.font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

            # draw BG thumbnail
            bg_img = state.get("bg") if isinstance(state, dict) else None
            if bg_img is not None:
                H, W = frame.shape[:2]
                inset_h, inset_w = H // 4, W // 4
                bg_thumb = cv2.resize(bg_img, (inset_w, inset_h))
                y1, y2 = H - inset_h - 10, H - 10
                x1, x2 = W - inset_w - 10, W - 10
                roi = frame[y1:y2, x1:x2]
                cv2.addWeighted(bg_thumb, 0.4, roi, 0.6, 0, dst=roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(frame, "BG", (x1 + 4, y1 + 14), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            frame_q.put_nowait((cam_id, frame))

        cap.release()

    def display_frames(self):
        for cam_id, fq in zip(self.camera_ids, self.frame_queues):
            try:
                cam_id, frame = fq.get_nowait()
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.image_pub.publish(msg)
                cv2.imshow(f"Cam {cam_id}", frame)
                cv2.waitKey(1)
            except queue.Empty:
                pass

    def handle_set_background(self, request, response):
        try:
            idx = self.camera_ids.index(request.camera_id)
            self.ctrl_queues[idx].put_nowait("set_bg")
            response.success = True
            response.message = f"Set background for camera {request.camera_id}"
        except ValueError:
            response.success = False
            response.message = f"Camera ID '{request.camera_id}' not found"
        except queue.Full:
            response.success = False
            response.message = f"Control queue full for camera {request.camera_id}"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BirdDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
