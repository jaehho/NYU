import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2

class Camera(Node):
    def __init__(self):
        super().__init__('camera')

        # Parameters
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('fps', 30.0)
        self.cam_id = self.get_parameter('camera_id').value
        self.fps = self.get_parameter('fps').value

        # Publisher for compressed image
        self.pub = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 1)

        # Open camera
        self.cap = cv2.VideoCapture(int(self.cam_id)) if self.cam_id.isdigit() else cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera {self.cam_id}")
            rclpy.shutdown()
            return

        # Timer for frame publishing
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Encode as JPEG for CompressedImage
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Camera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
