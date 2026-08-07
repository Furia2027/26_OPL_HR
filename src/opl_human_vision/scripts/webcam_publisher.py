#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        self.declare_parameter('image_topic', 'camera/image_raw')
        self.declare_parameter('video_device', 0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('framerate', 30.0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        device_param = self.get_parameter('video_device').value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        fps = self.get_parameter('framerate').get_parameter_value().double_value

        try:
            video_device = int(device_param)
        except ValueError:
            video_device = str(device_param)

        # FIX: Publish using SensorDataQoS so the AI nodes don't backlog old frames
        self.publisher_ = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(video_device)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(f"Webcam initialized on [{video_device}] @ {actual_w}x{actual_h}")
        else:
            self.get_logger().error(f"Could not open webcam device [{video_device}]!")

        timer_period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.publisher_.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
