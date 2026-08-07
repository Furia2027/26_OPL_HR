#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from opl_interfaces.msg import TrackedHumanArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

class PipelineVisualizer(Node):
    def __init__(self):
        super().__init__('pipeline_visualizer')
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', 'camera/camera/color/image_raw')
        self.declare_parameter('recognized_topic', 'recognized_humans')
        self.declare_parameter('show_depth', True)
        self.show_depth = self.get_parameter('show_depth').get_parameter_value().bool_value

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        recognized_topic = self.get_parameter('recognized_topic').get_parameter_value().string_value

        self.latest_frame = None
        self.latest_humans = []
        self.frame_updated = False
        self.first_frame_received = False

        # Create a dummy frame so the window pops up immediately
        self.rendered_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.rendered_frame, "Waiting for Camera...", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            image_qos
        )

        self.human_sub = self.create_subscription(
            TrackedHumanArray,
            recognized_topic,
            self.recognized_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f"Visualizer subscribed to '{image_topic}' & '{recognized_topic}'")

    def image_callback(self, msg):
        try:
            if not self.first_frame_received:
                self.get_logger().info(f"First image received! Resolution: {msg.width}x{msg.height}")
                self.first_frame_received = True

            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_updated = True
        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")

    def recognized_callback(self, human_msg):
        self.latest_humans = human_msg.humans

    def render_frame(self):
        if self.frame_updated and self.latest_frame is not None:
            self.frame_updated = False

            display_frame = self.latest_frame.copy()
            humans = self.latest_humans
            img_h, img_w = display_frame.shape[:2]

            for human in humans:
                x, y = int(human.bbox.x_offset), int(human.bbox.y_offset)
                w, h = int(human.bbox.width), int(human.bbox.height)

                if x > 2147483647: x -= 4294967296
                if y > 2147483647: y -= 4294967296

                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))

                name_lower = human.name.lower()
                if "scanning" in name_lower:
                    color = (0, 215, 255)
                elif name_lower in ["unknown", ""]:
                    color = (128, 128, 128)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                landmarks = getattr(human, 'landmarks', [])
                for lm in landmarks:
                    cx, cy = int(lm.x), int(lm.y)
                    if 0 <= cx < img_w and 0 <= cy < img_h:
                        cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

                # --- CONSTRUCT LABEL WITH DEPTH ---
                label = f"ID:{human.track_id} | {human.name} ({human.confidence:.2f})"

                if self.show_depth and hasattr(human, 'position_3d'):
                    z_dist = human.position_3d.z
                    if z_dist > 0.0:
                        label += f" | {z_dist:.2f}m"

                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.55, 2)
                label_y = max(y - 8, text_h + 5)

                cv2.rectangle(display_frame, (x, label_y - text_h - 4), (x + text_w + 6, label_y + baseline), color, cv2.FILLED)
                text_color = (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255)
                cv2.putText(display_frame, label, (x + 3, label_y - 2), font, 0.55, text_color, 2)

            self.rendered_frame = display_frame

        if self.rendered_frame is not None:
            cv2.imshow("OPL Human Vision Pipeline", self.rendered_frame)


def main(args=None):
    rclpy.init(args=args)
    node = PipelineVisualizer()

    try:
        while rclpy.ok():
            # Process one callback (if available), don't block longer than 2ms
            rclpy.spin_once(node, timeout_sec=0.002)

            # Render frame
            node.render_frame()

            # Process GTK key/window events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press ESC to exit
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
