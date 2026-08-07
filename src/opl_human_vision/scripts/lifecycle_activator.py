#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
import time

class LifecycleActivator(Node):
    def __init__(self):
        super().__init__('lifecycle_activator')
        self.nodes_to_activate = [
            '/human_detector_node',
            '/human_tracker_node',
            '/face_recognizer_node'
        ]

    def transition_node(self, node_name, transition_id, transition_label):
        srv_name = f'{node_name}/change_state'
        client = self.create_client(ChangeState, srv_name)

        if not client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(f"Service {srv_name} not available. Node might have crashed during GPU load.")
            return False

        req = ChangeState.Request()
        req.transition.id = transition_id
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if res is not None and res.success:
            self.get_logger().info(f"Successfully {transition_label} {node_name}")
            return True
        else:
            self.get_logger().error(f"Failed to {transition_label} {node_name}!")
            return False

    def activate_all(self):
        # Give container a moment to initialize the nodes
        time.sleep(2.0)

        for node_name in self.nodes_to_activate:
            if not self.transition_node(node_name, Transition.TRANSITION_CONFIGURE, "configured"):
                continue
            self.transition_node(node_name, Transition.TRANSITION_ACTIVATE, "activated")

def main(args=None):
    rclpy.init(args=args)
    activator = LifecycleActivator()
    activator.activate_all()
    activator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
