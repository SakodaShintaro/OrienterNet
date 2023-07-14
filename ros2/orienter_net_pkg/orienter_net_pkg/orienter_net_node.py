import rclpy
from rclpy.node import Node


class OrienterNetNode(Node):
    def __init__(self):
        super().__init__("mile_node")
        self.get_logger().info(f"Initializing ...")
        self.get_logger().info(f"Ready")


def main(args=None):
    rclpy.init(args=args)
    node = OrienterNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
