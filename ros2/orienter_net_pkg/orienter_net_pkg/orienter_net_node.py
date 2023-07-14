import rclpy
from rclpy.node import Node
from maploc.demo import Demo


class OrienterNetNode(Node):
    def __init__(self):
        super().__init__("orienter_net_node")
        self.get_logger().info(f"Initializing ...")
        self.demo = Demo(num_rotations=256, device='cpu')
        self.get_logger().info(f"Ready")


def main(args=None):
    rclpy.init(args=args)
    node = OrienterNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
