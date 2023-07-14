import rclpy
from rclpy.node import Node
from maploc.demo import Demo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import numpy as np
from maploc.utils.geo import BoundaryBox, Projection
from maploc.utils.wrappers import Camera
from maploc.osm.viz import GeoPlotter, Colormap
from maploc.osm.tiling import TileManager
from maploc.utils.viz_2d import features_to_RGB, plot_images
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
import matplotlib.pyplot as plt
from get_lat_lon import get_lat_lon


class OrienterNetNode(Node):
    def __init__(self):
        super().__init__("orienter_net_node")
        self.get_logger().info(f"Initializing ...")
        self.demo = Demo(num_rotations=256, device='cpu')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, "/sensing/camera/traffic_light/image_raw", self.image_callback, qos_profile)
        self.sub_camera_info = self.create_subscription(
            CameraInfo, "/sensing/camera/traffic_light/camera_info", self.camera_info_callback, qos_profile)
        self.sub_pose = self.create_subscription(
            PoseStamped, "/localization/pose_twist_fusion_filter/pose", self.pose_callback, qos_profile)
        self.latest_latlon = None
        self.get_logger().info(f"Ready")

    def image_callback(self, msg: Image):
        self.get_logger().info(f"Received image")
        image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        tile_size_meters = 128

        gravity = None
        if self.latest_latlon is None:
            self.get_logger().info(f"self.latest_latlon is None")
            return
        else:
            latlon = self.latest_latlon
        latlon = (35.68601896795643, 139.6890468576763)
        latlon = np.array(latlon)
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = BoundaryBox(center, center) + tile_size_meters
        prior_latlon = latlon

        h, w = image.shape[:2]
        f = 365
        camera = Camera.from_dict(
            dict(
                model="SIMPLE_PINHOLE",
                width=w,
                height=h,
                params=[f, w / 2 + 0.5, h / 2 + 0.5],
            )
        )

        # Show the query area in an interactive map
        plot = GeoPlotter(zoom=16)
        plot.points(prior_latlon[:2], "red", name="location prior", size=10)
        plot.bbox(proj.unproject(bbox), "blue", name="map tile")

        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(
            proj, bbox + 10, self.demo.config.data.pixel_per_meter)
        canvas = tiler.query(bbox)

        # Show the inputs to the model: image and raster map
        map_viz = Colormap.apply(canvas.raster)

        # Run the inference
        uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
            image, camera, canvas, roll_pitch=gravity)

        # Visualize the predictions
        overlay = likelihood_overlay(
            prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
        (neural_map_rgb,) = features_to_RGB(neural_map.numpy())
        plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
        ax = plt.gcf().axes[0]
        ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
        plot_dense_rotations(ax, prob, w=0.005, s=1/25)
        add_circle_inset(ax, uv)
        save_path = "./output.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved visualization to {save_path}")

    def camera_info_callback(self, msg: CameraInfo):
        pass
        # TODO
        # h, w = image.shape[:2]
        # f = 365
        # camera = Camera.from_dict(
        #     dict(
        #         model="SIMPLE_PINHOLE",
        #         width=w,
        #         height=h,
        #         params=[f, w / 2 + 0.5, h / 2 + 0.5],
        #     )
        # )

    def pose_callback(self, msg: PoseStamped):
        # 4x4行列に変換
        pose = msg.pose
        self.latest_latlon = get_lat_lon(pose.position.x, pose.position.y)


def main(args=None):
    rclpy.init(args=args)
    node = OrienterNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
