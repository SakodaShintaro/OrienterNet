from maploc.utils.viz_2d import features_to_RGB
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import plot_images
from maploc.osm.viz import Colormap
from maploc.osm.tiling import TileManager
from maploc.osm.viz import GeoPlotter
import matplotlib.pyplot as plt
from maploc.demo import Demo
import argparse
from maploc.utils.wrappers import Camera
from maploc.utils.io import read_image
import numpy as np
from maploc.utils.geo import BoundaryBox, Projection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path

    # Increasing the number of rotations increases the accuracy but requires more GPU memory.
    # The highest accuracy is achieved with num_rotations=360
    # but num_rotations=64~128 is often sufficient.
    # To reduce the memory usage, we can reduce the tile size in the next cell.
    demo = Demo(num_rotations=256, device='cpu')

    tile_size_meters = 128

    image = read_image(image_path)
    gravity = None
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
    tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
    canvas = tiler.query(bbox)

    # Show the inputs to the model: image and raster map
    map_viz = Colormap.apply(canvas.raster)

    # Run the inference
    uv, yaw, prob, neural_map, image_rectified = demo.localize(
        image, camera, canvas, roll_pitch=gravity)

    # Visualize the predictions
    overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
    (neural_map_rgb,) = features_to_RGB(neural_map.numpy())
    plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
    ax = plt.gcf().axes[0]
    ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
    plot_dense_rotations(ax, prob, w=0.005, s=1/25)
    add_circle_inset(ax, uv)
    save_path = "./output.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved visualization to {save_path}")
