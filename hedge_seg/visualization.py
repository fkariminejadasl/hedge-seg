import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def draw_rectangle_on_image(image_path, xmin, ymin, xmax, ymax):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)

    # Rectangle expects bottom-left corner in data coords (x, y) with y increasing downward in images
    rect = Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=2, edgecolor="red"
    )
    ax.add_patch(rect)

    ax.axis("off")
    return ax


def draw_polylines_on_image(image_path, json_path):
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if json_path.suffix == ".json":
        polylines = json.load(open(json_path))["polylines_px"]
    if json_path.suffix == ".npz":
        polylines = np.load(json_path)["polylines"]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    for i in polylines:
        i = np.array(i)
        ax.plot(i[:, 0], i[:, 1])
    plt.show(block=False)


def visualize_sample(
    main_path, num, folder="test_256_None", label_folder="labels", label_prefix="json"
):
    """
    label_folder: "labels" or "embs_polylines"
    label_prefix: "json" or "npz"
    """
    image_path = main_path / f"{folder}/images/pos_{num:06d}.png"
    json_path = main_path / f"{folder}/{label_folder}/pos_{num:06d}.{label_prefix}"
    draw_polylines_on_image(image_path, json_path)


"""
from pathlib import Path
main_path = Path(f"/home/fatemeh/Downloads/hedge/results")
visualize_sample(main_path, num=1, folder="pdok_dataset") # test_256_dino256
print("Done")

a = np.array([[106, 99], [104, 104], [102, 106], [119, 108], [128, 109]])
xmin, ymin, xmax, ymax = 102, 99, 128, 109
image_path = Path(
    "/home/fatemeh/Downloads/hedge/results/test_dataset_with_osm/images/pos_000000.png"
)
ax = draw_rectangle_on_image(image_path, xmin, ymin, xmax, ymax)
ax.plot(a[:, 0], a[:, 1], "*r")
plt.show(block=False)
"""
