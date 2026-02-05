from pathlib import Path

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


"""
a = np.array([[106, 99], [104, 104], [102, 106], [119, 108], [128, 109]])
xmin, ymin, xmax, ymax = 102, 99, 128, 109
image_path = Path(
    "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/images/pos_000000.png"
)
ax = draw_rectangle_on_image(image_path, xmin, ymin, xmax, ymax)
ax.plot(a[:, 0], a[:, 1], "*r")
plt.show(block=False)
"""
