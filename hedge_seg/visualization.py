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
    polylines = json.load(open(json_path))["polylines_px"]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    for i in polylines:
        i = np.array(i)
        ax.plot(i[:, 0], i[:, 1])
    plt.show(block=False)


"""
num = 9520 # 1400
image_path = Path(f"/home/fatemeh/Downloads/hedge/results/test_mini3/images/pos_{num:06d}.png")
json_path = Path(f"/home/fatemeh/Downloads/hedge/results/test_mini3/labels/pos_{num:06d}.json")
draw_polylines_on_image(image_path, json_path)
print("Done")
"""

"""
from pathlib import Path
import numpy as np
a = np.array([[106, 99], [104, 104], [102, 106], [119, 108], [128, 109]])
xmin, ymin, xmax, ymax = 102, 99, 128, 109
image_path = Path(
    "/home/fatemeh/Downloads/hedge/results/test_dataset_with_osm/images/pos_000000.png"
)
ax = draw_rectangle_on_image(image_path, xmin, ymin, xmax, ymax)
ax.plot(a[:, 0], a[:, 1], "*r")
plt.show(block=False)
"""
