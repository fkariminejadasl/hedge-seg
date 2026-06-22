import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def yolo_cxcywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    return (
        (cx - w / 2.0) * img_w,
        (cy - h / 2.0) * img_h,
        (cx + w / 2.0) * img_w,
        (cy + h / 2.0) * img_h,
    )


def draw_yolo_bounding_box_on_image(main_path, num, folder="val"):
    image_path = main_path / f"images/{folder}/pos_{num:06d}.png"
    label_path = main_path / f"labels/{folder}/pos_{num:06d}.txt"

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    labels = np.loadtxt(label_path, ndmin=2)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    for row in labels:
        class_id, cx, cy, w, h = row
        img_h, img_w = img_rgb.shape[:2]
        xmin, ymin, xmax, ymax = yolo_cxcywh_to_xyxy(cx, cy, w, h, img_w, img_h)

        # Rectangle expects bottom-left corner in data coords (x, y) with y increasing downward in images
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=2,
            edgecolor="red",
        )
        ax.add_patch(rect)

    ax.axis("off")
    plt.show(block=False)
    return ax


def read_yolo_seg_labels(label_path):
    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]], dtype=float)

            if len(coords) % 2 != 0:
                print(f"Skipping malformed row with odd number of coords: {label_path}")
                continue

            coords = coords.reshape(-1, 2)
            labels.append((class_id, coords))

    return labels


def draw_yolo_segmentation_on_image(main_path, num, folder="val"):
    stem = f"pos_{num:06d}"
    image_path = main_path / "images" / folder / f"{stem}.png"
    label_path = main_path / "labels" / folder / f"{stem}.txt"
    if not image_path.exists():
        image_path = main_path / f"{stem}.png"
    if not label_path.exists():
        label_path = main_path / f"{stem}.txt"

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    labels = read_yolo_seg_labels(label_path)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)

    img_h, img_w = img_rgb.shape[:2]

    for class_id, coords in labels:
        xs = coords[:, 0] * img_w
        ys = coords[:, 1] * img_h

        ax.fill(xs, ys, alpha=0.25, color="red")
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], linewidth=2, color="red")
        ax.plot(xs, ys, "*", color="red", markersize=4)

    ax.axis("off")
    plt.show(block=False)
    return ax


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


def show_image_with_mask(image, mask, alpha=0.25):
    """
    Plot a red transparent mask over an image.

    image: HxW or HxWx3 array
    mask:  HxW array, boolean or numeric
    """
    mask_bool = mask > 0

    overlay = np.zeros((*mask.shape, 4))  # RGBA
    overlay[mask_bool] = [1, 0, 0, alpha]  # red with transparency

    plt.figure()
    plt.imshow(image)
    plt.imshow(overlay)
    plt.axis("off")
    plt.show(block=False)


def _read_polylines(polyline_path):
    polyline_path = Path(polyline_path)

    if polyline_path.suffix == ".json":
        data = json.load(open(polyline_path))
        if "polylines_px_resampled" in data:
            return np.asarray(data["polylines_px_resampled"], dtype=float)
        return np.asarray(data["polylines_px"], dtype=float)

    data = np.load(polyline_path)
    if "polylines" in data:
        return data["polylines"]
    return data["polylines_px"]


def _sample_id_from_path(path):
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse sample id from {path}")
    return int(digits[-6:])


def _polyline_files(polyline_dir, polyline_pattern):
    if polyline_pattern is not None:
        suffix = Path(polyline_pattern.format(0)).suffix
        files = sorted(polyline_dir.glob(f"*{suffix}"))
        if files:
            return files

    return sorted([*polyline_dir.glob("*.npz"), *polyline_dir.glob("*.json")])


def _polyline_path_for_id(polyline_dir, sample_id, polyline_pattern):
    candidates = []
    if polyline_pattern is not None:
        candidates.append(polyline_dir / polyline_pattern.format(sample_id))
    candidates.extend(
        [
            polyline_dir / f"pos_{sample_id:06d}.npz",
            polyline_dir / f"pos_{sample_id:06d}.json",
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def show_polyline_grid(
    image_dir,
    polyline_dir,
    ids=None,
    n=16,
    image_pattern="pos_{:06d}.png",
    polyline_pattern=None,
    linewidth=1.5,
    seed=42,
    title="GT",
):
    image_dir = Path(image_dir)
    polyline_dir = Path(polyline_dir)

    if ids is None:
        files = _polyline_files(polyline_dir, polyline_pattern)
        if len(files) == 0:
            raise RuntimeError(f"No polylines found in {polyline_dir}")
        rng = np.random.default_rng(seed)
        files = rng.choice(files, size=min(n, len(files)), replace=False)
        ids = [_sample_id_from_path(p) for p in files]
    else:
        ids = ids[:n]

    fig, axes = plt.subplots(4, 4, figsize=(10.8, 10.8))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    for ax, i in zip(axes, ids):
        image_path = image_dir / image_pattern.format(i)
        polyline_path = _polyline_path_for_id(polyline_dir, i, polyline_pattern)

        image = cv2.imread(str(image_path))
        if image is None:
            ax.set_title(f"missing pos_{i:06d}", fontsize=8, pad=0)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        if polyline_path.exists():
            polylines = _read_polylines(polyline_path)
            for polyline in polylines:
                polyline = np.asarray(polyline)
                ax.plot(
                    polyline[:, 0],
                    polyline[:, 1],
                    linewidth=linewidth,
                )

        ax.set_title(f"pos_{i:06d}", fontsize=8, pad=0)

    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.01, hspace=0.05
    )
    plt.show(block=False)


def show_mask_grid(
    image_dir,
    mask_dir,
    ids=None,
    n=16,
    image_pattern="pos_{:06d}.png",
    mask_pattern="pos_{:06d}.png",
    alpha=0.25,
    color=(1, 0, 0),
    seed=42,
    title="GT",
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if ids is None:
        files = sorted(image_dir.glob("*.png"))
        rng = np.random.default_rng(seed)
        files = rng.choice(files, size=n, replace=False)
        ids = [int(p.stem.split("_")[-1]) for p in files]
    else:
        ids = ids[:n]

    fig, axes = plt.subplots(4, 4, figsize=(10.8, 10.8))

    for ax, i in zip(axes.ravel(), ids):
        image_path = image_dir / image_pattern.format(i)
        mask_path = mask_dir / mask_pattern.format(i)

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bool = mask > 0

        overlay = np.zeros((*mask.shape, 4))
        overlay[mask_bool] = (*color, alpha)

        ax.imshow(image)
        ax.imshow(overlay)
        ax.set_title(f"pos_{i:06d}", fontsize=8, pad=0)
        ax.axis("off")

    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.01, hspace=0.05
    )
    plt.show(block=False)


"""
from pathlib import Path
main_path = Path(f"/home/fatemeh/Downloads/hedge/results")
visualize_sample(main_path, num=1, folder="pdok_dataset") # test_256_dino256
for i in range(359,1000):
    folder= "pdok_dataset"
    visualize_sample(main_path, num=i, folder=folder)
    plt.savefig(main_path/f"{folder}/tmp/pos_{i:06d}.png")
    plt.close()
print("Done")

# visualize yolo bounding boxes and segmentations
from pathlib import Path
main_path = Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset_yolo_bbox2")
draw_yolo_bounding_box_on_image(main_path, 9, folder="train")
main_path = Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset_yolo_seg2")
draw_yolo_segmentation_on_image(main_path, 9, folder="train")
print("Done")

a = np.array([[106, 99], [104, 104], [102, 106], [119, 108], [128, 109]])
xmin, ymin, xmax, ymax = 102, 99, 128, 109
image_path = Path(
    "/home/fatemeh/Downloads/hedge/results/test_dataset_with_osm/images/pos_000000.png"
)
ax = draw_rectangle_on_image(image_path, xmin, ymin, xmax, ymax)
ax.plot(a[:, 0], a[:, 1], "*r")
plt.show(block=False)

# visualize semantic segmentation masks
image = cv2.imread("/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg2_1image/images/train/pos_000000.png")
mask = cv2.imread("/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg2_1image/inference/masks/pos_000000.png", cv2.IMREAD_GRAYSCALE)
show_image_with_mask(image, mask)
mask = cv2.imread("/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg2_1image/masks/train/pos_000000.png", cv2.IMREAD_GRAYSCALE)
show_image_with_mask(image, mask)

# visualize a grid of masks
# ids=[6712, 2591, 18168, 18742, 1050, 10454, 28379, 17717, 9641, 28365, 5019, 26894, 8816, 13029, 4124, 24082],
for seed in [42, 123, 456]:
    image_dir = "/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg3/images/val"
    mask_dir = "/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg3/masks/val"
    show_mask_grid(
        image_dir=image_dir,
        mask_dir=mask_dir,
        seed=seed,
    )

    image_dir = "/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg3/images/val"
    mask_dir = "/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg3/inference/masks"
    show_mask_grid(
        image_dir=image_dir,
        mask_dir=mask_dir,
        seed=seed,
        title="Inference",
    )

# visualize a grid of polylines
show_polyline_grid(
    image_dir="/home/fatemeh/Downloads/hedge/results/test_256_dino256/images",
    polyline_dir="/home/fatemeh/Downloads/hedge/results/test_256_dino256/inference/polylines",
    title="Inference",
)
show_polyline_grid(
    image_dir="/home/fatemeh/Downloads/hedge/results/test_256_dino256/images",
    polyline_dir="/home/fatemeh/Downloads/hedge/results/test_256_dino256/labels_processed",
    title="GT",
)
"""
