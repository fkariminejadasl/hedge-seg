"""
Convert PDOK hedge polyline JSON labels to per-image polyline NPZ files for
DETR-style polyline training (scripts/train_detr_maptr_polyline.py).

Unlike build_lidar_training_dataset.py, this only converts ground truth: no
embeddings are computed, so the same NPZ files work for any image backbone.

Input dataset layout (from build_pdok_wms_dataset.py):

    pdok_dataset/
      images/
        pos_000001.png
        ...
      labels/
        pos_000001.json   # contains "polylines_px": [[[x,y], ...], ...]
        ...

Output layout:

    out_root/
      polylines/
        pos_000001.npz
        ...
      overlays/           # optional GT visualization for verification
        pos_000001.png
        ...

NPZ content per image (same target format as the embs_polylines NPZs, minus feat):

    polylines: (N, num_points, 2) float32, xy pixel coords, equidistant resampled
    labels: (N,) int64, all class_id
    image_size: (2,) int32, (H, W)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hedge_seg.label_postprocess import resample_polyline_equidistant


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polyline_length_px(points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def convert_label_file(
    json_path: Path,
    image_path: Path,
    num_points: int,
    class_id: int,
    min_length_px: float,
):
    data = load_json(json_path)
    raw_polylines = data.get("polylines_px", []) or []

    with Image.open(image_path) as im:
        W, H = im.size

    kept = []
    n_skipped = 0
    for line in raw_polylines:
        if polyline_length_px(line) < min_length_px:
            n_skipped += 1
            continue
        resampled = resample_polyline_equidistant(line, num_points)
        kept.append(resampled)

    if kept:
        polylines = np.asarray(kept, dtype=np.float32)  # (N,K,2)
        polylines[..., 0] = polylines[..., 0].clip(0, W - 1)
        polylines[..., 1] = polylines[..., 1].clip(0, H - 1)
    else:
        polylines = np.zeros((0, num_points, 2), dtype=np.float32)

    labels = np.full((polylines.shape[0],), class_id, dtype=np.int64)
    image_size = np.asarray([H, W], dtype=np.int32)
    return polylines, labels, image_size, n_skipped


def save_overlay(
    image_path: Path, polylines: np.ndarray, out_path: Path, dpi: int = 150
):
    """
    Draw GT polylines on the image. Point order is shown as a color gradient
    (dark = first point, bright = last point) with a circle at the first point,
    so point ordering and endpoints are visually verifiable.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]

    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(image)
    cmap = plt.get_cmap("tab10")
    for i, poly in enumerate(polylines):
        color = cmap(i % 10)
        ax.plot(poly[:, 0], poly[:, 1], "-", color=color, linewidth=1.0)
        order = np.linspace(0.3, 1.0, poly.shape[0])
        ax.scatter(
            poly[:, 0], poly[:, 1], c=order, cmap="viridis", s=4, zorder=3
        )
        ax.plot(poly[0, 0], poly[0, 1], "o", color=color, markersize=5, zorder=4)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def convert_dataset(
    dataset_root: Path,
    out_root: Path,
    num_points: int = 20,
    class_id: int = 0,
    min_length_px: float = 2.0,
    save_overlays: bool = True,
):
    image_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    polyline_dir = out_root / "polylines"
    polyline_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_root / "overlays"
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("pos_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No pos_*.json files in {labels_dir}")

    n_lines_total = 0
    n_skipped_total = 0
    max_lines = 0
    for json_path in json_files:
        stem = json_path.stem
        image_path = None
        for suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f"No image for {stem} in {image_dir}")

        polylines, labels, image_size, n_skipped = convert_label_file(
            json_path, image_path, num_points, class_id, min_length_px
        )
        np.savez_compressed(
            polyline_dir / f"{stem}.npz",
            polylines=polylines,
            labels=labels,
            image_size=image_size,
        )

        n_lines_total += polylines.shape[0]
        n_skipped_total += n_skipped
        max_lines = max(max_lines, polylines.shape[0])

        if save_overlays:
            save_overlay(image_path, polylines, overlay_dir / f"{stem}.png")

    print(
        f"Converted {len(json_files)} images to {polyline_dir}: "
        f"{n_lines_total} polylines kept, {n_skipped_total} skipped "
        f"(< {min_length_px}px), max lines per image: {max_lines}"
    )
    if save_overlays:
        print(f"GT overlays written to {overlay_dir}")


def main() -> None:
    convert_dataset(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        out_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2_detr"),
        num_points=20,
        class_id=0,
        # Drop degenerate polylines shorter than this (in pixels).
        min_length_px=2.0,
        save_overlays=True,
    )


if __name__ == "__main__":
    main()
