"""
Convert PDOK hedge polyline JSON labels to per-image polyline NPZ files for
DETR-style polyline training (scripts/train_detr_unet_polyline.py).

Unlike build_lidar_training_dataset.py, this only converts ground truth: no
embeddings are computed, so the same NPZ files work for any image backbone.

Train/val split is geographic, not random. Crops are sampled per polyline, so
crops from the same location overlap (in pdok_dataset3, 72.6% of crops overlap
at least one other crop), and a random split would leak: most val crops would
overlap a training crop (see quantify_geographic_crop_overlap_pdok_dataset3.py).
Here crops are grouped into blocks of block_m meters using center_world from the
label JSON, blocks are hashed into train/val, and any connected group of overlapping
crops that touches a train block goes entirely to train. A KDTree check asserts that
no val crop overlaps a train crop. Optionally, val also avoids areas used by another
dataset (for example the semseg backbone training data), via avoid_label_dirs.

Label cleaning:
- closed rings (first and last point within closed_eps_px) are opened by
  removing the last point, so the forward/reverse ordered L1 loss applies
- polylines shorter than min_length_px (after opening) are dropped

Input dataset layout (from build_pdok_wms_dataset.py):

    pdok_dataset/
      images/
        pos_000001.png
        ...
      labels/
        pos_000001.json   # "polylines_px", "center_world", "chip_size_m"
        ...

Output layout:

    out_root/
      polylines/
        train/
          pos_000001.npz
          ...
        val/
          ...
      overlays/           # GT visualization for a random sample of images
        train_pos_000001.png / val_pos_000004.png
        ...

NPZ content per image (same target format as the embs_polylines NPZs, minus feat):

    polylines: (N, num_points, 2) float32, xy pixel coords, equidistant resampled
    labels: (N,) int64, all class_id
    image_size: (2,) int32, (H, W)
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from hedge_seg.label_postprocess import resample_polyline_equidistant


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polyline_length_px(points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def open_closed_ring(points: list, closed_eps_px: float) -> Tuple[list, bool]:
    """
    If first and last point are within closed_eps_px, drop the last point so
    the polyline becomes an open chain (the fwd/rev ordered L1 loss assumes
    open chains; a ring would punish a correct shape drawn with another cut).
    """
    if len(points) >= 3:
        gap = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        if gap <= closed_eps_px:
            return points[:-1], True
    return points, False


def convert_label_file(
    json_path: Path,
    image_path: Path,
    num_points: int,
    class_id: int,
    min_length_px: float,
    closed_eps_px: float,
):
    data = load_json(json_path)
    raw_polylines = data.get("polylines_px", []) or []

    with Image.open(image_path) as im:
        W, H = im.size

    kept = []
    n_skipped = 0
    n_opened = 0
    for line in raw_polylines:
        line, opened = open_closed_ring(line, closed_eps_px)
        n_opened += int(opened)
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
    return polylines, labels, image_size, n_skipped, n_opened


# =========================
# Geographic train/val split
# =========================


def _tile_is_val(ix: int, iy: int, seed: int, val_fraction: float) -> bool:
    h = hashlib.md5(f"{seed}_{ix}_{iy}".encode()).hexdigest()
    return (int(h, 16) % 10**8) / 10**8 < val_fraction


def _load_centers(labels_dir: Path) -> Tuple[List[Path], np.ndarray, float]:
    json_files = sorted(Path(labels_dir).glob("pos_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No pos_*.json files in {labels_dir}")
    centers = []
    chip_m = 0.0
    for f in json_files:
        data = load_json(f)
        centers.append(data["center_world"])
        chip_m = max(chip_m, float(data["chip_size_m"]))
    return json_files, np.asarray(centers, dtype=np.float64), chip_m


def spatial_train_val_split(
    labels_dir: Path,
    block_m: float = 5000.0,
    val_fraction: float = 0.2,
    seed: int = 42,
    avoid_label_dirs: Optional[List[Path]] = None,
) -> Dict[str, str]:
    """
    Returns {stem: "train" or "val"}. Two crops of size chip_m overlap iff the
    Chebyshev distance of their centers is below chip_m; connected components
    of the overlap graph are assigned as a whole, so no val crop can overlap a
    train crop.
    Chebyshev distance: max(abs(x1 - x2), abs(y1 - y2)). np.inf indicates it.
    """
    json_files, centers, chip_m = _load_centers(labels_dir)
    n = len(json_files)

    tiles = np.floor(centers / block_m).astype(np.int64)
    tile_val = np.array([_tile_is_val(ix, iy, seed, val_fraction) for ix, iy in tiles])

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=chip_m, p=np.inf, output_type="ndarray")
    if len(pairs):
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        _, comp = connected_components(graph, directed=False)
    else:
        comp = np.arange(n)

    near_avoid = np.zeros(n, dtype=bool)
    if avoid_label_dirs:
        for avoid_dir in avoid_label_dirs:
            _, avoid_centers, avoid_chip = _load_centers(avoid_dir)
            avoid_tree = cKDTree(avoid_centers)
            dist, _ = avoid_tree.query(centers, p=np.inf)
            near_avoid |= dist < max(chip_m, avoid_chip)

    # A component is val only if every member sits in a val block and none of
    # its members overlaps an avoided area; otherwise the whole component is
    # train. This guarantees zero val/train overlap by construction.
    is_val = np.zeros(n, dtype=bool)
    n_forced_train = 0
    for c in np.unique(comp):
        members = np.where(comp == c)[0]
        if tile_val[members].all() and not near_avoid[members].any():
            is_val[members] = True
        elif tile_val[members].any():
            n_forced_train += 1

    # Verify: no val crop overlaps a train crop.
    for i, j in pairs:
        assert is_val[i] == is_val[j], "split leak: val crop overlaps train crop"

    print(
        f"Spatial split ({block_m:.0f}m blocks, chip {chip_m:.0f}m): "
        f"train={(~is_val).sum()}, val={is_val.sum()} "
        f"({is_val.mean() * 100:.1f}% val, target {val_fraction * 100:.0f}%), "
        f"{n_forced_train} mixed components forced to train, "
        f"{int(near_avoid.sum())} crops near avoided areas"
    )
    return {f.stem: ("val" if v else "train") for f, v in zip(json_files, is_val)}


# =========================
# Overlay visualization
# =========================


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
        ax.scatter(poly[:, 0], poly[:, 1], c=order, cmap="viridis", s=4, zorder=3)
        ax.plot(poly[0, 0], poly[0, 1], "o", color=color, markersize=5, zorder=4)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =========================
# Dataset conversion
# =========================


def convert_dataset(
    dataset_root: Path,
    out_root: Path,
    num_points: int = 20,
    class_id: int = 0,
    min_length_px: float = 40.0,
    closed_eps_px: float = 2.0,
    block_m: float = 5000.0,
    val_fraction: float = 0.2,
    seed: int = 42,
    n_overlays: int = 20,
    avoid_label_dirs: Optional[List[Path]] = None,
):
    image_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    split = spatial_train_val_split(
        labels_dir,
        block_m=block_m,
        val_fraction=val_fraction,
        seed=seed,
        avoid_label_dirs=avoid_label_dirs,
    )

    for name in ["train", "val"]:
        (out_root / "polylines" / name).mkdir(parents=True, exist_ok=True)
    overlay_dir = out_root / "overlays"
    if n_overlays > 0:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("pos_*.json"))
    rng = np.random.default_rng(seed)
    overlay_stems = set(
        rng.choice(
            [f.stem for f in json_files],
            size=min(n_overlays, len(json_files)),
            replace=False,
        )
    )

    n_lines_total = 0
    n_skipped_total = 0
    n_opened_total = 0
    n_empty = 0
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

        polylines, labels, image_size, n_skipped, n_opened = convert_label_file(
            json_path, image_path, num_points, class_id, min_length_px, closed_eps_px
        )
        np.savez_compressed(
            out_root / "polylines" / split[stem] / f"{stem}.npz",
            polylines=polylines,
            labels=labels,
            image_size=image_size,
        )

        n_lines_total += polylines.shape[0]
        n_skipped_total += n_skipped
        n_opened_total += n_opened
        n_empty += int(polylines.shape[0] == 0)
        max_lines = max(max_lines, polylines.shape[0])

        if stem in overlay_stems:
            save_overlay(
                image_path, polylines, overlay_dir / f"{split[stem]}_{stem}.png"
            )

    print(
        f"Converted {len(json_files)} images to {out_root / 'polylines'}: "
        f"{n_lines_total} polylines kept, {n_skipped_total} skipped "
        f"(< {min_length_px}px), {n_opened_total} rings opened, "
        f"{n_empty} images left without polylines, "
        f"max lines per image: {max_lines}"
    )
    if n_overlays > 0:
        print(f"GT overlays for {len(overlay_stems)} images in {overlay_dir}")


def main() -> None:
    convert_dataset(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        out_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2_polylines"),
        num_points=20,
        class_id=0,
        # Drop polylines shorter than this (in pixels): 40 px = 10 m at 25 cm/px.
        min_length_px=40.0,
        # Open rings whose start and end are within this distance (in pixels).
        closed_eps_px=2.0,
        # Geographic split: block size, val fraction, seed.
        block_m=5000.0,
        val_fraction=0.2,
        seed=42,
        n_overlays=20,
        # On the cluster, set this to [Path(".../pdok_dataset2/labels")] so val
        # avoids the areas used to train the semseg backbone.
        avoid_label_dirs=None,
    )


if __name__ == "__main__":
    main()
