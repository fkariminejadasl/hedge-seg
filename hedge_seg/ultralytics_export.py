"""
Shared Ultralytics export utilities for hedge polyline datasets.

This module supports exporting generated JSON/polyline labels to:

1. YOLO detection bbox format
2. YOLO segmentation polygon format

Expected input dataset layout:

    dataset_root/
      images/
        pos_000001.jpg
        ...
      labels/
        pos_000001.json
        ...

Output YOLO layout:

    output_root/
      images/
        train/
        val/
      labels/
        train/
        val/
      dataset.yaml
"""

import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from shapely.geometry import LineString, MultiPolygon, Polygon, box

# =========================
# Config
# =========================


@dataclass(frozen=True)
class BaseYoloExportConfig:
    dataset_root: Path
    output_root: Path

    class_id: int = 0
    class_name: str = "hedge"

    val_fraction: float = 0.2
    seed: int = 123

    copy_images: bool = True

    allowed_image_suffixes: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    )

    # If all chips are fixed size, set this to avoid opening images.
    # Example: image_size=(1000, 1000)
    image_size: Optional[Tuple[int, int]] = None

    max_workers: Optional[int] = None


@dataclass(frozen=True)
class YoloBboxExportConfig(BaseYoloExportConfig):
    bbox_expand_px: int = 1


@dataclass(frozen=True)
class YoloSegExportConfig(BaseYoloExportConfig):
    # Total physical hedge width used to convert centerlines into mask polygons.
    # Example:
    #   pixel_size_m = 0.25
    #   mask_width_m = 3.0
    #   buffer radius = 3.0 / 0.25 / 2 = 6 px
    #   total mask width = 12 px
    mask_width_m: float = 3.0

    # Simplifies buffered polygons to avoid very large YOLO label files.
    simplify_tolerance_px: float = 1.0

    # Shapely buffer styles:
    #   cap_style: 1 round, 2 flat, 3 square
    #   join_style: 1 round, 2 mitre, 3 bevel
    cap_style: int = 2
    join_style: int = 2


# =========================
# Path helpers
# =========================


def ensure_yolo_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "images_train": output_root / "images" / "train",
        "images_val": output_root / "images" / "val",
        "labels_train": output_root / "labels" / "train",
        "labels_val": output_root / "labels" / "val",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def iter_paired_samples(
    dataset_root: Path,
    allowed_suffixes: Sequence[str],
) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    allowed = {suffix.lower() for suffix in allowed_suffixes}

    image_by_stem = {
        image_path.stem: image_path
        for image_path in images_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in allowed
    }

    pairs: List[Tuple[Path, Path]] = []

    for label_path in labels_dir.glob("*.json"):
        image_path = image_by_stem.get(label_path.stem)
        if image_path is not None:
            pairs.append((image_path, label_path))

    pairs = sorted(pairs, key=lambda pair: pair[0].stem)

    if not pairs:
        raise RuntimeError(
            f"No matching image and JSON label pairs were found in {dataset_root}"
        )

    return pairs


# =========================
# IO helpers
# =========================


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def copy_image(src: Path, dst: Path) -> None:
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_text_file(path: Path, lines: Iterable[str]) -> None:
    content = "\n".join(lines).strip()

    with path.open("w", encoding="utf-8") as file:
        if content:
            file.write(content)
            file.write("\n")


def write_dataset_yaml(
    output_root: Path,
    class_id: int,
    class_name: str,
) -> Path:
    yaml_path = output_root / "dataset.yaml"

    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  {class_id}: {class_name}\n"
    )

    yaml_path.write_text(yaml_text, encoding="utf-8")

    return yaml_path


# =========================
# Split helpers
# =========================


def split_train_val(
    pairs: Sequence[Tuple[Path, Path]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    pairs = list(pairs)

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_val = max(1, int(round(n_total * val_fraction))) if n_total > 1 else 0
    n_val = min(n_val, n_total)

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    return train_pairs, val_pairs


# =========================
# Generic geometry helpers
# =========================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def clamp_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    xmin = clamp(xmin, 0, img_w - 1)
    ymin = clamp(ymin, 0, img_h - 1)
    xmax = clamp(xmax, 0, img_w - 1)
    ymax = clamp(ymax, 0, img_h - 1)

    if xmax < xmin:
        xmax = xmin

    if ymax < ymin:
        ymax = ymin

    return xmin, ymin, xmax, ymax


def polylines_from_label_data(label_data: dict) -> List[List[List[float]]]:
    return label_data.get("polylines_px", []) or []


# =========================
# YOLO bbox conversion
# =========================


def polyline_to_bbox_xyxy(
    polyline: Sequence[Sequence[float]],
    img_w: int,
    img_h: int,
    expand_px: int = 1,
) -> Optional[Tuple[float, float, float, float]]:
    if not polyline:
        return None

    xs = [float(point[0]) for point in polyline]
    ys = [float(point[1]) for point in polyline]

    xmin = min(xs) - expand_px
    ymin = min(ys) - expand_px
    xmax = max(xs) + expand_px
    ymax = max(ys) + expand_px

    xmin, ymin, xmax, ymax = clamp_xyxy(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        img_w=img_w,
        img_h=img_h,
    )

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


def xyxy_to_yolo_xywh(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    box_w = xmax - xmin
    box_h = ymax - ymin

    center_x = xmin + box_w / 2.0
    center_y = ymin + box_h / 2.0

    return (
        center_x / img_w,
        center_y / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def format_yolo_bbox_row(
    class_id: int,
    xywh_norm: Tuple[float, float, float, float],
) -> str:
    x_center, y_center, width, height = xywh_norm
    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


def label_json_to_yolo_bbox_lines(
    label_data: dict,
    img_w: int,
    img_h: int,
    cfg: YoloBboxExportConfig,
) -> List[str]:
    rows: List[str] = []

    polylines = polylines_from_label_data(label_data)

    for polyline in polylines:
        bbox = polyline_to_bbox_xyxy(
            polyline=polyline,
            img_w=img_w,
            img_h=img_h,
            expand_px=cfg.bbox_expand_px,
        )

        if bbox is None:
            continue

        yolo_xywh = xyxy_to_yolo_xywh(
            *bbox,
            img_w=img_w,
            img_h=img_h,
        )

        rows.append(
            format_yolo_bbox_row(
                class_id=cfg.class_id,
                xywh_norm=yolo_xywh,
            )
        )

    return rows


# =========================
# YOLO segmentation conversion
# =========================


def normalize_polygon_coords(
    coords: Sequence[Tuple[float, float]],
    img_w: int,
    img_h: int,
) -> List[float]:
    values: List[float] = []

    for x, y in coords:
        x = clamp(float(x), 0.0, float(img_w - 1))
        y = clamp(float(y), 0.0, float(img_h - 1))

        values.append(x / img_w)
        values.append(y / img_h)

    return values


def keep_largest_polygon(geom) -> Optional[Polygon]:
    if geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            return None
        return max(geom.geoms, key=lambda polygon: polygon.area)

    return None


def polygon_to_exterior_coords(
    polygon: Polygon,
    simplify_tolerance_px: float,
) -> Optional[List[Tuple[float, float]]]:
    if polygon.is_empty or polygon.area <= 0:
        return None

    if simplify_tolerance_px > 0:
        polygon = polygon.simplify(
            simplify_tolerance_px,
            preserve_topology=True,
        )

    polygon = keep_largest_polygon(polygon)

    if polygon is None or polygon.is_empty or polygon.area <= 0:
        return None

    coords = list(polygon.exterior.coords)

    # YOLO segmentation labels do not need the duplicated closing point.
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        return None

    return [(float(x), float(y)) for x, y in coords]


def polyline_to_yolo_seg_row(
    polyline: Sequence[Sequence[float]],
    img_w: int,
    img_h: int,
    class_id: int,
    mask_width_m: float,
    pixel_size_m: float,
    simplify_tolerance_px: float,
    cap_style: int,
    join_style: int,
) -> Optional[str]:
    if len(polyline) < 2:
        return None

    points = [(float(x), float(y)) for x, y in polyline]

    line = LineString(points)

    if line.is_empty or line.length <= 0:
        return None

    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")

    if mask_width_m <= 0:
        raise ValueError(f"mask_width_m must be positive, got {mask_width_m}")

    buffer_radius_px = mask_width_m / pixel_size_m / 2.0

    polygon = line.buffer(
        buffer_radius_px,
        cap_style=cap_style,
        join_style=join_style,
    )

    image_bounds = box(0, 0, img_w - 1, img_h - 1)
    polygon = polygon.intersection(image_bounds)

    polygon = keep_largest_polygon(polygon)

    if polygon is None:
        return None

    coords = polygon_to_exterior_coords(
        polygon=polygon,
        simplify_tolerance_px=simplify_tolerance_px,
    )

    if coords is None:
        return None

    normalized = normalize_polygon_coords(
        coords=coords,
        img_w=img_w,
        img_h=img_h,
    )

    if len(normalized) < 6:
        return None

    return f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def label_json_to_yolo_seg_lines(
    label_data: dict,
    img_w: int,
    img_h: int,
    cfg: YoloSegExportConfig,
) -> List[str]:
    rows: List[str] = []

    pixel_size_m = float(label_data.get("pixel_size_m", 0.25))
    polylines = polylines_from_label_data(label_data)

    for polyline in polylines:
        row = polyline_to_yolo_seg_row(
            polyline=polyline,
            img_w=img_w,
            img_h=img_h,
            class_id=cfg.class_id,
            mask_width_m=cfg.mask_width_m,
            pixel_size_m=pixel_size_m,
            simplify_tolerance_px=cfg.simplify_tolerance_px,
            cap_style=cfg.cap_style,
            join_style=cfg.join_style,
        )

        if row is not None:
            rows.append(row)

    return rows


# =========================
# Export internals
# =========================


def _label_json_to_lines(
    label_data: dict,
    img_w: int,
    img_h: int,
    cfg: BaseYoloExportConfig,
    task: str,
) -> List[str]:
    if task == "bbox":
        if not isinstance(cfg, YoloBboxExportConfig):
            raise TypeError("bbox task requires YoloBboxExportConfig")

        return label_json_to_yolo_bbox_lines(
            label_data=label_data,
            img_w=img_w,
            img_h=img_h,
            cfg=cfg,
        )

    if task == "seg":
        if not isinstance(cfg, YoloSegExportConfig):
            raise TypeError("seg task requires YoloSegExportConfig")

        return label_json_to_yolo_seg_lines(
            label_data=label_data,
            img_w=img_w,
            img_h=img_h,
            cfg=cfg,
        )

    raise ValueError(f"Unknown task: {task}")


def export_one_sample(
    sample: Tuple[Path, Path],
    image_out_dir: Path,
    label_out_dir: Path,
    cfg: BaseYoloExportConfig,
    image_size: Tuple[int, int],
    task: str,
) -> int:
    image_path, label_path = sample
    img_w, img_h = image_size

    label_data = load_json(label_path)

    yolo_rows = _label_json_to_lines(
        label_data=label_data,
        img_w=img_w,
        img_h=img_h,
        cfg=cfg,
        task=task,
    )

    out_image_path = image_out_dir / image_path.name
    out_label_path = label_out_dir / f"{image_path.stem}.txt"

    if cfg.copy_images:
        copy_image(image_path, out_image_path)

    # Empty files are allowed and useful if an image has no valid objects.
    write_text_file(out_label_path, yolo_rows)

    return len(yolo_rows)


def export_split(
    pairs: Sequence[Tuple[Path, Path]],
    image_out_dir: Path,
    label_out_dir: Path,
    cfg: BaseYoloExportConfig,
    image_size: Tuple[int, int],
    task: str,
) -> int:
    if not pairs:
        return 0

    worker = partial(
        export_one_sample,
        image_out_dir=image_out_dir,
        label_out_dir=label_out_dir,
        cfg=cfg,
        image_size=image_size,
        task=task,
    )

    max_workers = cfg.max_workers or max(1, (os.cpu_count() or 2) - 1)

    if max_workers == 1:
        counts = [worker(pair) for pair in pairs]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            counts = list(executor.map(worker, pairs, chunksize=100))

    return sum(counts)


def export_pdok_json_dataset_to_yolo(
    cfg: BaseYoloExportConfig,
    task: str,
) -> None:
    paths = ensure_yolo_dirs(cfg.output_root)

    pairs = iter_paired_samples(
        dataset_root=cfg.dataset_root,
        allowed_suffixes=cfg.allowed_image_suffixes,
    )

    image_size = cfg.image_size or read_image_size(pairs[0][0])

    train_pairs, val_pairs = split_train_val(
        pairs=pairs,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    n_train_labels = export_split(
        pairs=train_pairs,
        image_out_dir=paths["images_train"],
        label_out_dir=paths["labels_train"],
        cfg=cfg,
        image_size=image_size,
        task=task,
    )

    n_val_labels = export_split(
        pairs=val_pairs,
        image_out_dir=paths["images_val"],
        label_out_dir=paths["labels_val"],
        cfg=cfg,
        image_size=image_size,
        task=task,
    )

    yaml_path = write_dataset_yaml(
        output_root=cfg.output_root,
        class_id=cfg.class_id,
        class_name=cfg.class_name,
    )

    print(f"Task: {task}")
    print(f"Total image/label pairs: {len(pairs)}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Train labels written: {n_train_labels}")
    print(f"Val labels written: {n_val_labels}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"YOLO dataset written to: {cfg.output_root.resolve()}")
    print(f"Dataset YAML: {yaml_path.resolve()}")


def export_pdok_json_dataset_to_yolo_bbox(
    cfg: YoloBboxExportConfig,
) -> None:
    export_pdok_json_dataset_to_yolo(
        cfg=cfg,
        task="bbox",
    )


def export_pdok_json_dataset_to_yolo_seg(
    cfg: YoloSegExportConfig,
) -> None:
    export_pdok_json_dataset_to_yolo(
        cfg=cfg,
        task="seg",
    )
