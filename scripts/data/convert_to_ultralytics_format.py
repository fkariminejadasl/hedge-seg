import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image

# =========================
# Config
# =========================


@dataclass(frozen=True)
class YoloExportConfig:
    dataset_root: Path  # root that currently contains images/ and labels/
    output_root: Path  # new YOLO dataset root
    class_id: int = 0  # single-class dataset by default
    class_name: str = "hedge"
    val_fraction: float = 0.2
    seed: int = 123
    bbox_expand_px: int = 1  # enlarge bbox by 1 pixel on each side
    copy_images: bool = True  # False -> hardlink/copy fallback can be added
    allowed_image_suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


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
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def find_image_for_label(
    images_dir: Path, stem: str, allowed_suffixes: Sequence[str]
) -> Optional[Path]:
    for suffix in allowed_suffixes:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def iter_paired_samples(
    dataset_root: Path, allowed_suffixes: Sequence[str]
) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        image_path = find_image_for_label(images_dir, label_path.stem, allowed_suffixes)
        if image_path is None:
            continue
        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No matching image/label pairs were found.")

    return pairs


# =========================
# Geometry helpers
# =========================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def clamp_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
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


def polyline_to_bbox_xyxy(
    polyline: Sequence[Sequence[float]],
    img_w: int,
    img_h: int,
    expand_px: int = 1,
) -> Optional[Tuple[float, float, float, float]]:
    if not polyline:
        return None

    xs = [float(p[0]) for p in polyline]
    ys = [float(p[1]) for p in polyline]

    xmin = min(xs) - expand_px
    ymin = min(ys) - expand_px
    xmax = max(xs) + expand_px
    ymax = max(ys) + expand_px

    xmin, ymin, xmax, ymax = clamp_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)

    # reject degenerate boxes after clamping
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


def format_yolo_row(class_id: int, xywh_norm: Tuple[float, float, float, float]) -> str:
    x_c, y_c, w, h = xywh_norm
    return f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


# =========================
# Label conversion
# =========================


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def polylines_from_label_data(label_data: dict) -> List[List[List[float]]]:
    return label_data.get("polylines_px", []) or []


def label_json_to_yolo_lines(
    label_data: dict,
    img_w: int,
    img_h: int,
    class_id: int,
    expand_px: int,
) -> List[str]:
    rows: List[str] = []
    polylines = polylines_from_label_data(label_data)

    for polyline in polylines:
        bbox = polyline_to_bbox_xyxy(
            polyline=polyline,
            img_w=img_w,
            img_h=img_h,
            expand_px=expand_px,
        )
        if bbox is None:
            continue

        yolo_xywh = xyxy_to_yolo_xywh(*bbox, img_w=img_w, img_h=img_h)
        rows.append(format_yolo_row(class_id, yolo_xywh))

    return rows


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
# IO helpers
# =========================


def copy_image(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)


def write_text_file(path: Path, lines: Iterable[str]) -> None:
    content = "\n".join(lines).strip()
    with path.open("w", encoding="utf-8") as f:
        if content:
            f.write(content)
            f.write("\n")


def write_dataset_yaml(output_root: Path, class_name: str) -> Path:
    yaml_path = output_root / "dataset.yaml"
    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


# =========================
# Main export
# =========================


def export_split(
    pairs: Sequence[Tuple[Path, Path]],
    image_out_dir: Path,
    label_out_dir: Path,
    cfg: YoloExportConfig,
) -> None:
    for image_path, label_path in pairs:
        img_w, img_h = read_image_size(image_path)
        label_data = load_json(label_path)

        yolo_rows = label_json_to_yolo_lines(
            label_data=label_data,
            img_w=img_w,
            img_h=img_h,
            class_id=cfg.class_id,
            expand_px=cfg.bbox_expand_px,
        )

        out_image_path = image_out_dir / image_path.name
        out_label_path = label_out_dir / f"{image_path.stem}.txt"

        if cfg.copy_images:
            copy_image(image_path, out_image_path)

        # Write label file even if empty, which is usually fine for YOLO.
        write_text_file(out_label_path, yolo_rows)


def export_pdok_json_dataset_to_yolo(cfg: YoloExportConfig) -> None:
    paths = ensure_yolo_dirs(cfg.output_root)
    pairs = iter_paired_samples(cfg.dataset_root, cfg.allowed_image_suffixes)
    train_pairs, val_pairs = split_train_val(pairs, cfg.val_fraction, cfg.seed)

    export_split(
        pairs=train_pairs,
        image_out_dir=paths["images_train"],
        label_out_dir=paths["labels_train"],
        cfg=cfg,
    )
    export_split(
        pairs=val_pairs,
        image_out_dir=paths["images_val"],
        label_out_dir=paths["labels_val"],
        cfg=cfg,
    )

    yaml_path = write_dataset_yaml(cfg.output_root, cfg.class_name)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print(f"YOLO dataset written to: {cfg.output_root.resolve()}")
    print(f"Dataset YAML: {yaml_path.resolve()}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    cfg = YoloExportConfig(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        output_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset_yolo2"),
        class_id=0,
        class_name="hedge",
        val_fraction=0.2,
        seed=123,
        bbox_expand_px=1,
    )
    export_pdok_json_dataset_to_yolo(cfg)
