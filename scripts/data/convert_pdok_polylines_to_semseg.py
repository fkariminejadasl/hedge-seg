"""
Convert PDOK hedge polyline JSON labels to semantic segmentation masks.

Input dataset layout:
    pdok_dataset/
        images/
            pos_000000.jpg
            neg_000000.jpg
            ...
        labels/
            pos_000000.json
            neg_000000.json
            ...

Each JSON file is expected to contain:
    {
        "chip_size_px": 1000,
        "pixel_size_m": 0.25,
        "polylines_px": [
            [[x0, y0], [x1, y1], ...],
            ...
        ]
    }

Output dataset layout:
    pdok_dataset_semseg/
        images/
            train/
            val/
        masks/
            train/
            val/
        centerlines/
            train/
            val/
"""

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm


@dataclass
class SemSegExportConfig:
    dataset_root: Path
    output_root: Path
    val_fraction: float = 0.2
    seed: int = 123

    image_size: tuple[int, int] = (1000, 1000)

    # This should match the label width you used for YOLO sanity checking.
    # Example:
    #   15 m at 0.25 m/px = 60 px total line width.
    mask_width_m: float = 15.0

    # Thin centerline target. This is useful because the final task is polyline extraction.
    centerline_width_m: float = 1.0

    copy_images: bool = True


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        p = image_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def _safe_polyline(points: Iterable) -> list[tuple[int, int]]:
    out = []
    for p in points:
        if len(p) < 2:
            continue
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        out.append((x, y))
    return out


def _draw_polylines(
    polylines_px: list,
    image_size: tuple[int, int],
    line_width_px: int,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for line in polylines_px:
        pts = _safe_polyline(line)
        if len(pts) == 1:
            x, y = pts[0]
            r = max(1, line_width_px // 2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        elif len(pts) >= 2:
            # PIL draws a rasterized thick polyline.
            # joint="curve" is available in recent Pillow versions.
            try:
                draw.line(pts, fill=255, width=line_width_px, joint="curve")
            except TypeError:
                draw.line(pts, fill=255, width=line_width_px)

    return mask


def _load_polylines_and_pixel_size(label_path: Path) -> tuple[list, float]:
    if not label_path.exists():
        return [], 0.25

    with open(label_path, "r") as f:
        data = json.load(f)

    polylines_px = data.get("polylines_px", [])
    pixel_size_m = float(data.get("pixel_size_m", 0.25))
    return polylines_px, pixel_size_m


def export_pdok_json_dataset_to_semseg(cfg: SemSegExportConfig) -> None:
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.output_root = Path(cfg.output_root)

    image_dir = cfg.dataset_root / "images"
    label_dir = cfg.dataset_root / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    for split in ["train", "val"]:
        (cfg.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (cfg.output_root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (cfg.output_root / "centerlines" / split).mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.json"))
    if len(label_files) == 0:
        raise RuntimeError(f"No JSON labels found in {label_dir}")

    random.seed(cfg.seed)
    random.shuffle(label_files)

    n_val = int(round(cfg.val_fraction * len(label_files)))
    val_stems = {p.stem for p in label_files[:n_val]}

    n_written = 0
    n_missing_images = 0

    for label_path in tqdm(label_files, desc="Exporting semantic masks"):
        stem = label_path.stem
        image_path = _find_image(image_dir, stem)

        if image_path is None:
            n_missing_images += 1
            continue

        split = "val" if stem in val_stems else "train"

        polylines_px, pixel_size_m = _load_polylines_and_pixel_size(label_path)

        mask_width_px = max(1, int(round(cfg.mask_width_m / pixel_size_m)))
        centerline_width_px = max(1, int(round(cfg.centerline_width_m / pixel_size_m)))

        mask = _draw_polylines(
            polylines_px=polylines_px,
            image_size=cfg.image_size,
            line_width_px=mask_width_px,
        )
        centerline = _draw_polylines(
            polylines_px=polylines_px,
            image_size=cfg.image_size,
            line_width_px=centerline_width_px,
        )

        out_image_path = cfg.output_root / "images" / split / image_path.name
        out_mask_path = cfg.output_root / "masks" / split / f"{stem}.png"
        out_centerline_path = cfg.output_root / "centerlines" / split / f"{stem}.png"

        if cfg.copy_images:
            shutil.copy2(image_path, out_image_path)
        else:
            if out_image_path.exists():
                out_image_path.unlink()
            out_image_path.hardlink_to(image_path)

        mask.save(out_mask_path)
        centerline.save(out_centerline_path)

        n_written += 1

    dataset_info = {
        "dataset_root": str(cfg.dataset_root),
        "output_root": str(cfg.output_root),
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "image_size": list(cfg.image_size),
        "mask_width_m": cfg.mask_width_m,
        "centerline_width_m": cfg.centerline_width_m,
        "n_written": n_written,
        "n_missing_images": n_missing_images,
    }

    with open(cfg.output_root / "dataset_semseg.yaml", "w") as f:
        OmegaConf.save(OmegaConf.create(dataset_info), f)

    print(f"Done. Wrote {n_written} samples to {cfg.output_root}")
    if n_missing_images:
        print(f"Warning: {n_missing_images} labels had no matching image.")


def main() -> None:
    cfg = SemSegExportConfig(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        output_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset_semseg2"),
        val_fraction=0.2,
        seed=123,
        image_size=(1000, 1000),
        # Use the same width you used for YOLO labels first.
        mask_width_m=15.0,
        # This is the centerline supervision target.
        centerline_width_m=1.0,
        copy_images=False,
    )

    export_pdok_json_dataset_to_semseg(cfg)


if __name__ == "__main__":
    main()
