"""
Convert PDOK hedge polyline JSON labels to Ultralytics YOLO segmentation format.

This script is a thin entry point. Shared conversion logic lives in:

    hedge_seg.ultralytics_export

Input dataset layout:

    pdok_dataset/
      images/
        pos_000001.jpg
        ...
      labels/
        pos_000001.json
        ...

Output dataset layout:

    pdok_dataset_yolo_seg/
      images/
        train/
        val/
      labels/
        train/
        val/
      dataset.yaml

YOLO segmentation label format:

    class_id x1 y1 x2 y2 x3 y3 ... xn yn

Coordinates are normalized to [0, 1].
"""

from pathlib import Path

from hedge_seg.ultralytics_export import (
    YoloSegExportConfig,
    export_pdok_json_dataset_to_yolo_seg,
)


def main() -> None:
    cfg = YoloSegExportConfig(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        output_root=Path(
            "/home/fatemeh/Downloads/hedge/results/pdok_dataset_yolo_seg2"
        ),
        class_id=0,
        class_name="hedge",
        val_fraction=0.2,
        seed=123,
        # Images are 1000 x 1000 pixels and 250 m x 250 m.
        # Therefore pixel_size_m is 0.25 m/px.
        #
        # mask_width_m=3.0 gives:
        #   buffer radius = 3.0 / 0.25 / 2 = 6 px
        #   total mask width = 12 px
        mask_width_m=15.0,  # 3.0,
        # Increase this if label files become too large.
        # Decrease this if polygons become too coarse.
        simplify_tolerance_px=1.0,
        # For hedgerow centerlines:
        #   cap_style=2 gives flat ends.
        #   join_style=2 gives sharp joins.
        #
        # Alternative:
        #   cap_style=1, join_style=1 for rounder masks.
        cap_style=2,
        join_style=2,
        # Set this because images are fixed size.
        image_size=(1000, 1000),
        copy_images=True,
        max_workers=None,
    )

    export_pdok_json_dataset_to_yolo_seg(cfg)


if __name__ == "__main__":
    main()
