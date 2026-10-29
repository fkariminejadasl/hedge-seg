"""
Convert PDOK hedge polyline JSON labels to Ultralytics YOLO detection bbox format.

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

    pdok_dataset_yolo_bbox/
      images/
        train/
        val/
      labels/
        train/
        val/
      dataset.yaml

YOLO detection label format:

    class_id x_center y_center width height

Coordinates are normalized to [0, 1].
"""

from pathlib import Path

from hedge_seg.ultralytics_export import (
    YoloBboxExportConfig,
    export_pdok_json_dataset_to_yolo_bbox,
)


def main() -> None:
    cfg = YoloBboxExportConfig(
        dataset_root=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset2"),
        output_root=Path(
            "/home/fatemeh/Downloads/hedge/results/pdok_dataset_yolo_bbox2"
        ),
        class_id=0,
        class_name="hedge",
        val_fraction=0.2,
        seed=123,
        # Expands each bbox by this many pixels on every side.
        bbox_expand_px=1,
        # Set this because images are fixed size.
        image_size=(1000, 1000),
        copy_images=True,
        max_workers=None,
    )

    export_pdok_json_dataset_to_yolo_bbox(cfg)


if __name__ == "__main__":
    main()
