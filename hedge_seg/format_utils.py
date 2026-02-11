import json
from pathlib import Path

import numpy as np


def pack_pos_npz(
    embeddings_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    feat_key="feat",
    label_key="bboxes_px_per_polyline",
):
    """
    For each embeddings_dir/pos_*.npz:
      - loads feat from npz
      - loads labels_dir/<same_stem>.json
      - reads list of xyxy pixel boxes from label_key
      - writes output_dir/<same_name>.npz with: feat, boxes_xyxy, image_size
    Skips files with missing labels or zero valid boxes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0

    for emb_path in sorted(embeddings_dir.glob("pos_*.npz")):
        lab_path = labels_dir / f"{emb_path.stem}.json"
        if not lab_path.exists():
            skipped += 1
            continue

        with open(lab_path, "r") as f:
            lab = json.load(f)
        H = W = lab["chip_size_px"]  # image size

        boxes = []
        for b in lab.get(label_key, []):
            if not (isinstance(b, list) and len(b) == 4):
                continue
            x0, y0, x1, y1 = map(float, b)

            # fix order + clamp
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
            y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))

            if x1 == x0:
                x1 = min(W, x0 + 1.0)
            if y1 == y0:
                y1 = min(H, y0 + 1.0)

            if (x1 - x0) > 0 and (y1 - y0) > 0:
                boxes.append([x0, y0, x1, y1])

        if len(boxes) == 0:
            skipped += 1
            continue
        boxes = np.asarray(boxes, dtype=np.float32)  # (N,4) xyxy pixels
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)  # single class -> 0

        feat = np.load(emb_path)[feat_key]
        feat = np.ascontiguousarray(feat).astype(np.float32)

        out_path = output_dir / emb_path.name
        np.savez_compressed(
            out_path,
            feat=feat,
            boxes=boxes,
            labels=labels,
            image_size=np.asarray([H, W], dtype=np.int32),
        )
        written += 1

    print(f"Written: {written}, Skipped: {skipped}, Output: {output_dir}")


main_dir = Path("/home/fatemeh/Downloads/hedg/results/test_mini")
pack_pos_npz(
    embeddings_dir=main_dir / "embeddings",
    labels_dir=main_dir / "labels_processed",
    output_dir=main_dir / "packed_pos_npz",
)
