"""
Embedding extraction and final dataset packing.

Embedding extraction
  - Compute embeddings for each training image
  - Save embeddings to disk (per-image or sharded) so this step can run on a cluster

Packing
  - Combine embeddings with (post-processed) labels
"""

import json
from pathlib import Path

import numpy as np
from transformers import pipeline
from transformers.image_utils import load_image


def save_DINOv3_embeddings_sequential(image_dir, embed_dir):
    feature_extractor = pipeline(
        model="facebook/dinov3-vitl16-pretrain-lvd1689m",
        task="image-feature-extraction",
    )

    def save_DINOv3_embeddings_per_image(image_path, embed_dir):
        image = load_image(str(image_path))
        features = feature_extractor(image)
        features = np.asarray(features, dtype=np.float32)  # (1, 201, 1024)
        features = features.squeeze(0)  # (201, 1024)
        #  1 class token + 4 register tokens + 196 patch tokens

        save_file = embed_dir / f"{image_path.stem}.npz"
        np.savez(save_file, **{"feat": features[5:]})  # *.npz

    # Save embeddings for each image
    for image_path in Path(image_dir).glob("*.png"):
        save_DINOv3_embeddings_per_image(image_path, embed_dir)


def save_DINOv3_embeddings(image_dir, embed_dir, batch_size=32):
    feature_extractor = pipeline(
        model="facebook/dinov3-vitl16-pretrain-lvd1689m",
        task="image-feature-extraction",
    )

    paths = image_dir.glob("*.png")
    batch_paths = [str(p) for p in paths]

    def run_batch_and_save(batch_paths, batch_files):
        # feats_list is a list of length N, each like tensor (1, 201, 1024)
        import time

        t0 = time.time()
        feats_list = feature_extractor(
            batch_paths, batch_size=batch_size, return_tensors=True
        )
        print("done feature", len(batch_paths), "in", time.time() - t0, "seconds")
        t0 = time.time()
        for fp, f in zip(batch_files, feats_list):
            features = np.asarray(f, dtype=np.float32).squeeze(0)  # (201, 1024)
            save_file = embed_dir / f"{fp.stem}.npz"
            # drop class + 4 register tokens
            np.savez(save_file, feat=features[5:])  # np.savez_compressed much slower
        print("done saving", len(batch_paths), "in", time.time() - t0, "seconds")

    batch_paths = []
    batch_files = []

    for p in image_dir.glob("*.png"):
        batch_files.append(p)
        batch_paths.append(str(p))

        if len(batch_paths) == batch_size:
            run_batch_and_save(batch_paths, batch_files)
            batch_paths.clear()
            batch_files.clear()

    # flush remainder
    if batch_paths:
        run_batch_and_save(batch_paths, batch_files)


def pack_embeddings_bbs_npz(
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


def pack_embeddings_polylines_npz(
    embeddings_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    feat_key="feat",
    poly_key="polylines_px_resampled",
):
    """
    For each embeddings_dir/pos_*.npz:
      - loads feat from npz
      - loads labels_dir/<same_stem>.json
      - reads fixed-length polylines from poly_key
      - writes output_dir/<same_name>.npz with: feat, polylines, labels, image_size
    Skips files with missing labels or zero valid polylines.
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

        H = W = lab["chip_size_px"]
        P = lab["n_points_per_line"]

        polys = []
        for pl in lab.get(poly_key, []):
            if not (isinstance(pl, list) and len(pl) == P):
                continue
            pts = []
            for x, y in pl:
                x = float(x)
                y = float(y)
                x = max(0.0, min(x, float(W)))
                y = max(0.0, min(y, float(H)))
                pts.append([x, y])
            polys.append(pts)

        if len(polys) == 0:
            skipped += 1
            continue

        polylines = np.asarray(polys, dtype=np.float32)  # (N,P,2)
        labels = np.zeros((polylines.shape[0],), dtype=np.int64)  # single class -> 0

        feat = np.load(emb_path)[feat_key]
        feat = np.ascontiguousarray(feat).astype(np.float32)

        out_path = output_dir / emb_path.name
        np.savez(
            out_path,
            feat=feat,
            polylines=polylines,
            labels=labels,
            image_size=np.asarray([H, W], dtype=np.int32),
        )
        written += 1

    print(f"Written: {written}, Skipped: {skipped}, Output: {output_dir}")
