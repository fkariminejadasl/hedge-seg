"""
End-to-end dataset prep for hedge segmentation:

1) Samples positive training patches (chips) from a hedge polyline shapefile and a LiDAR raster (multiprocessing),
   then merges the generated dataset parts. This step can be rerun to generate additional data.
2) Post-processes labels by resampling polylines to a fixed number of points and computing bounding boxes.
3) Extracts DINOv3 image embeddings for all generated patches (chips).
4) Packs embeddings together with the processed polylines into NPZ files for downstream training/experiments.
"""

from pathlib import Path

from hedge_seg.embeddings_and_pack import (
    pack_embeddings_polylines_npz,
    save_DINOv3_embeddings,
)
from hedge_seg.label_postprocess import process_labels_for_dir
from hedge_seg.training_data import generate_dataset_mp, merge_parts


def main():
    # ----------------------------
    # Training data generation
    # ----------------------------
    n_pos = 10  # 300_000
    n_proc = 2  # or min(10, os.cpu_count())
    size_px = 64
    out_size_px = 4 * size_px
    dir_name = "test_mini4"  # f"test_{res}"
    out_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")
    shp_path = Path(
        "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
    )
    tif_path = Path(
        "/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
    )
    seed = 123
    max_tries = 200
    n_neg = 0
    label_mode = "polylines"
    use_osm = False
    n_points = 20

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset with {n_pos} positive samples...")
    # from hedge_seg.training_data import generate_dataset
    # generate_dataset(
    #     shp_path=shp_path,
    #     tif_path=tif_path,
    #     out_dir=out_dir,
    #     n_pos=n_pos,
    #     n_neg=0,
    #     size_px=256,
    #     out_size_px=None, # 256
    #     seed=seed,
    #     label_mode=label_mode,
    #     use_osm=True,
    # )
    generate_dataset_mp(
        n_pos,
        n_neg,
        n_proc,
        shp_path,
        tif_path,
        out_dir,
        size_px,
        out_size_px,
        seed,
        max_tries,
        label_mode,
        use_osm,
    )

    print(f"Dataset parts generated in: {out_dir}. Merging parts...")
    merge_parts(out_dir)

    # """
    # ----------------------------
    # Process labels (resample polylines, get bboxes)
    # ----------------------------
    labels_dir = out_dir / "labels"
    plabels_dir = out_dir / "labels_processed"
    plabels_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing labels: resampling polylines to {n_points} points, getting bboxes..."
    )
    process_labels_for_dir(labels_dir, plabels_dir, n_points)

    # ----------------------------
    # Save DINOv3 embeddings
    # ----------------------------
    # huggingface-cli login # from ~/.cache/huggingface/token
    image_dir = out_dir / "images"
    embed_dir = out_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving DINOv3 embeddings for images in: {image_dir}...")
    save_DINOv3_embeddings(image_dir, embed_dir, batch_size=512)

    # ----------------------------
    # Pack embeddings and polylines into npz
    # ----------------------------
    print(
        f"Packing embeddings and polylines into npz files in: {out_dir / 'embs_polylines'}..."
    )
    pack_embeddings_polylines_npz(
        embeddings_dir=out_dir / "embeddings",
        labels_dir=out_dir / "labels_processed",
        output_dir=out_dir / "embs_polylines",
    )
    # """


if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.freeze_support()  # harmless on Linux, needed when frozen, OK to keep
    main()
