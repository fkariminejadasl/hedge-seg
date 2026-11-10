import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from show_polyline_results import figure_path, split_run_name  # noqa: E402


def _cfg(tmp_path, tag=""):
    return OmegaConf.create(
        dict(
            save=True,
            save_dir=str(tmp_path),
            score_thresh=0.9,
            tag=tag,
            model="detr_unet_polyline",
        )
    )


def test_page_number_follows_the_model_name(tmp_path):
    run_dir = tmp_path / "10_val_t0.05"
    cfg = _cfg(tmp_path)
    names = [figure_path(cfg, run_dir, "10", page=p).name for p in range(2)]
    assert names == [
        "detr_unet_polyline_p1_10_val_t.90.png",
        "detr_unet_polyline_p2_10_val_t.90.png",
    ]


def test_tag_comes_after_the_page_number(tmp_path):
    run_dir = tmp_path / "best_2_val_cluster_t0.05"
    path = figure_path(_cfg(tmp_path, tag="trees"), run_dir, "gt", page=0)
    assert path.name == "detr_unet_polyline_p1_trees_gt_val_cluster_t.90.png"


def test_old_val_cluster_run_names_still_parse():
    # Run directories made before 2026-11-10 carry val_cluster in their name.
    assert split_run_name(Path("best_10_val_t0.05")) == ("best_10", "val")
    assert split_run_name(Path("best_2_val_cluster_t0.05")) == ("best_2", "val_cluster")
