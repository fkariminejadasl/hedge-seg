import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hedge_seg.training_data import generate_dataset


def _read_png(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return arr


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_stems(folder: Path, suffix: str) -> list[str]:
    return sorted(p.stem for p in folder.glob(f"*{suffix}") if p.is_file())


def _assert_same_file_set(
    gt_dir: Path, gen_dir: Path, sub: str, suffix: str
) -> list[str]:
    gt_sub = gt_dir / sub
    gen_sub = gen_dir / sub
    assert gt_sub.exists(), f"Missing gt folder: {gt_sub}"
    assert gen_sub.exists(), f"Missing gen folder: {gen_sub}"

    gt = _list_stems(gt_sub, suffix)
    gen = _list_stems(gen_sub, suffix)

    assert gt == gen, f"File set differs in {sub}. gt={gt}, gen={gen}"
    return gt  # stems


def _assert_equal_pngs(gt_dir: Path, gen_dir: Path, sub: str, stems: list[str]) -> None:
    for stem in stems:
        a = _read_png(gt_dir / sub / f"{stem}.png")
        b = _read_png(gen_dir / sub / f"{stem}.png")
        assert np.array_equal(a, b), f"PNG differs: {sub}/{stem}.png"


def _assert_equal_jsons(gt_dir: Path, gen_dir: Path, stems: list[str]) -> None:
    for stem in stems:
        a = _read_json(gt_dir / "labels" / f"{stem}.json")
        b = _read_json(gen_dir / "labels" / f"{stem}.json")
        assert a == b, f"JSON differs: labels/{stem}.json"


def assert_images_close(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_abs_diff: int = 2,
    max_bad_px_frac: float = 1e-5,
):
    """
    Accept if:
      - shapes match
      - most pixels match closely
    max_abs_diff: allowed absolute difference per channel
    max_bad_px_frac: fraction of pixels allowed to exceed max_abs_diff
    """
    assert a.shape == b.shape, f"shape differs: {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        a = a.astype(np.int16)
        b = b.astype(np.int16)
    else:
        a = a.astype(np.int16)
        b = b.astype(np.int16)

    diff = np.abs(a - b)
    bad = diff > max_abs_diff

    # if color image, count a pixel "bad" if any channel is bad
    if bad.ndim == 3:
        bad_px = bad.any(axis=2)
    else:
        bad_px = bad

    bad_count = int(bad_px.sum())
    total = int(bad_px.size)
    frac = bad_count / max(total, 1)
    print(
        f"Bad pixels: {bad_count}/{total} ({frac:.2e}) with max_abs_diff={max_abs_diff}"
    )

    assert (
        frac <= max_bad_px_frac
    ), f"too many differing pixels: {bad_count}/{total} ({frac:.2e})"


def _assert_close_pngs(gt_dir: Path, gen_dir: Path, sub: str, stems: list[str]) -> None:
    for stem in stems:
        a = _read_png(gt_dir / sub / f"{stem}.png")
        b = _read_png(gen_dir / sub / f"{stem}.png")
        assert_images_close(a, b, max_abs_diff=20, max_bad_px_frac=9e-3)


@pytest.mark.local
@pytest.mark.parametrize(
    "label_mode,use_osm,out_size_px",
    [
        ("both", True, None),
        # ("both", True, None),
        # ("both", True, 512),
    ],
)
def test_generate_dataset_matches_ground_truth(
    tmp_path: Path, label_mode: str, use_osm: bool, out_size_px: int | None
):
    """
    Regression test: compare a freshly generated mini dataset against a ground truth folder.
    """
    base_dir = Path("/home/fatemeh/Downloads/hedge")
    gt_dir = (base_dir / "results/test_mini_gt").resolve()

    # Use a temp output directory for the generated data
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated dataset will be in: {gen_dir}")

    # Use the exact same inputs and parameters you used to create gt_mini
    shp_path = Path(
        "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
    )
    tif_path = Path(
        "/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
    )

    generate_dataset(
        shp_path=shp_path,
        tif_path=tif_path,
        out_dir=gen_dir,
        n_pos=10,
        n_neg=0,
        seed=123,
        max_tries_per_sample=200,
        size_px=256,
        out_size_px=out_size_px,
        label_mode=label_mode,
        use_osm=use_osm,
        band=1,
        line_width_px=2,
    )

    # Compare labels first, since they define the ids
    stems = _assert_same_file_set(gt_dir, gen_dir, "labels", ".json")
    _assert_equal_jsons(gt_dir, gen_dir, stems)

    # Compare images and masks
    img_stems = _assert_same_file_set(gt_dir, gen_dir, "images", ".png")
    assert img_stems == stems
    _assert_equal_pngs(gt_dir, gen_dir, "images", stems)

    # If you generate masks folder in both modes, compare it. If not, only compare when exists in gt.
    gt_masks = (gt_dir / "masks").exists() and any((gt_dir / "masks").glob("*.png"))
    gen_masks = (gen_dir / "masks").exists() and any((gen_dir / "masks").glob("*.png"))
    assert gt_masks == gen_masks, "Mask presence differs between gt and gen"
    if gt_masks:
        mask_stems = _assert_same_file_set(gt_dir, gen_dir, "masks", ".png")
        assert mask_stems == stems
        _assert_equal_pngs(gt_dir, gen_dir, "masks", stems)

    # Compare OSM only if enabled in this test case
    gt_osm = (gt_dir / "osm").exists() and any((gt_dir / "osm").glob("*.png"))
    gen_osm = (gen_dir / "osm").exists() and any((gen_dir / "osm").glob("*.png"))
    assert gt_osm == gen_osm, "OSM presence differs between gt and gen"
    if gt_osm:
        osm_stems = _assert_same_file_set(gt_dir, gen_dir, "osm", ".png")
        assert osm_stems == stems
        # osm images may differ slightly
        _assert_close_pngs(gt_dir, gen_dir, "osm", osm_stems)
