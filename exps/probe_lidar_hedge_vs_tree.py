"""
Can AHN4 lidar tell a hedge from a tree row?

Top10NL separates the two classes by whether the vegetation blocks the view at
about eye level, not by how tall it is (https://kadaster.github.io/imbrt/):

- bomenrij: a row of at least 3 trees, spaced so that "tot manshoogte geen
  zichtbelemmering" (up to man-height it does NOT block the view).
- heg, haag: a row of trees, with or without shrubs, spaced or under-grown so
  that "tot minstens manshoogte het zicht belemmerd wordt" (up to at least
  man-height it DOES block the view). Trunk planting width up to 3 m. Or a row
  of clipped shrubs, at least 1 m high.

So a tall row of trees is a heg if it has understory and a bomenrij if it does
not. Height cannot be the discriminator, but the density of lidar returns in
the 1-3 m layer is almost a literal restatement of the rule, and the pulse
penetration ratio measures the same openness from the ground side.

This samples all 25 AHN4 metrics along features of both layers and asks how
well they separate. Train and test are split by 5 km geographic blocks, the
same reason as for the crops: features cluster, so a random split flatters.

Result (2026-08-10, 2000 features per layer, seed 0, 5 samples per feature):
see the printout. The headline is that vertical structure separates the two
classes far better than height does, which reverses an earlier reading of this
file that used `perc_95` height alone and concluded lidar could not help.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_lidar_hedge_vs_tree.py
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPO = Path("/home/fatemeh/Downloads/hedge/Topo10NL2023")
LIDAR = Path("/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4")

# Short names, from Table 3 of the metrics paper (essd.copernicus.org/articles/
# 17/3641/2025). BR_x_y is the share of vegetation returns between x and y m.
SHORT = {
    "ahn4_10m_band_ratio_normalized_height_1": "BR_below_1",
    "ahn4_10m_band_ratio_1_normalized_height_2": "BR_1_2",
    "ahn4_10m_band_ratio_2_normalized_height_3": "BR_2_3",
    "ahn4_10m_band_ratio_3_normalized_height_4": "BR_3_4",
    "ahn4_10m_band_ratio_4_normalized_height_5": "BR_4_5",
    "ahn4_10m_band_ratio_5_normalized_height_20": "BR_5_20",
    "ahn4_10m_band_ratio_20_normalized_height": "BR_above_20",
    "ahn4_10m_band_ratio_3_normalized_height": "BR_above_3",
    "ahn4_10m_band_ratio_normalized_height_5": "BR_below_5",
    "ahn4_10m_pulse_penetration_ratio": "pulse_penetration",
    "ahn4_10m_density_absolute_mean_normalized_height": "density_above_mean",
    "ahn4_10m_max_normalized_height": "height_max",
    "ahn4_10m_mean_normalized_height": "height_mean",
    "ahn4_10m_median_normalized_height": "height_median",
    "ahn4_10m_perc_25_normalized_height": "height_p25",
    "ahn4_10m_perc_50_normalized_height": "height_p50",
    "ahn4_10m_perc_75_normalized_height": "height_p75",
    "ahn4_10m_perc_95_normalized_height": "height_p95",
    "ahn4_10m_std_normalized_height": "std",
    "ahn4_10m_var_normalized_height": "var",
    "ahn4_10m_coeff_var_normalized_height": "coeff_var",
    "ahn4_10m_sigma_z": "sigma_z",
    "ahn4_10m_skew_normalized_height": "skew",
    "ahn4_10m_kurto_normalized_height": "kurtosis",
    "ahn4_10m_entropy_normalized_height": "entropy",
}


def feature_points(shp, n_feat, n_pts, rng):
    """Points along n_feat random features, plus the feature index of each."""
    gdf = gpd.read_file(shp, columns=[])
    idx = rng.choice(len(gdf), size=min(n_feat, len(gdf)), replace=False)

    points, owner, centers = [], [], []
    for i, geom in enumerate(gdf.geometry.iloc[idx]):
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        part = max(parts, key=lambda p: p.length)
        if part.length < 1:
            continue
        for t in np.linspace(0.1, 0.9, n_pts):
            p = part.interpolate(t, normalized=True)
            points.append((p.x, p.y))
            owner.append(len(centers))
        c = part.interpolate(0.5, normalized=True)
        centers.append((c.x, c.y))
    return np.array(points), np.array(owner), np.array(centers)


def sample_all_metrics(points, owner, n_features):
    """Median of each metric along each feature. Returns (n_features, n_metrics)."""
    names, columns = [], []
    for tif in sorted(LIDAR.glob("*.tif")):
        stem = tif.stem
        if stem not in SHORT:
            print(f"  skipping unknown raster {stem}")
            continue
        t0 = time.time()
        with rasterio.open(tif) as src:
            vals = np.array([v[0] for v in src.sample(points)], dtype=np.float64)
            nodata = src.nodata
        # Each raster carries its own nodata, and some use +3.4e38 while others
        # use -99999. A single blanket filter would silently keep the positive
        # sentinels and poison the band ratios.
        bad = ~np.isfinite(vals)
        if nodata is not None:
            bad |= np.isclose(vals, nodata, rtol=1e-6)
        bad |= np.abs(vals) > 1e30
        vals[bad] = np.nan

        per_feature = np.full(n_features, np.nan)
        for f in range(n_features):
            v = vals[owner == f]
            if np.any(np.isfinite(v)):
                per_feature[f] = np.nanmedian(v)
        names.append(SHORT[stem])
        columns.append(per_feature)
        print(
            f"  {SHORT[stem]:20s} {time.time() - t0:5.1f}s "
            f"valid {np.mean(np.isfinite(per_feature)):.1%}"
        )
    return names, np.column_stack(columns)


def block_split(centers, block_m, test_fraction, rng):
    """Assign whole 5 km blocks to test, so nearby features cannot straddle."""
    blocks = np.floor(centers / block_m).astype(np.int64)
    keys = [tuple(b) for b in blocks]
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_fraction)))
    test_blocks = set(uniq[:n_test])
    return np.array([k in test_blocks for k in keys])


def single_cut_accuracy(x, y):
    """Best balanced accuracy from one threshold on one metric, either sign."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(np.unique(y)) < 2:
        return 0.5
    cuts = np.percentile(x, np.arange(2, 99, 2))
    best = 0.5
    for c in np.unique(cuts):
        for pred in (x >= c, x < c):
            best = max(best, balanced_accuracy_score(y, pred))
    return best


def main(cfg):
    rng = np.random.default_rng(cfg["seed"])

    print("sampling features")
    pts_h, own_h, cen_h = feature_points(
        TOPO / cfg["hedge_shp"], cfg["n_features"], cfg["n_points"], rng
    )
    pts_t, own_t, cen_t = feature_points(
        TOPO / cfg["treeline_shp"], cfg["n_features"], cfg["n_points"], rng
    )
    n_h, n_t = len(cen_h), len(cen_t)
    points = np.vstack([pts_h, pts_t])
    owner = np.concatenate([own_h, own_t + n_h])
    centers = np.vstack([cen_h, cen_t])
    y = np.concatenate([np.zeros(n_h, int), np.ones(n_t, int)])  # 1 = tree row
    print(f"  heg {n_h}, bomenrij {n_t}, {len(points)} sample points\n")

    print(f"sampling {len(SHORT)} lidar metrics")
    names, X = sample_all_metrics(points, owner, len(centers))

    keep = np.isfinite(X).all(axis=1)
    X, y, centers = X[keep], y[keep], centers[keep]
    print(
        f"\n{keep.sum()} of {len(keep)} features have all metrics "
        f"({np.mean(y == 0):.0%} heg, {np.mean(y == 1):.0%} bomenrij)"
    )

    print("\nper-metric medians and best single cut, ranked:")
    print(f"  {'metric':20s} {'heg':>10s} {'bomenrij':>10s}  best cut acc")
    rows = []
    for j, name in enumerate(names):
        rows.append(
            (
                single_cut_accuracy(X[:, j], y),
                name,
                np.median(X[y == 0, j]),
                np.median(X[y == 1, j]),
            )
        )
    for acc, name, m0, m1 in sorted(rows, reverse=True):
        print(f"  {name:20s} {m0:10.3f} {m1:10.3f}      {acc:.3f}")

    is_test = block_split(centers, cfg["block_m"], cfg["test_fraction"], rng)
    print(
        f"\ngeographic split: train {int((~is_test).sum())}, "
        f"test {int(is_test.sum())} ({cfg['block_m'] / 1000:.0f} km blocks)"
    )

    print("\nmultivariate, scored on the held-out blocks:")
    for label, model, cols in [
        (
            "height only (p95)",
            HistGradientBoostingClassifier(random_state=0),
            [names.index("height_p95")],
        ),
        (
            "all height metrics",
            HistGradientBoostingClassifier(random_state=0),
            [j for j, n in enumerate(names) if n.startswith("height_")],
        ),
        (
            "structure only (no height)",
            HistGradientBoostingClassifier(random_state=0),
            [j for j, n in enumerate(names) if not n.startswith("height_")],
        ),
        (
            "all 25, logistic",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            list(range(len(names))),
        ),
        (
            "all 25, gradient boosting",
            HistGradientBoostingClassifier(random_state=0),
            list(range(len(names))),
        ),
    ]:
        model.fit(X[~is_test][:, cols], y[~is_test])
        prob = model.predict_proba(X[is_test][:, cols])[:, 1]
        acc = balanced_accuracy_score(y[is_test], prob >= 0.5)
        auc = roc_auc_score(y[is_test], prob)
        print(f"  {label:28s} balanced acc {acc:.3f}   AUC {auc:.3f}")


if __name__ == "__main__":
    cfg = dict(
        hedge_shp="Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp",
        treeline_shp="Treelines_polylines/"
        "Top10NL2023_inrichtingselementen_lijn_bomenrij.shp",
        n_features=2000,  # per layer
        n_points=5,  # samples along each feature, reduced to a median
        seed=0,
        block_m=5000.0,  # same block size as the crop split
        test_fraction=0.3,
    )
    main(cfg)
