"""
Can AHN4 height tell a hedge from a tree row?

Adding tree rows as a second class only helps if the two can be separated. The
idea was that height does it, since a tree row is tall and a hedge is short.
This samples the AHN4 95th-percentile normalized-height raster along a random
sample of features from both Top10NL layers and compares them.

Result (2026-08-10, 2,000 features per layer, seed 0):

    layer               p25      median   p75
    heg (hedge)         4.0 m    8.0 m    12.3 m
    bomenrij (tree row) 8.3 m    12.2 m   15.7 m

Best single height cut is 9.0 m at balanced accuracy 0.638, so height alone
does not separate them. The reason is that Top10NL "heg, haag" covers tall
hedgerows and houtwallen, not only clipped hedges.

Two caveats make this a lower bound: 10 m cells pick up neighbouring trees and
buildings, and AHN4 is about 2020 while the labels come from 2014 photos. But
do not schedule the LiDAR branch as the way to separate the two classes.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_lidar_hedge_vs_tree.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPO = Path("/home/fatemeh/Downloads/hedge/Topo10NL2023")
LIDAR = Path("/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4")


def sample_layer(shp, name, src, rng, n_feat, n_pts):
    """Median height along each of n_feat random features of a layer."""
    gdf = gpd.read_file(shp, columns=[])
    idx = rng.choice(len(gdf), size=min(n_feat, len(gdf)), replace=False)

    points, spans = [], []
    for geom in gdf.geometry.iloc[idx]:
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        part = max(parts, key=lambda p: p.length)
        if part.length < 1:
            continue
        along = [
            part.interpolate(t, normalized=True) for t in np.linspace(0.05, 0.95, n_pts)
        ]
        spans.append((len(points), len(along)))
        points.extend((p.x, p.y) for p in along)

    values = np.array([v[0] for v in src.sample(points)], dtype=float)
    values[values < -100] = np.nan  # nodata
    per_feature = np.array(
        [
            (
                np.nan
                if np.all(np.isnan(values[s : s + n]))
                else np.nanmedian(values[s : s + n])
            )
            for s, n in spans
        ]
    )
    per_feature = per_feature[np.isfinite(per_feature)]

    qs = np.percentile(per_feature, [10, 25, 50, 75, 90])
    print(f"\n{name}: {len(per_feature)} features")
    print(
        "  height along the line (m): "
        + "  ".join(f"p{p}={v:.1f}" for p, v in zip([10, 25, 50, 75, 90], qs))
    )
    return per_feature


def main(cfg):
    rng = np.random.default_rng(cfg["seed"])
    with rasterio.open(LIDAR / cfg["height_tif"]) as src:
        heg = sample_layer(
            TOPO / cfg["hedge_shp"],
            "heg (hedge)",
            src,
            rng,
            cfg["n_features"],
            cfg["n_points"],
        )
        tree = sample_layer(
            TOPO / cfg["treeline_shp"],
            "bomenrij (tree row)",
            src,
            rng,
            cfg["n_features"],
            cfg["n_points"],
        )

    print("\nseparation by a single height cut:")
    print("   cut    hedge below    tree above    balanced acc")
    best = (None, -1.0)
    for cut in np.arange(cfg["cut_min"], cfg["cut_max"] + 0.01, cfg["cut_step"]):
        below, above = float((heg < cut).mean()), float((tree >= cut).mean())
        acc = (below + above) / 2
        print(f"  {cut:4.1f} m     {below:.3f}         {above:.3f}         {acc:.3f}")
        if acc > best[1]:
            best = (cut, acc)
    print(f"\nbest cut {best[0]:.1f} m, balanced accuracy {best[1]:.3f} (0.5 = chance)")


if __name__ == "__main__":
    cfg = dict(
        height_tif="ahn4_10m_perc_95_normalized_height.tif",
        hedge_shp="Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp",
        treeline_shp="Treelines_polylines/"
        "Top10NL2023_inrichtingselementen_lijn_bomenrij.shp",
        n_features=2000,
        n_points=8,  # samples along each feature, reduced to a median
        seed=0,
        cut_min=2.0,
        cut_max=12.0,
        cut_step=0.5,
    )
    main(cfg)
