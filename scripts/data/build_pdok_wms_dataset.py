from pathlib import Path

from omegaconf import OmegaConf

from hedge_seg import pdok_training_data as pdok


def main():
    cfg = dict(
        shp_path=Path(
            "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
        ),
        bbox_csv=None,  # or Path("/home/fatemeh/Downloads/hedge/area_bbox.csv")
        out_dir=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset"),
        n_pos=10,
        num_workers=8,
        seed=123,
        crs="EPSG:28992",
        layer_name="Actueel_ortho25",  # or "2018_ortho25"
        chip_size_m=250.0,
        out_size_px=1000,
        image_format="image/jpeg",
        timeout=120,
        min_len_px=10.0,
    )
    cfg = OmegaConf.create(cfg)
    pdok.build_dataset(cfg)


if __name__ == "__main__":
    main()
