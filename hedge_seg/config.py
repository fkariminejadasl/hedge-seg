import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("HEDGE_SEG_DATA_ROOT", "/home/fatemeh/Downloads/hedg")
)

TOPO10NL_ROOT = DATA_ROOT / "Topo10NL2023"
HEDGES_POLYLINES_SHP = (
    TOPO10NL_ROOT
    / "Hedges_polylines"
    / "Top10NL2023_inrichtingselementen_lijn_heg.shp"
)

LIDAR_METRICS_ROOT = DATA_ROOT / "LiDAR_metrics_AHN4"
AHN4_HEIGHT_TIF = LIDAR_METRICS_ROOT / "ahn4_10m_perc_95_normalized_height.tif"

RESULTS_DIR = DATA_ROOT / "results"
TEST_DATASET_DIR = RESULTS_DIR / "test_dataset"
TEST_DATASET_WITH_OSM_DIR = RESULTS_DIR / "test_dataset_with_osm"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"

BUFFER_VIEW_PATH = RESULTS_DIR / "buffer_view.png"
SAMPLE_POS_LABEL_PATH = TEST_DATASET_WITH_OSM_DIR / "labels" / "pos_000000.json"
SAMPLE_POS_IMAGE_PATH = TEST_DATASET_WITH_OSM_DIR / "images" / "pos_000000.png"
