# Description

## Data

All LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 out of 62,415 items, 2 pts 13,307.
In 256x256 image the max number of polylines are 441 and after removal of short lengths or 10 pixels 276 and per polyline we take 20 points.

There are 691006 polylines and 4121 closed shape, which some are not originally closed but start and end are closeby.

## Training Data Workflows

There are three related data-generation workflows:

- `hedge_seg/training_data.py`: generates chips and labels from a local hedge shapefile plus a local GeoTIFF raster, currently used for LiDAR-derived training data.
- `hedge_seg/pdok_training_data.py`: generates chips and labels by requesting aerial imagery directly from PDOK WMS and combining it with the hedge shapefile.
- `exps/pdok_wmts_training_data.py`: experimental WMTS workflow. It fetches PDOK WMTS tiles, caches them, builds a local GeoTIFF for a bbox, and can then generate training samples from that local cache.

## Scripts

- `scripts/data/build_lidar_training_dataset.py`: end-to-end local-raster pipeline. It creates images and labels with `hedge_seg.training_data`, postprocesses labels with `hedge_seg.label_postprocess`, computes DINOv3 embeddings, and packs embeddings with labels using `hedge_seg.embeddings_and_pack`.
- `scripts/data/build_pdok_wms_dataset.py`: example entry point for the PDOK WMS data-generation workflow in `hedge_seg.pdok_training_data`.
- `scripts/data/convert_to_ultralytics_format.py`: converts the generated JSON/polyline dataset into a YOLO/Ultralytics dataset layout.
