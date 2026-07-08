# Description

## Data

Ground truth: All LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 out of 62,415 items, 2 pts 13,307.

Lowres LiDAR height data: In 256 x 256 image the max number of polylines are 441 and after removal of short lengths or 10 pixels 276 and per polyline we take 20 points. The number of images 17,000.

There are 691,006 polylines and 4121 closed shape, which some are not originally closed but start and end are closeby, e.g. test_256_None.

Highres aerial image: 25 cm per pixel, 1000 x 1000 image crop. Max number of polylines 50. 106,812 polylines in 30,000 images, e.g. pdok_dataset3.

## Overview

- **Initial Investigation**: SAM3, PiDiNet, conventional methods for hedgerow delineation.

- **train_neg_pos_classifier**: Validated DINOv3 on LiDAR-crop classification (hedge vs. non-hedge from ground truth). Result: Good performance. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detr_hf**: Explored HuggingFace DETR for polyline prediction. Decided to implement custom DETR instead, as HF DETR is streamlined for object detection and difficult to adapt. Data generated in-code.

- **train_detr_dino**: DETR baseline before polyline implementation. Data: `hedge_seg.embeddings_and_pack` (bounding box embeddings).

- **train_detr_dino_polyline**: DETR-like polyline with LiDAR crops, DINOv3 frozen. Added optional diffusion refinement head (3-stage training). **Result: Learned polyline distribution but not precise locations**, even with diffusion. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detr_dino_polyline_rel**: Same as above, but predicted per-query bounding box first, then polyline points as offsets relative to box. Added geometric losses (`loss_smooth`, `loss_len`, `loss_dir`, `loss_card`). **Result: Still could not learn precise locations, only distribution**. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detect_ultralytics**: YOLO bounding-box detection on high-resolution PDOK aerial crops to test if low image resolution was the issue. **Result: 70% precision, 60% recall**. Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_yolo_bbox.py}`.

- **train_seg_ultralytics**: YOLO instance segmentation with polylines buffered to 15 m radius on high-resolution aerial data. **Result: 80% precision, 70% recall**; outperformed bounding-box detection. Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_yolo_seg.py}`.

- **train_semseg_unet_resnet18**: Since integrating Ultralytics with polyline DETR was impractical, trained ResNet18-UNet for binary semantic segmentation (hedge mask + centerline) on high-resolution aerial data with 15 m polyline buffers. **Result: 60% precision, 40% recall numerically, but visual segmentation quality was reasonable**. Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_semseg.py}`.

- **train_detr_maptr_polyline**: Reimplemented polyline regression without diffusion, using MapTR hierarchical queries (per-polyline instance + per-point embeddings). Initialized backbone with pretrained ResNet18-UNet. Hypothesized hierarchical query design limited precision learning, but results similar to relative-coordinate approach.

### Key Findings

- **Polyline regression from embeddings** learns distribution well but struggles with precise coordinate prediction.
- **High-resolution imagery** (PDOK aerial at 25 cm/px) significantly improves detection/segmentation over low-resolution LiDAR crops.
- **Diffusion refinement** and **relative-coordinate heads** provide minimal improvement for location precision.
- **Segmentation-based approaches** (YOLO-seg, UNet) outperform direct polyline regression, suggesting the task may be better framed as mask generation.



## Code Description

### Core Library Modules

- `hedge_seg/training_data.py`: generates images and labels from a local hedge shapefile plus a local GeoTIFF raster, currently used for LiDAR-derived training data.
- `hedge_seg/label_postprocess.py`: resamples polylines to equidistant points and adds derived per-segment annotations (e.g. bounding boxes) as a post-processing pass over generated labels.
- `hedge_seg/embeddings_and_pack.py`: computes DINOv3 image embeddings (sequential or batched) and packs them together with (post-processed) labels into the `embs_*` datasets consumed by the DETR-from-embeddings training scripts.
- `hedge_seg/pdok_training_data.py`: generates images and labels by requesting aerial imagery directly from PDOK WMS and combining it with the hedge shapefile.
- `hedge_seg/ultralytics_export.py`: shared utility functions for exporting generated JSON/polyline datasets to Ultralytics/YOLO dataset layouts. It contains reusable path creation, image/label pairing, train/val splitting, image copying, dataset YAML writing, JSON loading, image-size reading, and common coordinate helpers.
- `hedge_seg/utils.py`: dataset inspection helpers (e.g. counting polylines/points per JSON label file) used ad hoc for dataset stats.
- `hedge_seg/visualization.py`: plotting helpers for inspecting datasets and predictions, e.g. drawing YOLO boxes/segmentation and polylines on top of chip images.
- `hedge_seg/training_utils.py`: small shared training helpers, currently `set_seed` for reproducibility across `random`/`numpy`/`torch`.

### Scripts

#### Data

- `scripts/data/build_lidar_training_dataset.py`: end-to-end local-raster pipeline. It creates images and labels with `hedge_seg.training_data`, postprocesses labels with `hedge_seg.label_postprocess`, computes DINOv3 embeddings, and packs embeddings with labels using `hedge_seg.embeddings_and_pack`.
- `scripts/data/build_pdok_wms_dataset.py`: entry point for the PDOK WMS data-generation workflow in `hedge_seg.pdok_training_data`.
- `scripts/data/convert_pdok_polylines_to_yolo_bbox.py`: converts the generated JSON/polyline dataset into an Ultralytics detection dataset. Each polyline is converted to one normalized YOLO bounding box row.
- `scripts/data/convert_pdok_polylines_to_yolo_seg.py`: converts the generated JSON/polyline dataset into an Ultralytics segmentation dataset. Each polyline is buffered into a thin polygon mask and written as a normalized YOLO segmentation row.
- `scripts/data/convert_pdok_polylines_to_semseg.py`: converts the generated JSON/polyline dataset into a semantic-segmentation dataset (`images/`, `masks/`, `centerlines/` train/val folders). Each polyline is buffered into a foreground mask and its raw skeleton is kept separately as the centerline target; used by `train_semseg_unet_resnet18.py`.

#### Training

All training scripts read datasets produced by the `scripts/data/*` workflows above and write TensorBoard logs plus best-checkpoint `.pt` files (except the Ultralytics ones, which use Ultralytics' own trainer/logging).

- `scripts/train_neg_pos_classifier.py`: trains a small linear-probe MLP (`LinearProbe`/`MLP`) on packed DINOv3 patch embeddings to classify chip-level embedding files as positive (contains hedge, filename has "pos") vs. negative.
- `scripts/train_detect_ultralytics.py`: trains a YOLO (Ultralytics) object-detection model on the YOLO bbox dataset from `convert_pdok_polylines_to_yolo_bbox.py`; also contains a commented-out inference/visualization snippet using `hedge_seg.visualization.draw_yolo_bounding_box_on_image`.
- `scripts/train_seg_ultralytics.py`: trains a YOLO-seg (Ultralytics) instance-segmentation model on the YOLO segmentation dataset from `convert_pdok_polylines_to_yolo_seg.py`, initialized from the detection checkpoint produced by `train_detect_ultralytics.py`.
- `scripts/train_semseg_unet_resnet18.py`: trains a ResNet18-UNet for binary semantic segmentation (hedge vs. background) plus a centerline channel, directly from RGB aerial images on the dataset produced by `convert_pdok_polylines_to_semseg.py`.
- `scripts/train_detr_hf.py`: fine-tunes a Hugging Face `AutoModelForObjectDetection` (DETR) on a COCO-style bounding-box dataset built from the chip images/labels; adapted from the HF "fine-tuning DETR" cookbook. Independent of the custom DETR implementations below (uses `transformers.Trainer` instead of a hand-written loop).
- `scripts/train_detr_dino.py`: trains a from-scratch DETR-style encoder/decoder (`DetrFromEmbeddings`) with a Hungarian matcher and class/bbox/GIoU losses, operating on precomputed DINOv3 patch embeddings + bounding boxes (via `DetrEmbDataset`) rather than raw images.
- `scripts/train_detr_dino_polyline.py`: extends the DINO-embedding DETR to predict whole polylines (`DetrPolylineFromEmbeddings`, `pred_polylines` of shape `(Q, K, 2)`) instead of boxes, with a `HungarianMatcherPolyline`/`DetrPolylineCriterion`, and adds an optional second-stage diffusion refinement head (`DetrWithDiffusion` / `GaussianPolylineDiffusion`) trained in stages (`cfg.stage`: 1 = DETR polyline, 2 = diffusion only, 3 = joint).
- `scripts/train_detr_dino_polyline_rel.py`: same DETR+diffusion staged design as `train_detr_dino_polyline.py`, but the polyline head first predicts a per-query bounding box (`pred_boxes`, cxcywh) and then predicts each polyline point as a bounded offset relative to that box (`cxcy + tanh(offset) * wh`) rather than regressing absolute point coordinates directly; adds extra matching/loss terms (`poly_cost`, `loss_smooth`, `loss_len`, `loss_dir`, `loss_card`, aux losses).
- `scripts/train_detr_maptr_polyline.py`: variant of `train_detr_dino_polyline_rel.py` using a MapTR-style query layout (per-polyline instance embedding combined with per-point embeddings, `num_polylines x num_points` queries) instead of DINO-style flat polyline queries; drops the diffusion stage; a single `main(cfg)` supports both `cfg.mode="train"` and `cfg.mode="infer"` (loading from `cfg.infer_embed_dir` or a `cfg.infer_split_file` file list), so there is no separate inference script for this variant.

#### Inference

- `scripts/infer_detr_dino_polyline.py`: loads a checkpoint from `train_detr_dino_polyline.py` (plain `model` or `model_with_diffusion`) and runs/visualizes polyline predictions on precomputed embeddings.
- `scripts/infer_detr_dino_polyline_rel.py`: same as above for checkpoints from `train_detr_dino_polyline_rel.py` (box + relative-offset polyline head).


### Experiments

- `exps/data.py`: raster/vector helpers (load GeoTIFF, clip/window a raster to a bbox, rasterize hedge polylines) shared by the training-data workflows.
- `exps/pdok_wmts_training_data.py`: experimental WMTS workflow. It fetches PDOK WMTS tiles, caches them, builds a local GeoTIFF for a bbox, and can then generate training samples from that local cache.