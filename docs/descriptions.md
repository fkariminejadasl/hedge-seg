# Description

What exists: the data, the model and every script. Results and next steps are
in `docs/status.md`, lessons in `docs/lesson_learned.md`.

## Data

### Labels

- Hedges: the Top10NL2023 layer `inrichtingselementen_lijn_heg`, 62,415 line features. Tree rows: `inrichtingselementen_lijn_bomenrij`, 266,783 features, class 1 in the tree row dataset. TODO 8 in `docs/status.md` adds the script for these counts.
- What the labels hold and miss, and how old they are: "Labels and data" in `docs/lesson_learned.md`.

### Aerial images (pdok_dataset3)

The data of every polyline run.

- 30,000 crops of 1000 x 1000 px at 25 cm per pixel, so 250 m x 250 m on the ground. One crop per sampled hedge polyline.
- 106,793 hedge polylines inside the crops, at most 50 per crop. `scripts/data/convert_pdok_polylines_to_detr_polyline.py` opens the 302 closed rings (0.45% of the 67,422 polylines with at least 3 points) and drops the 3,377 polylines under 10 m (3.16%). 103,414 are left: two more fall under 10 m after resampling to 20 points (`exps/dataset_stats.py`).
- Crops overlap. 72.6% of crops overlap at least one other crop, up to 41. A random 80/20 split would put 68.5% of val crops on top of a training crop (`exps/quantify_geographic_crop_overlap_pdok_dataset3.py`).

### Train/val split

Split by location, in `spatial_train_val_split` of `scripts/data/convert_pdok_polylines_to_detr_polyline.py`:

1. The country is cut into 5 km x 5 km blocks. A seeded hash puts 20% of the blocks in val and the rest in train.
1. Crops that overlap are joined into groups. A group goes to val only if every crop in it lies in a val block and none overlaps the pdok_dataset2 crops the UNet backbone was trained on. Otherwise the whole group goes to train.
1. An assert checks that no val crop overlaps a train crop.

This gives train 26,902 and val 3,098 crops. Val is 10.3%, not 20%, because groups that overlap the backbone's crops go to train. Every run is scored on these 3,098.

The split guarantees no overlap, not a distance. Because whole groups go to train, train crops also sit inside val blocks: 235 of the 285 blocks that hold val crops also hold train crops. The gap from a val crop to its nearest train crop, edge to edge, is 593 m at the median. 5% are under 50 m, 95% under 2.1 km, and the largest is 4.5 km. 3 val crops touch a train crop. So val is ground the model has not seen, but mostly nearby ground, not a distant region (`exps/probe_split_gap.py`).

The block size only sets the val fraction. 5 km is much larger than a 250 m crop, so most groups of overlapping crops fit inside one block, and it still gives many blocks to hash. Without the backbone filter it gives 19.1% val against the 20% target (`exps/quantify_geographic_crop_overlap_pdok_dataset3.py`). If the exact fraction ever matters, retry seeds rather than tune the block size.

## Model

`scripts/train_detr_unet_polyline.py` as in exp 9, the best run. 20.2 M parameters (`exps/probe_model_size.py`):

- Backbone, ResNet18-UNet: 14.4 M. The model reads its features at up3, so 13.5 M are used. The other 0.85 M (up2, up1, up0 and the segmentation head) are carried along and never used.
- Polyline head: 5.8 M. Transformer encoder (1 layer) 0.8 M, decoder (4 layers) 4.2 M, the rest (input projection, queries, point refinement, class head) 0.8 M.
- Input: the crop padded with black to 1024 x 1024, on the bottom and right, so it divides by the backbone stride of 16. The backbone turns it into 64 x 64 = 4,096 tokens of 256 numbers.
- Output: 60 polylines of 20 points, each with a hedge score.
- Point loss: `loss_poly` is the mean L1 error per coordinate, on points divided by `pad_to - 1` = 1023. Times 1023 gives pixels, and times 0.25 more gives metres, so a loss of 0.004 is about 1 m (`meters_to_px` in `hedge_seg/metrics.py`). x and y are averaged separately, so the distance from a point to its target is up to 1.4 times larger.
- Exps 1 to 8 froze the backbone, so only the 5.8 M head trained. Exp 9 trains all of it, the backbone at a 10x lower learning rate (`backbone_lr=1e-5`).

## What was tried, in order

- **Initial investigation**: SAM 3, PiDiNet and conventional methods for hedgerow delineation. Tested directly, neither SAM 3 nor PiDiNet gave usable hedge lines.

- **train_neg_pos_classifier**: checked DINOv3 on lidar crops, hedge against no hedge from the labels. Result: good. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detr_hf**: tried Hugging Face DETR for box detection. It is built for boxes and hard to adapt to polylines, so a custom DETR followed. Data made in the script.

- **train_detr_dino**: DETR baseline on boxes, before the polyline version. Data: `hedge_seg.embeddings_and_pack` (DINOv3 embeddings with boxes).

- **train_detr_dino_polyline**: DETR-like polylines on lidar crops, DINOv3 frozen, with an optional diffusion refinement head trained in 3 stages. Result: it learned where hedges tend to be, but not where a given hedge is, even with diffusion. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detr_dino_polyline_rel**: as above, but each query first predicts a box and then its points as offsets inside it, with extra geometric losses (`loss_smooth`, `loss_len`, `loss_dir`, `loss_card`). Result: still only where hedges tend to be. Dataset: `scripts/data/build_lidar_training_dataset.py`.

- **train_detect_ultralytics**: YOLO box detection on the 25 cm PDOK crops, to test whether the low resolution of the lidar crops was the problem. Result: precision 0.68, recall 0.54 by Ultralytics' own score, where a box counts as found when it overlaps a labelled box by at least half (run 3, last epoch, `/home/fatemeh/Downloads/hedge/snellius/detect_ultralytics/3/results.csv`). Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_yolo_bbox.py}`.

- **train_seg_ultralytics**: YOLO instance segmentation on the same crops, each polyline buffered by 15 m. Result: mask precision 0.81, recall 0.68 by the same kind of score, better than boxes (run 1, last epoch, `/home/fatemeh/Downloads/hedge/snellius/seg_ultralytics/1/results.csv`). Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_yolo_seg.py}`.

- **train_semseg_unet_resnet18**: ResNet18-UNet that colours hedge pixels (mask plus centre line) on the same crops, polylines buffered by 15 m, because Ultralytics was impractical to combine with a polyline DETR. Result: pixel precision 0.55, recall 0.48 at a pixel cut-off of 0.1, with masks that looked reasonable (run 4, last epoch, `/home/fatemeh/Downloads/hedge/snellius/semseg_unet/4/4_23187501.out`). This network is the backbone of the line model. Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_semseg.py}`.

  These three scores come from each tool's own measure, on a random split of overlapping crops, so they cannot be set beside the line model. Scoring one of them the same way is a TODO in `docs/status.md`.

- **train_detr_unet_polyline**: polyline regression without diffusion, with the MapTR parts: hierarchical queries (one per line, one per point), per-layer reference-point refinement, learned content queries, geometric losses off. Backbone: the ResNet18-UNet of `train_semseg_unet_resnet18`, frozen in exps 1 to 8 and trained in exp 9. Result: the first polyline model that draws lines on real hedges, not only where hedges tend to be. Best runs and all numbers: `docs/status.md`. Dataset: `scripts/data/{build_pdok_wms_dataset.py,convert_pdok_polylines_to_detr_polyline.py}`. **train_detr_maptr_polyline** is the same head on DINOv3 embeddings, with another dataset and dataloader.

## Code Description

### Core Library Modules

- `hedge_seg/training_data.py`: generates images and labels from a local hedge shapefile plus a local GeoTIFF raster, currently used for LiDAR-derived training data. Its line helpers (`lines_in_bbox`, `geom_to_lines`, `clip_px_point`, ...) also make every PDOK label, through `make_polylines_for_chip` in `hedge_seg/pdok_training_data.py`. `tests/test_training_data.py` checks them against a saved 10-crop copy in `test_mini_gt`; after an intended change, rebuild that copy with `PYTHONPATH=. python tests/test_training_data.py`.
- `hedge_seg/label_postprocess.py`: resamples polylines to equidistant points and adds derived per-segment annotations (e.g. bounding boxes) as a post-processing pass over generated labels.
- `hedge_seg/embeddings_and_pack.py`: computes DINOv3 image embeddings (sequential or batched) and packs them together with (post-processed) labels into the `embs_*` datasets consumed by the DETR-from-embeddings training scripts.
- `hedge_seg/pdok_training_data.py`: generates images and labels by requesting aerial imagery directly from PDOK WMS and combining it with the hedge shapefile.
- `hedge_seg/ultralytics_export.py`: shared utility functions for exporting generated JSON/polyline datasets to Ultralytics/YOLO dataset layouts. It contains reusable path creation, image/label pairing, train/val splitting, image copying, dataset YAML writing, JSON loading, image-size reading, and common coordinate helpers.
- `hedge_seg/utils.py`: dataset inspection helpers (e.g. counting polylines/points per JSON label file) used ad hoc for dataset stats.
- `hedge_seg/visualization.py`: plotting helpers for inspecting datasets and predictions, e.g. drawing YOLO boxes/segmentation and polylines on top of chip images. `show_polyline_grid` is the 4x4 comparison view used by `scripts/show_polyline_results.py`, with each line coloured by class (hedge blue, tree row orange) and a legend when both are drawn; `show_polyline_single` is one crop at full size with optional polylines and an optional raster overlay, for looking hard at one thing rather than comparing many. The overlay is stretched with `extent` and drawn with nearest-neighbour interpolation, so a 25x25 lidar patch needs no resampling and its cell edges stay visible, which is what makes a half-cell misalignment obvious.
- `hedge_seg/training_utils.py`: small shared training helpers, currently `set_seed` for reproducibility across `random`/`numpy`/`torch`.
- `hedge_seg/paths.py`: resolves `DATA_ROOT`, `EXP_ROOT` and `CLUSTER_EXP_ROOT` from a filesystem marker, so the same script runs on the laptop and on Snellius with no path edits. `CLUSTER_EXP_ROOT` is `~/exps/hedge` on the cluster and the local mirror `~/Downloads/hedge/snellius` on the laptop, which is why checkpoint configs point at it rather than at `EXP_ROOT`.
- `hedge_seg/metrics.py`: detection metrics for polyline predictions. `buffered_length_pr` is the one to use: precision is the share of predicted line length within r of any ground-truth line, recall is the share of ground-truth line length within r of any prediction, computed per image and then averaged over images. Nothing is paired, so a prediction covering one leg of an L-shaped hedge counts as partly right and a hedge stored as several overlapping ground-truth lines is not penalised. `matched_pr` is the chamfer + Hungarian variant, kept only to reproduce why it was rejected (F1 0.41 against 0.64 on the same predictions). Also holds `densify`, `merge_close_polylines`, `straightness` and `meters_to_px` (PDOK crops are 0.25 m per pixel, so 10 m is 40 px).

### Scripts

#### Data

- `scripts/data/build_lidar_patches.py`: cuts the AHN4 lidar metrics into one 25x25 cell patch per crop (250 m crop, 10 m cell), for the six metrics that separate hedge from tree row best. Writes a single `lidar_patches.npy` plus a stem index rather than one file per crop, because inodes are the tight resource on the cluster and 30,000 crops would cost 30,000 of them. Keyed by stem, so one file serves any split. About 26 min and 450 MB for the full dataset; `cfg.limit` builds a subset when only a check is needed.

- `scripts/data/build_lidar_training_dataset.py`: end-to-end local-raster pipeline. It creates images and labels with `hedge_seg.training_data`, postprocesses labels with `hedge_seg.label_postprocess`, computes DINOv3 embeddings, and packs embeddings with labels using `hedge_seg.embeddings_and_pack`.

- `scripts/data/build_pdok_wms_dataset.py`: entry point for the PDOK WMS data-generation workflow in `hedge_seg.pdok_training_data`. Asks for the fixed layer `2025_ortho25`, the photos of pdok_dataset2 and 3, not `Actueel_ortho25`, which changes with each new flight. A rebuild still does not give back the stored images byte for byte (`exps/probe_image_label_dates.py`), so the stored images are the dataset.

- `scripts/data/convert_pdok_polylines_to_yolo_bbox.py`: converts the generated JSON/polyline dataset into an Ultralytics detection dataset. Each polyline is converted to one normalized YOLO bounding box row.

- `scripts/data/convert_pdok_polylines_to_yolo_seg.py`: converts the generated JSON/polyline dataset into an Ultralytics segmentation dataset. Each polyline is buffered into a thin polygon mask and written as a normalized YOLO segmentation row.

- `scripts/data/convert_pdok_polylines_to_semseg.py`: converts the generated JSON/polyline dataset into a semantic-segmentation dataset (`images/`, `masks/`, `centerlines/` train/val folders). Each polyline is buffered into a foreground mask and its raw skeleton is kept separately as the centerline target; used by `train_semseg_unet_resnet18.py`.

- `scripts/data/convert_pdok_polylines_to_detr_polyline.py`: converts the generated JSON/polyline dataset into per-image polyline NPZs for `train_detr_unet_polyline.py`, decoupled from any backbone/embeddings. Cleans labels (opens closed rings, drops polylines under `min_length_px`) and splits crops into `polylines/{train,val}/` by geographic blocks on `center_world`, so overlapping crops never land on both sides of the split (see `hedge_seg.utils.geographic_overlap_stats`/`verify_no_split_overlap`, used by `exps/quantify_geographic_crop_overlap_pdok_dataset3.py` and `exps/dataset_stats.py`). With `treeline_shp` set it also clips the Top10NL tree row layer (`bomenrij`) to each crop and stores it as class 1 next to the hedges (class 0), reusing `make_polylines_for_chip` and `lines_in_bbox`, the same functions that produced the hedge labels, so the two classes cannot end up in different pixel conventions. No imagery is regenerated: the crops are the existing hedge-centred ones. `val_stems_file` takes the split from an existing run's val stem list instead of recomputing it, which keeps a new dataset, for example one with other labels, scoreable against that run. `_clean_polylines` holds the ring/length/resample pass so every class goes through it identically.

#### Training

All training scripts read datasets produced by the `scripts/data/*` workflows above and write TensorBoard logs plus best-checkpoint `.pt` files (except the Ultralytics ones, which use Ultralytics' own trainer/logging).

- `scripts/train_neg_pos_classifier.py`: trains a small linear-probe MLP (`LinearProbe`/`MLP`) on packed DINOv3 patch embeddings to classify chip-level embedding files as positive (contains hedge, filename has "pos") vs. negative.
- `scripts/train_detect_ultralytics.py`: trains a YOLO (Ultralytics) object-detection model on the YOLO bbox dataset from `convert_pdok_polylines_to_yolo_bbox.py`; also contains a commented-out inference/visualization snippet using `hedge_seg.visualization.draw_yolo_bounding_box_on_image`.
- `scripts/train_seg_ultralytics.py`: trains a YOLO-seg (Ultralytics) instance-segmentation model on the YOLO segmentation dataset from `convert_pdok_polylines_to_yolo_seg.py`, initialized from the detection checkpoint produced by `train_detect_ultralytics.py`.
- `scripts/train_semseg_unet_resnet18.py`: trains a ResNet18-UNet for binary semantic segmentation (hedge vs. background) plus a centerline channel, directly from RGB aerial images on the dataset produced by `convert_pdok_polylines_to_semseg.py`.
- `scripts/train_detr_hf.py`: fine-tunes a Hugging Face `AutoModelForObjectDetection` (DETR) on a COCO-style bounding-box dataset built from the chip images/labels; adapted from the HF "fine-tuning DETR" cookbook. Independent of the custom DETR implementations below (uses `transformers.Trainer` instead of a hand-written loop). The only user of `albumentations`, so it needs the `hf` extra: `pip install -e ".[hf]"`.
- `scripts/train_detr_dino.py`: trains a from-scratch DETR-style encoder/decoder (`DetrFromEmbeddings`) with a Hungarian matcher and class/bbox/GIoU losses, operating on precomputed DINOv3 patch embeddings + bounding boxes (via `DetrEmbDataset`) rather than raw images.
- `scripts/train_detr_dino_polyline.py`: extends the DINO-embedding DETR to predict whole polylines (`DetrPolylineFromEmbeddings`, `pred_polylines` of shape `(Q, K, 2)`) instead of boxes, with a `HungarianMatcherPolyline`/`DetrPolylineCriterion`, and adds an optional second-stage diffusion refinement head (`DetrWithDiffusion` / `GaussianPolylineDiffusion`) trained in stages (`cfg.stage`: 1 = DETR polyline, 2 = diffusion only, 3 = joint).
- `scripts/train_detr_dino_polyline_rel.py`: same DETR+diffusion staged design as `train_detr_dino_polyline.py`, but the polyline head first predicts a per-query bounding box (`pred_boxes`, cxcywh) and then predicts each polyline point as a bounded offset relative to that box (`cxcy + tanh(offset) * wh`) rather than regressing absolute point coordinates directly; adds extra matching/loss terms (`poly_cost`, `loss_smooth`, `loss_len`, `loss_dir`, `loss_card`, aux losses).
- `scripts/train_detr_maptr_polyline.py`: successor to `train_detr_dino_polyline_rel.py` (which it supersedes). Uses a MapTR-style query layout (per-polyline instance embedding combined with per-point embeddings, `num_polylines x num_points` queries) with iterative reference-point refinement in the decoder (global attention, not deformable), and drops the diffusion stage. Geometric losses (bbox/gIoU/length/direction/smoothness/cardinality) are optional and off by default (only class + ordered point L1 + aux). The data comes from `scripts/data/build_lidar_training_dataset.py` (precomputed DINOv3 embedding + polyline NPZ, no raw images). A single `main(cfg)` supports `cfg.mode="train"` and `cfg.mode="infer"`.
- `scripts/train_detr_unet_polyline.py`: the polyline model of every current run, successor to `train_detr_maptr_polyline.py`. One `main(cfg)` with `cfg.mode` set to `"train"`, `"infer"` or `"preview"`; preview draws the exact model input with the labels on top.
  - Model: the MapTR-style head of `train_detr_maptr_polyline.py` on the ResNet18-UNet from the `train_semseg_unet_resnet18.py` checkpoint. `freeze_backbone` keeps it fixed, with BatchNorm in eval mode; when it trains, `backbone_lr` gives it its own learning rate (`build_optimizer`).
  - Data: images from `scripts/data/build_pdok_wms_dataset.py`, polyline NPZs from `scripts/data/convert_pdok_polylines_to_detr_polyline.py`, read from `train_polyline_dir` and `val_polyline_dir`, so the split is the converter's. Images are padded from 1000 to 1024 px and points are normalised by the padded size. Flip and rot90 augmentation, on train only.
  - Checks: `DetrPolylineImageDataset` lists the image folder once and names every crop that has no image or no lidar patch, because two downloads of a dataset need not hold the same crops (`exps/probe_dataset_stem_mismatch.py`). It also stops on a label at or above `num_classes`, which `cls_loss="ce"` would otherwise train as background without any error.
  - Lidar, when `cfg.lidar_path` is set: one patch per crop from `scripts/data/build_lidar_patches.py`, augmented with the image and placed on the token grid at `cfg.lidar_stride`, which must equal the feature stride (asserted in `main`). An extra channel marks where the laser found any vegetation. `LidarTokenEncoder` (BatchNorm, then two 1x1 convolutions whose last layer starts at zero) adds it to the image tokens. `lidar_path=None` builds no branch at all, so old checkpoints still load.
  - Class head, `cfg.cls_loss`: `"ce"` is a softmax with a no-object column (all runs but exp 3), `"focal"` one sigmoid per class (exp 3). A checkpoint only runs with the setting it was trained with.
  - `cfg.merge_classes=True` turns every label into class 0, so the tree row dataset trains one woody line class (exp 10).
  - `infer` writes to `<infer_out_dir>/<ckpt stem>_<split dir name>_t<threshold>/`: `polylines/` with the predictions and `gt/` with links to the labels of exactly those crops, so runs never overwrite each other.

#### Inference

- `scripts/show_polyline_results.py`: shows inference runs of `train_detr_unet_polyline.py` on the image. List run directories in its cfg block. It draws one figure of the labels and one per run, all on the same crops, so the panels can be compared square by square. Each run directory already holds its own labels. `score_thresh` keeps only lines scoring at or above it, so a run inferred at 0.05 can be viewed at any higher threshold without inferring again. Lines are coloured by class; `class_id` shows one class. `gt_only` draws only the labels, and `ids` shows chosen crops instead of random ones. Press `n` for the next page of crops, `q` to stop. `save` writes each figure to `save_dir` as `<model>_p<page>_<tag>_<run>_<split>_t<thresh>.png`, with the page counting from 1, for example `detr_unet_polyline_p1_10_val_t.90.png`.
- `scripts/infer_detr_dino_polyline.py`: loads a checkpoint from `train_detr_dino_polyline.py` (plain `model` or `model_with_diffusion`) and runs/visualizes polyline predictions on precomputed embeddings.
- `scripts/infer_detr_dino_polyline_rel.py`: same as above for checkpoints from `train_detr_dino_polyline_rel.py` (box + relative-offset polyline head).

### Experiments

One-off probes are named `probe_*.py`. Each records what it found in its top docstring, so the answer does not have to be found again by running it. This list says only what each one asks.

Scoring:

- `exps/probe_polyline_pr.py`: scores an inference run of `train_detr_unet_polyline.py`: precision and recall by length at 5, 10 and 15 m, by crowding, over a sweep of the score cut-off, per class (`cfg.classes`) or with all lines together (`None`, the woody line score). Also the checks that shaped the metric: the rejected one-to-one pairing, label merging, campsite exclusion, and straightness and length against the labels. The source of the results table in `docs/status.md`.
- `exps/probe_f1_gap_bootstrap.py`: how much an F1 gap between two runs moves only because of which crops are in the val set.
- `exps/probe_recall_null_model.py`: how much of the recall is luck from drawing many lines. It scores each crop's lines against another crop's labels.
- `exps/probe_score_quality.py`: what the confidence score ranks by, correctness or straightness.
- `exps/probe_worst_crops.py`: writes id lists of the worst false positives, the worst misses and the probable missing labels, ranked by length, ready for the `ids` field of `scripts/show_polyline_results.py`.
- `exps/probe_backbone_trained.py`: did a run really train its backbone? Compares the backbone weights in a checkpoint with the pretrained ones; `requires_grad` cannot tell, because a checkpoint does not save it.

Labels:

- `exps/probe_treeline_overlap.py`: how much of the wrongly drawn length lies on a Top10NL tree row.
- `exps/probe_recreation_crops.py`: finds the crops inside a campsite or holiday park (Top10NL `functioneel_gebied_vlak`) and writes their stem list, to test an exclusion at scoring time.
- `exps/probe_image_label_dates.py`: which year the stored crops and the labels come from.
- `exps/probe_tree_dataset_stats.py`: what adding tree rows did to the dataset: lines per crop, the busiest crop, and whether the hedge labels stayed the same.
- `exps/probe_tree_class_labels.py`: do the two classes of the tree row dataset sit on the layers they came from?
- `exps/probe_hedge_tree_confusion.py`: does a two-class model find the hedges but call them tree rows? Tree row recall is counted only on crops that have a tree row.

Lidar:

- `exps/probe_lidar_hedge_vs_tree.py`: how well the 25 AHN4 metrics separate hedge features from tree row features, on held-out 5 km blocks.
- `exps/probe_lidar_nodata_vs_hedge.py`: what a missing value in the AHN4 metrics means, from how often it occurs under hedges and elsewhere.
- `exps/probe_lidar_crop_alignment.py`: does a lidar patch land on the same ground as its crop, before and after augmentation? Shift test and augmentation test, plus PNGs and `crop_footprints.geojson` for QGIS.
- `exps/probe_lidar_branch_absent.py`: does `lidar_path=None` really remove the lidar branch and change nothing else?

Data and split:

- `exps/dataset_stats.py`: label counts before and after the converter: polylines, closed rings, lines under 10 m.
- `exps/quantify_geographic_crop_overlap_pdok_dataset3.py`: how much the crops overlap, and how much a random split would leak (`hedge_seg.utils.geographic_overlap_stats`). The evidence for the block split.
- `exps/probe_split_gap.py`: how far each val crop is from the nearest train crop, and how many val blocks also hold train crops.
- `exps/probe_split_reproduces.py`: does `spatial_train_val_split` still give the stored split?
- `exps/probe_dataset_stem_mismatch.py`: does a polyline dataset name only crops this machine has, each with an image and a lidar patch? Run it on both machines before moving a dataset.
- `exps/probe_failed_downloads.py`: which stored crops are failed WMS downloads, found by file size.
- `exps/probe_laptop_matches_cluster.py`: does the laptop copy of the datasets and runs match the cluster, file by file? The copy steps are in `CLAUDE.md`.

Code and machines:

- `exps/smoke_test_train_detr_unet_polyline.py`: fast end-to-end check of `train_detr_unet_polyline.py`, a few steps on a handful of crops through the switch from eval back to train, on the exp 10 data. Run it before a cluster job.
- `exps/probe_refactor_equivalence.py`: did a refactor change the model, or only the code? Runs the old and the new code on CPU with the same weights and compares the tensors; the old script is imported from a git revision.
- `exps/probe_batch_size.py`: the largest batch that fits on the current GPU, and the time per image, from real training steps. `FROZEN=False` measures a run that trains the backbone.
- `exps/probe_model_size.py`: parameter count of the exp 9 model, per part.
- `exps/probe_osm_user_agent.py`: why the OSM chips came out as "Access blocked" images.

Older helpers, from before the DETR work:

- `exps/data.py`: raster and vector helpers (load a GeoTIFF, clip a raster to a bbox, rasterize hedge polylines) shared by the training-data workflows.
- `exps/basic_data.py`: early scratch helpers for the raw hedge shapefile and rasters, superseded by `exps/data.py`.
- `exps/pdok_wmts_training_data.py`: experimental WMTS workflow. Fetches PDOK WMTS tiles, caches them, builds a local GeoTIFF for a bbox, and can make training samples from that cache.
- `exps/download_pdok_aerial_images.py`, `exps/download_dutch_satellite_data.py`: one-off downloaders for PDOK aerial crops and Dutch satellite imagery.
- `exps/parametric_line.py`, `exps/pidinet_edge_detector.py`: from the initial investigation. Parametric line fitting and the PiDiNet edge detector as conventional baselines.

### Cluster

- `slurm/snellius_<model>.sh`: one submission script per model, copied to `~/exps/hedge/<model>/<n>.sh` and edited per run (see `CLAUDE.md`). Each prints the git hash, `hedge_seg/paths.py`, the data scripts and the whole training script before starting, so the `.out` log alone documents what ran.
