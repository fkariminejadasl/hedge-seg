# Experiment log

Raw notebook of individual runs, terse style. Held to a lower bar than
docs/lesson_learned.md (curated lessons) and docs/descriptions.md (per-script
overview). Merged from the older docs/descriptions.txt notes.

Config shorthand for detr_polyline rows:
epochs, n_polyline (num_queries), n_points (num_points), eos_coef,
dino grid (14 or 16), then extras. grid size (14,14) or (16,16).

## Reference

Timing: 2 min per epoch for 17,381 images on an A100 40GB.

Parameter counts:
- ResNet18: 11.7 M
- ResNet18-UNet: 14.4 M
- Polyline total: 7.9 M (encoder 3.2 M, decoder 4.2 M, rest 0.6 M)
- Diffusion: 6.1 M

MapTR image/feature/BEV grid sizes:

| Model          | Resized input | Padded image | Feature grid | BEV grid  |
| -------------- | ------------: | -----------: | -----------: | --------: |
| MapTR-nano R18 |     320 x 180 |    320 x 192 |       10 x 6 |   80 x 40 |
| MapTR-tiny R50 |     800 x 450 |    800 x 480 |      25 x 15 | 200 x 100 |

## Overfitting (laptop), detr_polyline on DINO embeddings

experiment: epochs, n_polyline, n_points, eos_coef, dino grid, extras, dataset

- detr_polyline_3:  3000, 100, 10, eos=.064_256, dino_14, wd=1e-2, dropout=0.1, giou=1, test_64_256_tiny
- detr_polyline_4:  3000, 100, 10, eos=.5, dino_14, wd=1e-2, dropout=0.1, giou=1, test_64_256_tiny
- detr_polyline_5:  3000, 100, 20, eos=.1, dino_14, wd=1e-2, dropout=0.1, giou=1, test_256
- detr_polyline_6:  3000, 450, 20, eos=.1, dino_14, wd=1e-2, dropout=0.1, giou=1, test_256
- detr_polyline_7:  3000, 450, 20, eos=.5, dino_14, wd=1e-2, dropout=0.1, giou=1, test_256
- detr_polyline_8:  3000, 450, 20, eos=.5, dino_16, wd=1e-2, dropout=0.1, giou=1, test_256_dino256
- detr_polyline_9:  3000,  16, 20, eos=.5, dino_16, wd=0,    dropout=0,    giou=0, test_256_dino256
- detr_polyline_10: 3000, 100, 20, eos=.1, dino_16, wd=1e-2, dropout=0.01, giou=1, test_256_dino256

overfit 10 has all defaults except dropout. Dropout larger than 0.01 collapses.
(11, (10,2)) -> test_64_256_tiny. (11, (20,2)) -> test_256, test_256_dino256
(same but dino seq_length differs). Ranking: 3, 8 < 6 < 4, 7 < 5. loss 4 was
lower when only 2 polylines; with 11 it is higher.

## snellius detr_polyline (DINO embeddings)

- best_detr_polyline_1: 500, 100, 20, eos=.1, dino_14, wd=1e-2, dropout=0.1, giou=1, batch=1280, test_256 -> no collapse.
  Max 441 polylines in some images, image 256x256, DINOv3 resized to 224.
  Top-20 scores 0.75-0.96, detections not good but show some linear structure.
- best_detr_polyline_2: 500, 100, 10, eos=.1, dino_14, wd=1e-2, dropout=0.1, giou=1, batch=256, test_64_256 -> collapse.
  Max 82 polylines, image 64->256 upsampled. Top-20 scores ~0.13, all stacked in the center.
- best_detr_polyline_3: 1000, 128, 10, eos=.2, dino_16, wd=1e-2, dropout=0.1, giou=0, batch=1024, test_256_None -> collapse.
  Used original DINOv3 size (no resize). Same collapse.
- best_detr_polyline_4: 500, 100, 10, eos=.1, dino_16, wd=1e-2, dropout=0, giou=1, batch=1024, test_256_None -> no collapse.
  Not good, but line structure around the ground truth.
- best_detr_polyline_5: 500, 276, 10, eos=.1, dino_16, wd=1e-2, dropout=0, giou=1, batch=768, test_256_None.
  Max polylines, lowered batch to fit memory.
- best_detr_polyline_6: 500, 276, 10, eos=.1, dino_16, wd=1e-2, dropout=0, giou=1, batch=768, test_256_None, n_enc=2, n_dec=2.
  Smaller model (from 4/4).
- best_detr_polyline_7: 500, 276, 10, eos=.1, dino_16, wd=1e-2, dropout=0, giou=1, batch=512, test_256_None, aux loss.
  stage1 (above), stage2 (batch=512, lr=1e-4, 200 ep), stage3 (batch=256, lr=3e-5, 200 ep).
- best_detr_polyline_8: as 7, remove image contrast. Same as 7 (distribution good, not locations).

## detr_polyline_rel (DINO embeddings)

- 1: 10, 276. Slow convergence but result looks better than the non-relative one.

## detect_ultralytics (YOLO bbox, pdok_dataset_yolo, 1000x1000, 250m)

bbox 3900/903.
- 1: yolo26l, 800/200, batch=16, 2000 ep, imgsz=1000 -> best=176, stop=276, not great.
- 2: yolo26n, 800/200, batch=16, 2000 ep, imgsz=1000 -> better than 1.
- 3: from 2/last.pt, 24000/6000 (47h, 1 GPU, budget ran out at 559). Better.

## seg_ultralytics (YOLO-seg, 24000/6000)

- 1: batch=64, 3M params, yolo26n-seg last, epoch=1183 (1400 requested, 72h,
  4 GPU, stopped on time). Good result but not all detected. 3.65 min/epoch.
  For 1 epoch, batch=16 with yolo26m-seg (26M params) took 12.46 min.

## semseg_unet (ResNet18-UNet)

- 1: 1000 img (800/200), centerline_weight=.5, threshold=.5, 20 ep, batch=32, pdok_dataset_semseg (0:00:25/ep)
- 2: 1000 img, centerline_weight=0, threshold=.5, 20 ep, batch=32, pdok_dataset_semseg
- 3: 1000 img, centerline_weight=0, threshold=.1, 20 ep, batch=32, pdok_dataset_semseg
- 4: 5000 img (4000/1000), centerline_weight=0, threshold=.1, 50 ep, batch=32, pdok_dataset_semseg2 (0:01:40/ep). backbone for polyline runs.
- 5: 30000 img (24000/6000), centerline_weight=0, threshold=.1, 50 ep, batch=32, pdok_dataset_semseg3 (0:08:27/ep). checked only 1 epoch.

## detr_unet_polyline (ResNet18-UNet backbone, image input)

- unet1 / detr_unet_polyline_1: overfit one image (val = same image), 2000 ep,
  4 enc / 4 dec (00:18:00). Validated frozen semseg features support polyline
  regression, sub-meter memorization. best_*.pt at epoch 1722.
- unet2 / detr_unet_polyline_2: 10 img, 2000 ep, 1 enc / 4 dec, augmentation
  (00:42:00). Better than unet3 on val (one almost good).
- unet3 / detr_unet_polyline_3: 10 img, 2000 ep, 1 enc / 4 dec, no aug (00:42:00).
  unet2/3: train poly ~0.007 (~1.8 m), eval poly ~0.17 (~43 m); eval rises after
  ~100 ep (n=8 cannot generalize). Inspect small-data runs with the final
  checkpoint ({exp}.pt), not best_*.pt (best froze at epoch 101 / 27). With the
  final ckpt, train images match GT to 0.6-6.3 px.

- 1 (cluster, Phase B baseline, 2026-07-21): job 24799874 on gpu_a100.
  5000-image train subset, full val 3098, frozen backbone up3, num_polylines=60,
  augment on, eval_every=5, cosine schedule, 1 enc / 4 dec, batch=16, workers=8,
  150 ep. 2:40/ep. Interrupted by cluster maintenance; best_1.pt saved on the
  cluster, results checkable Monday. This is the honest baseline (val is the
  stricter split that also avoids the semseg backbone areas).

- 1_laptop (2026-07-22, in progress): laptop RTX PRO 3000. Same as cluster 1 but
  laptop overrides: workers=4, eval_every=10, n_val_subset=1000 (leaky local
  split, so full val is not the honest number). ~7:30/ep, ~19 h total. Hit the
  forkserver DataLoader hang at epoch 11 (fixed, see lesson_learned.md), then
  relaunched. At epoch 10, eval poly 0.075 below train poly 0.080: generalizing,
  not memorizing (unlike the 8-image runs). Use it to watch the curve, not for a
  final number.

## Phase A data conversion

- local (2026-07-20): pdok_dataset3 (30k) -> pdok_dataset3_polylines.
  min_length_px=40 (10 m), closed_eps_px=2, block_m=5000, val_fraction=0.2,
  seed=42. train=24,257 val=5,743 (19.1%), 103,432 polylines, 302 rings opened,
  0 val-train overlap. Only 10 local pdok_dataset2 labels, so avoid_label_dirs
  excluded just 36 crops.
- cluster (2026-07-21): same, but avoid_label_dirs sees all 5,000 pdok_dataset2
  labels. train=26,902 val=3,098 (10.3%), 103,414 polylines, 0 val-train overlap.
  12,155 crops (40%) sit near semseg areas, so avoidance halves the val set. Kept
  it: no val image was seen by the frozen backbone during semseg training.

## Batch size probe (2026-07-21)

exps/probe_batch_size.py, real train step per batch size.
- A100 40GB (job 24798721): 16 = 5.76 GB, 32 = 11.41 GB, 64 = 22.67 GB;
  0.354 GB/image; s/image flat past 16 (0.0278 -> 0.0262). Chose batch 16.
- RTX PRO 3000 12.3 GB: 4 = 1.55, 16 = 5.76, 32 = 11.41, 48 = OOM.
  GPU-only 0.0532 s/image (A100 0.0278, ~1.9x). Real epoch ~2.8x slower than
  A100 because the laptop is CPU-bound on data loading and matching.
