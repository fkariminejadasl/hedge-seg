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

- 3 scored, 2026-08-11. **Lost.** Best F1 at 10 m 0.664 against exp 2's 0.699.
  Both checkpoints, all 3,098 val crops, `exps/probe_polyline_pr.py`:

  | run | best t | P | R | F1 @10m |
  |---|---|---|---|---|
  | exp 2 `best_2.pt` | 0.90 | .701 | .696 | **.699** |
  | exp 3 `best_3.pt` | 0.40 | .781 | .572 | .660 |
  | exp 3 `3.pt` (ep 45) | 0.40 | .779 | .578 | .664 |

  The last epoch beats the best-eval-loss checkpoint again, by 0.004. Small,
  but the third time the eval loss has ranked wrong.

  Perception unchanged: recall at t=0.05 is 0.842 against exp 2's 0.840. The
  head just got more reluctant. At its own optimum exp 3 draws 1.6 lines per
  image against exp 2's 3.2, with GT at 2.14. Precision rose to 0.78 and recall
  fell to 0.57.

  `exps/probe_score_quality.py`: AUC for correct against wrong 0.742 (exp 2
  0.767), inside the bent group 0.648 (exp 2 0.618). So the intended effect
  appeared at a third of the size and cost more elsewhere. Scores clumped near
  0.2 and above 0.98 rather than spreading; at >0.99 straightness, correct and
  wrong sit at 0.983 and 0.979.

  Geometry unchanged: predicted straightness 0.946 with bent<0.85 at 0.110,
  against exp 2's 0.942 / 0.101 and GT 0.909 / 0.220.

  Job 25396297, gpu_a100, git 34ea159, 11:28:40 for 45 epochs (15:05/epoch).
  Train 1.4102 / eval 1.5658 at epoch 45, best eval 1.5624 at epoch 40. These
  losses are not comparable to exp 2's, different normalisation.

- recall null model, 2026-08-11, `exps/probe_recall_null_model.py`. The reason
  exp 3 was worth running and the reason not to run a fourth score-head variant.
  Scoring each crop's predictions against a different crop's labels:

  | t | pred/img | real R | null R | skill |
  |---|---|---|---|---|
  | 0.05 | 12.6 | .840 | .379 | .461 |
  | 0.90 | 3.2 | .696 | .266 | .431 |

  Half of "recall 0.84 at t=0.05" is luck from drawing 13 lines per image.
  Skill peaks at 0.471 (t=0.40) and is 0.431 at the operating point, so a
  perfect score head is worth about 0.04 of recall. The score head is closed.

- 3 (cluster, run 2026-08-10, score-head A/B against exp 2): `cls_loss="focal"`, one
  sigmoid per class with focal loss instead of softmax over {hedge, no-object},
  focal_alpha=0.25, focal_gamma=2.0, class_cost and loss_ce raised 1.0 -> 2.0 to
  match MapTR's recipe (focal is normalized by matched targets, not by query
  count, so the weights are not on the old scale). Everything else identical to
  exp 2: full 26,902 train crops, 45 epochs, same val stems. Not warm started,
  the class head changed shape. ETA about 11 h on an A100, `--time=16:00:00`.

  Why focal and not eos_coef: `exps/probe_score_quality.py` on exp 2 shows the
  score ranks by straightness, not correctness. Correct bent lines have median
  score 0.453, wrong straight ones 0.957, and t=0.95 keeps 9% of the former
  against 74% of the latter. AUC for correct against wrong is 0.767 overall but
  0.618 inside the bent group. eos_coef scales the no-object column uniformly,
  so it moves all scores together and cannot fix a ranking problem.

  Score at t=0.05 and re-sweep the threshold. A focal score is not on the same
  scale as the softmax one, so exp 2's optimum of 0.90 carries no information.
  Read it with `exps/probe_polyline_pr.py` and `exps/probe_score_quality.py`;
  the second is the one that says whether the head did its job.

  `loss_ce` and therefore `loss_total` are also on a new scale: focal is summed
  and divided by the matched-target count, cross-entropy was a mean over all
  queries. Do not compare exp 3's loss curve to exp 2's, only to itself.

- probes, 2026-08-10, no training:
  - `exps/probe_score_quality.py`: the score/straightness result above.
  - `exps/probe_lidar_hedge_vs_tree.py`: AHN4 p95 height along 2,000 features
    per layer. heg median 8.0 m, bomenrij 12.2 m, best single cut 9.0 m at
    balanced accuracy 0.638. Height alone does not separate the two classes.
  - Label dates, corrected 2026-08-11: the gap is three years, not ten.
    `Actueel_ortho25` is bit-identical to `2025_ortho25` (0.00 mean pixel
    difference, against 29.51 for `2022_ortho25`), and Top10NL2023 was revised
    on 2022 photos. `bronactual` sitting in 2014-2015 for 75.3% of features
    records the source of each feature's last edit, not the last check. Fix by
    using Top10NL2025 labels on the existing crops.

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
  150 ep. 2:40/ep, 7 h total.

  Losses: train 1.12 at the end, eval down to ~1.31 near epoch 65 then up to
  1.41 by 150. Overfits after ~65 by the loss.

  First run whose predictions sit on real hedgerows in images the model never
  saw. Lines follow tree rows and field boundaries. DINOv3 runs never did this.

  Scores are squashed high, deciles on 400 crops 0.775 / 0.909 / 0.961 / 0.983 /
  0.990 / 0.997, so infer_score_thresh went 0.5 -> 0.95.

  Checkpoints on the cluster val split (32 crops, t=0.95, GT 2.44 lines/img):

  | checkpoint | pred/img | abs err/img | 0 pred | over by 3+ | worst |
  |---|---|---|---|---|---|
  | best_1.pt (ep 65) | 1.75 | 1.06 | 0 | 0 | -5 |
  | 1_150.pt (ep 150) | 2.41 | 1.59 | 2 | 2 | +13 |

  The mean flatters 1_150. Per image it predicts nothing on pos_026918 and
  pos_028651 and 16 lines on pos_024293, which has 3, and those cancel. Counts
  cannot rank the two; the buffered metric below can.

- exp 1 scored, 2026-07-28. `exps/probe_polyline_pr.py`, buffered length,
  all 3,098 cluster val crops, per image then averaged, t=0.95.

  | checkpoint | 5 m | 10 m | 15 m |
  |---|---|---|---|
  | best_1.pt | .57/.39 F1 .466 | .75/.52 F1 .614 | .83/.59 F1 .688 |
  | 1_150.pt | .54/.45 F1 .489 | .70/.59 F1 .640 | .78/.67 F1 .717 |

  **1_150.pt wins at every buffer.** The eval loss turns at epoch 65 and the
  detector keeps improving to 150, so do not early-stop on it.

  1_150.pt at 10 m by GT line count. Precision flat, recall falls with density:

  | GT lines | crops | share of GT lines | P | R | F1 |
  |---|---|---|---|---|---|
  | 1 | 1438 | 22% | .70 | .71 | .704 |
  | 2-3 | 1220 | 43% | .71 | .51 | .596 |
  | 4-6 | 363 | 25% | .68 | .42 | .520 |
  | 7+ | 76 | 11% | .62 | .38 | .473 |

  Threshold sweep, 1_150.pt at 10 m, from the t0.05 inference run:

  | t | pred/img | P | R | F1 |
  |---|---|---|---|---|
  | 0.05 | 8.42 | .41 | .75 | .530 |
  | 0.80 | 4.70 | .54 | .70 | .612 |
  | 0.90 | 3.69 | .60 | .66 | .632 |
  | 0.95 | 2.26 | .70 | .59 | **.640** |
  | 0.98 | 0.83 | .86 | .35 | .494 |

  0.95 is already the optimum, so no change. But recall is 0.75 at t=0.05:
  the lines exist and the score cannot rank them. best_1.pt peaks at 0.90
  (F1 .634), still under 1_150.pt.

  Dead ends, all measured on the same predictions:

  - Chamfer + Hungarian instead of buffered length: F1 .411 for 1_150.pt at
    10 m against .640. Rejected, reason in docs/lesson_learned.md.
  - Merging GT lines within 10 m: removes 0.8% of GT lines, F1 .640 -> .640.
  - Excluding the 315 campsite crops (`exps/probe_recreation_crops.py`,
    10.2% of val): F1 .640 -> .652. Campsite crops alone score .542. Real but
    too small a share to be worth a dataset change.
  - Straightness pred .893 against GT .909, median length 116 m against 120 m.
    Predictions bend and stretch like the labels, so the "only straight lines"
    impression from a 16-crop figure was a sampling artifact.

  By eye (screenshots detr_unet_polyline_{1_gt,best_1,1_150}_*cluster_t.95.png):
  both accurate on simple single-hedge crops; both predict one line where
  several labelled lines meet at a junction (pos_009106, pos_013789,
  pos_027437); 1_150 falls apart on crowded crops. Label problems seen: hedges
  missing from GT, and one hedge split into several overlapping GT lines.

  polylines/val_cluster holds the 3,098 crops the cluster used for val, all
  found locally. Built with:

  ```
  ssh me "ls /projects/prjs1025/data/hedge/pdok_dataset3_polylines/polylines/val" \
    > /home/fatemeh/Downloads/hedge/cluster_val_stems.txt
  cd /home/fatemeh/Downloads/hedge/results/pdok_dataset3_polylines
  mkdir -p polylines/val_cluster
  while read f; do
    [ -f "polylines/val/$f" ] && ln -sfn "$(realpath polylines/val/$f)" "polylines/val_cluster/$f"
  done < /home/fatemeh/Downloads/hedge/cluster_val_stems.txt
  ```

- 1_laptop (2026-07-22, in progress): laptop RTX PRO 3000. Same as cluster 1 but
  laptop overrides: workers=4, eval_every=10, n_val_subset=1000 (leaky local
  split, so full val is not the honest number). ~7:30/ep, ~19 h total. Hit the
  forkserver DataLoader hang at epoch 11 (fixed, see lesson_learned.md), then
  relaunched. At epoch 10, eval poly 0.075 below train poly 0.080: generalizing,
  not memorizing (unlike the 8-image runs). Use it to watch the curve, not for a
  final number.

- inference at t=0.05 (2026-07-28, laptop): both exp 1 checkpoints re-inferred
  over the 3,098 val_cluster crops so the threshold could be swept downward.
  100 s per checkpoint, not the 30 min guessed from the training throughput:
  inference has no backward pass. Sweeping needs a low-threshold run, since a
  run saved at 0.95 can only be swept upward.

- 2 (cluster, Phase C data A/B, 2026-07-28): job 25000671 on gpu_a100, git
  601f93b. Full train split 26,902 instead of the 5,000 subset, 45 epochs,
  save_every=10, everything else identical to exp 1. Not warm started from
  exp 1. 14:26/epoch, 10:54:49 total. Same val stems as exp 1.

  Losses: train 1.1756, eval 1.2076 at epoch 45. **Eval loss fell at every
  single eval and best is the last epoch**. Exp 1 ended train 1.12 / eval 1.41, 
  so the train-eval gap went from 0.29 to 0.03. 5.4x the data removed the 
  overfitting completely, and the run was still improving when it stopped.

  | epoch | 5 | 15 | 25 | 35 | 45 |
  |---|---|---|---|---|---|
  | train | 1.4495 | 1.3076 | 1.2407 | 1.1949 | 1.1756 |
  | eval | 1.3990 | 1.2879 | 1.2455 | 1.2195 | 1.2076 |

  Buffered length against exp 1, both at t=0.95 on the same 3,098 crops:

  | buffer | exp 1 `1_150.pt` | exp 2 `best_2.pt` |
  |---|---|---|
  | 5 m | .535/.450 F1 .489 | .629/.484 F1 **.547** |
  | 10 m | .702/.589 F1 .640 | .783/.608 F1 **.685** |
  | 15 m | .778/.665 F1 .717 | .845/.674 F1 **.750** |

  More data bought mostly precision, +0.08 at 10 m against +0.02 recall, and
  the largest relative gain is at the tightest buffer (5 m F1 +0.058), so the
  lines also sit more accurately.

  **The optimal threshold moved from 0.95 to 0.90.** At 0.90 exp 2 gives
  P 0.701 / R 0.696, F1 **0.699** at 10 m, against 0.685 at 0.95. Re-sweep the
  threshold after any change to the model or the data; it is not a constant.
  At t=0.05 recall is 0.840, up from 0.752, so the model now puts a line within
  10 m of 84% of all GT hedge length.

  By GT line count at 10 m, t=0.95 (exp 1 -> exp 2). Precision rose everywhere,
  recall barely moved, so the dense-crop problem is not a data-volume problem:

  | GT lines | P | R |
  |---|---|---|
  | 1 | .701 -> .788 | .707 -> .731 |
  | 2-3 | .714 -> .786 | .512 -> .530 |
  | 4-6 | .682 -> .774 | .420 -> .435 |
  | 7+ | .619 -> .688 | .383 -> .372 |

  Campsite crops excluded: F1 .685 -> .696, same small effect as exp 1.

  Regression worth watching: predicted straightness rose to 0.942 against GT
  0.909, and the strongly bent share fell to 0.101 against GT 0.220. Exp 1
  matched the labels (0.893, 0.221). More data made the model draw straighter
  lines than the labels have, which is what L1 does under uncertainty about
  where a corner sits. Median length still matches, 120 m against 120 m.

- exp 2 error analysis, 2026-07-29. `exps/probe_worst_crops.py` writes id lists
  to `pdok_dataset3_polylines/`: `worst_fp_*.txt`, `worst_fn_*.txt`,
  `missing_labels_*.txt`, ranked by unmatched length at 10 m, t=0.95.

  Worst false positives are dominated by built-up and campsite crops:
  pos_007788 (1412 m unmatched), 016359, 014293, 025317, 019018, 029463.
  Worst false negatives are crops with ~1000 m of GT and almost nothing found:
  pos_016471 (recall .14), 010771 (.09), 029904 (.17), 028417 (.10).

  The missing-label rule (recall >= .8, precision <= .5, unmatched >= 100 m)
  flags 40 crops. By eye on 16 of them, most are correct: pos_025224, 008752,
  023818, 007358 and 025758 all show two or three clearly visible woody lines
  where the labels have one. pos_019018 is the exception, a built-up crop the
  model floods.

  `exps/probe_treeline_overlap.py` then explains a quarter of it. On 300 random
  val crops, 24.3% of unmatched predicted length is within 10 m of a Top10NL
  tree row (`bomenrij`), against 1.2% near another hedge. Sanity check: 100% of
  the training GT sits on the `heg` layer. On the 40 missing-label crops the
  figure is 21.3%, so this is a general effect, not a property of the worst
  crops.

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
