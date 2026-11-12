# Status

Where the project stands now. The details of each run are in
`docs/experiments_log.md`, the reasons behind each decision in
`docs/lesson_learned.md`.

## Best models

- Hedges: exp 9, `9.pt`, at score cut-off 0.90. Hedge F1 0.721 at 10 m.
- Woody lines, hedges and tree rows as one class: exp 10, `10.pt`, at score
  cut-off 0.90. Woody line F1 0.722 at 10 m.

A new run has to beat these by 0.02 F1 to count. A new random start alone
moves F1 by about 0.01 (see "How big a gap has to be before it counts" in
`docs/lesson_learned.md`).

The limit is perception: the model does not draw enough correct lines, and it
is worst on crowded crops. Exp 2 finds 0.731 of the hedge length on crops with
one labelled line and 0.372 on crops with seven or more (recall at 10 m,
t=0.95, `exps/probe_polyline_pr.py`).

## How the numbers are measured

- Precision: the share of drawn line length that lies within r of a labelled
  line.
- Recall: the share of labelled line length that has a drawn line within r.
- F1: one number from the two. It is high only when both are high.
- Each is computed per crop, then averaged over crops.
- r is the tolerance, 5, 10 or 15 m. 10 m is the headline.
- t is the score cut-off: lines scoring below it are dropped. Each run is
  scored at its own best t, found by a sweep.
- Always on the same 3,098 val crops. The split is by location, so these
  crops are ground the model has not seen. That makes the numbers lower than a
  random split would. They are the real baseline.

`buffered_length_pr` in `hedge_seg/metrics.py`, run by
`exps/probe_polyline_pr.py`.

## Results

Hedges, scored against the hedge labels only. P and R at 10 m.

| exp | what is different from exp 2  | checkpoint  | t    | P    | R    | F1 5 m | F1 10 m  | F1 15 m |
| --- | ----------------------------- | ----------- | ---- | ---- | ---- | ------ | -------- | ------- |
| 1   | 5,000 train crops, 150 epochs | `1_150.pt`  | 0.95 | .702 | .589 | .489   | .640     | .717    |
| 2   | nothing                       | `best_2.pt` | 0.90 | .701 | .696 | .553   | .699     | .767    |
| 3   | focal sigmoid score head      | `3.pt`      | 0.40 | .779 | .578 |        | .664     |         |
| 4   | tree rows as class 2, lidar   | `best_4.pt` | 0.90 | .693 | .703 | .533   | .698     | .773    |
| 5   | tree rows as class 2          | `best_5.pt` | 0.90 | .692 | .686 | .536   | .689     | .761    |
| 6   | lidar                         | `best_6.pt` | 0.90 | .698 | .709 | .545   | .703     | .774    |
| 7   | new random start, seed 43     | `best_7.pt` | 0.90 | .687 | .694 | .540   | .691     | .761    |
| 8   | 90 epochs instead of 45       | `8.pt`      | 0.90 | .709 | .715 | .569   | .712     | .775    |
| 9   | backbone trained, lr 1e-5     | `9.pt`      | 0.90 | .710 | .733 | .581   | **.721** | .779    |

For exp 3 only the 10 m score is recorded.

Woody lines, every drawn line against both label layers, at 10 m:

| exp | model                               | checkpoint  | t    | P    | R    | F1       |
| --- | ----------------------------------- | ----------- | ---- | ---- | ---- | -------- |
| 10  | one woody class, exp 9 recipe       | `10.pt`     | 0.90 | .683 | .765 | **.722** |
| 9   | hedges only                         | `9.pt`      | 0.70 | .690 | .663 | .676     |
| 5   | two classes pooled, frozen backbone | `best_5.pt` | 0.90 | .678 | .632 | .654     |

All from `exps/probe_polyline_pr.py`. The tree row class scores of exps 4 and
5 are left out on purpose, because they are too high (see TODO 2).

## Done

- Phase A, data and split: labels cleaned (closed loops opened, lines under
  10 m dropped), geographic split, train 26,902 / val 3,098.
- Phase B: the first run that draws real hedges on unseen ground (exp 1).
  Until then, checkpoints were judged by the eval loss, the training loss
  computed on the val crops, and by counting lines per crop. Neither says
  whether the lines lie on hedges, so Phase B added the length-based metric
  above. It overturned the eval loss at once: the loss preferred epoch 65 of
  exp 1, the metric epoch 150 (F1 at 10 m: 0.614 at epoch 65, 0.640 at epoch
  150). Checkpoints are ranked by F1 since. Training still saves
  `best_<n>.pt` at the lowest eval loss, so score `<n>.pt` too.
- Phase C:
  - The score head is closed. Exp 3 lost, and the null model showed that
    half of the recall it chased was luck.
  - Tree rows and lidar give no hedge gain (exps 4 to 6).
  - A new random start moves F1 by about 0.01 (exp 7). Training twice as long
    helps a little (exp 8).
  - Training the backbone is the first clear gain (exp 9).
  - One woody class gives the best woody line map (exp 10).

## TODO

In the order suggested:

1. **Look at the strange things in the figures.** In
   `/home/fatemeh/Downloads/hedge/screenshots/`, `exp2_8_9.png` shows exp 2
   with the lowest val loss, but by eye exp 9 is better, and its last
   checkpoint sometimes beats its best one. `strange_polylines1.png` (left
   `best_9_val_cluster_t0.05`, right GT) and `strange_polylines2.png` (left
   `9_val_cluster_t0.05`, middle `best_9_val_cluster_t0.05`, right GT) mark
   with arrows lines that lie on no hedgerow. Also check the short side of an
   L-shaped hedge being left out: if L shapes are rare, storing them as two
   lines may be better. Draw more with `scripts/show_polyline_results.py`.
   This may change what comes next.

1. **Fix the tree row score.** `exps/probe_polyline_pr.py` gives a crop with
   no tree row, and none drawn, a perfect 1.0, and 1,224 of the 3,098 val
   crops have none. On the 1,874 that do, tree recall at 10 m is 0.280 for
   exp 5 and 0.425 for exp 4 (`exps/probe_hedge_tree_confusion.py`). Score a
   class only on crops where it appears, then correct the tree row numbers in
   `docs/experiments_log.md` (exps 4 and 5), in the `exps/probe_polyline_pr.py`
   docstring, and in the talk (the exp 4 slide says the score is being fixed).
   Hedge and woody line scores are not affected: every val crop but one has a
   hedge.

1. **Labels from Top10NL2025 instead of 2023.** The photos are 2025, the
   labels 2023. Build 2025 labels for the same crops, pinning the split with
   `val_stems_file`, and score the exp 2 and exp 9 predictions against them
   before training anything. If they score clearly higher, the old labels
   hold the number down. Both shapefiles are in
   `/home/fatemeh/Downloads/hedge/Top10NL2025`. Keep the label year apart from
   any other change, or the run cannot be read.

1. **Null model per crowding bucket for exp 10.** Its woody recall barely
   falls on crowded crops, which would say part of the crowding problem is
   the label split. It also draws more lines there, so check before believing
   it.

1. **Remake the talk figures with exp 9.** They show exp 2 (`best_2`), while
   the headline numbers are exp 9. The exp 9 inference runs exist, so this
   needs only `scripts/show_polyline_results.py`. Steps in
   `presentation/README.md`.

1. **Next GPU runs**, ideas, not decided:

   - Finer features. `feature_stage="up3"` is stride 16, so a 3 m hedge,
     12 px, is under one feature cell. `FEATURE_STAGES` only has `enc4`
     (stride 32) and `up3` (stride 16), so this means adding the UNet decoder
     stage at stride 8. Batch 16 uses 5.8 of 40 GB, and stride 8 is 4x the
     tokens. The most direct attack on the limit.
   - Exp 9 again with another seed, to confirm its +0.022.
   - Exp 9 for longer. Exp 8 says longer adds only a little (+0.013).

1. **Why the stored crops differ from a fresh download.** Not urgent. The
   stored pdok_dataset3 crops differ from a fresh `2025_ortho25` download by
   5.6 to 13 mean pixel value, while pdok_dataset2 `pos_000000`, the same
   place, matched it exactly (`exps/probe_image_label_dates.py`). Look at one
   crop side by side: colour, sharpness or a small shift? Which pdok_dataset2
   copy matched is not recorded. The laptop and cluster copies differ: the
   cluster one is in `pdok_dataset2`, the laptop one in
   `pdok_dataset_semseg2/images/train/pos_000000.png`.

1. **A script for the short Top10NL features.** `docs/lesson_learned.md` says
   18.0% of heg features (11,208 of 62,415) and 16.9% of bomenrij features
   (45,028 of 266,783) are shorter than the nominal 100 m, but the code that
   measured it was not kept. Write `exps/probe_top10nl_short_features.py`:
   `geometry.length` on both shapefiles in EPSG:28992, where the unit is
   metres. It should also give the feature counts, 62,415 and 266,783, that
   `docs/descriptions.md` quotes. Small, no GPU.

For the paper:

- **Compare with the older mask and box models on the same measure.** The
  numbers of YOLO boxes, YOLO-seg and the UNet come from each tool's own score
  and a random split, so they cannot be set beside the line model. The UNet
  backbone `semseg_unet/4/best_4.pt` never saw the 3,098 val crops, because
  the split keeps them away from its training crops. Thin its masks to centre
  lines and score them with `buffered_length_pr`. No training is needed.

Then, results-driven:

- Image resolution. Everything so far is 25 cm. Downsampling to 50 cm or 1 m
  costs nothing to try and would say how much of the result depends on
  resolution, which matters for applying this outside PDOK coverage. Not urgent.
- Border filtering. Polylines are clipped to the crop bounds, so a hedge
  crossing the edge becomes a truncated line the model is asked to predict
  exactly. Nothing drops or down-weights them. Still untested rather than
  dismissed.
- LiDAR as the only input, with no aerial image.
- Another backbone or embedding: DINOv3 sat, Google Satellite embeddings,
  SkySense.

Checkpoints: the epoch snapshots `<n>_<epoch>.pt` are not needed and are not
kept (see Dropped). Training no longer writes them: `save_every` in
`scripts/train_detr_unet_polyline.py` is set above `n_epochs`. `1_150.pt` is 
the exp 1 result, not a spare snapshot, so keep it.

## Dropped, with the reason

- The score head, after exp 3. See "Most of the recall at a low threshold is
  clutter" in `docs/lesson_learned.md`.
- Bends as a separate work item. They are the threshold's doing, not the
  geometry's.
- A date-matched rebuild on `2016_ortho25`. 2016 imagery matches nothing here;
  the labels are 2022 and the crops are 2025.
- Scoring runs epoch by epoch. Exp 8 answers whether longer training helps
  (+0.013), and comparing `best_<n>.pt` with `<n>.pt` answers whether eval loss
  picks the right checkpoint (it does not). The probe is in git history:
  `git log -- exps/probe_f1_by_epoch.py`.
