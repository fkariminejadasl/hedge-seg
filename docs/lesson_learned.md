## MapTR lessons

BEV resolution: MapTR-tiny uses a 30 cm BEV cell size over a (30 x 60) m area, giving a (100 x 200) BEV grid (`projects/configs/maptr/maptr_tiny_r50_24e.py`). MapTR-nano uses 75 cm cells, giving a (40 x 80) grid. 

The decoder uses self-attention among hierarchical instance and point queries, and deformable cross-attention from these queries to the BEV features.

A coarse BEV token may summarize several nearby polylines. The important factor is the spatial granularity of the BEV features. If downsampling removes the polylines’ relative positions, and that information is not retained in the feature channels, the decoder cannot reliably separate or localize them.

Having more polylines than BEV tokens is not inherently a problem because there is no one-token-per-polyline assignment. Each hierarchical point query can sample multiple BEV locations, and multiple queries can use overlapping BEV features. The polylines remain distinguishable only if those features preserve enough spatial information about them.


## Why extra losses cause a zigzag (and why only poly + class overfits)

The problem: the model predicts 20 points per line, but the points come out in the
wrong order, so the line looks like a zigzag around the right place.

Main idea: only the ordered point loss (poly L1) knows which point is point 0, point 1,
and so on. All the other losses do not care about the order of the points. A zigzag makes them just as happy as a straight line. So the other losses cannot fix the order, and their gradients cover up the one loss that can.

Think of 20 numbered beads that should sit on a straight wire, in order.

| Loss | What it checks | Effect on a zigzag |
|---|---|---|
| **Ordered L1** (poly) | each bead k must go to its own target spot k | the only loss that cares about order. On its own it fixes the line, so the model overfits. |
| **Box** (L1 / gIoU) | the outer rectangle around the points | does not care about order. Any shape that fills the same rectangle gets the same score, and it only moves the corner points. This is the main cause (removing it helped). |
| **Length** | the distance between neighbor points | any zigzag with the right spacing scores zero. Early in training it spreads the points in random directions, which starts the zigzag. |
| **Direction** (1 - cos angle) | the angle of each small piece | if fully satisfied it gives a straight line, so it does not prefer a zigzag. It is not a main cause (changing it did not help). |

## Why the ordered loss cannot fix the zigzag

The box and length losses are happy with a zigzag, so the ordered loss is the only one
that complains, and it gets outvoted:

- Forward/backward switching: A line has two ends, and reading its points from
either end describes the same line. So the loss matches the prediction to the ground
truth read forward or read backward, whichever is closer. A clean line clearly matches
one way, so the choice is stable. A zigzag sits evenly on both sides of the true line,
so both readings are about equally close and the model keeps switching. Each switch
sends a point's target to the opposite end of the line. Point 3 is told "go to the top"
one step and "go to the bottom" the next step, so the two orders cancel and every point
drifts to the middle. The order never settles.

- Gradient clipping: When two points sit almost on top of each other, the direction
loss gradient becomes very large. The gradient clipping then shrinks the whole gradient, including the ordered loss that would fix the order. This is why the loss keeps going down very slowly for thousands of epochs but never finishes.

Note: the direction loss has zero gradient when a piece points exactly forward (0 degrees) or exactly backward (180 degrees). That is true for the formula, but it only happens at those exact angles. Real zigzag pieces sit at in-between angles where the gradient is fine, so this is not the real cause here.

## The rule

The order of the points can only come from a loss that checks each point by its number.
The box, length, and direction losses do not check the order, so they should only be small
extra helpers on top of a strong ordered loss, never large.

MapTR agrees. Its recipe is: class loss plus ordered point loss at weight 5.0, direction
at a tiny 0.005, and no length or box loss at all. This matches the experiments: poly plus
class (plus aux) overfits, and adding box or length brings back the zigzag.

## What makes MapTR-style point queries work

Two more lessons, both confirmed in the code:

- Per-layer reference-point refinement plus dynamic query position: Each decoder layer
   does not predict the points from scratch. It starts from the current reference points and
   predicts a small correction, so the points get sharper layer by layer. At each layer the
   query position is rebuilt from the current reference point, so the attention knows where
   each point sits right now. This is the main reason MapTR-style point queries learn fast.

- The query content is learned, not only the query position: Each query has two learned
   parts: one for position (query_pos) and one for content (query_content, the starting input
   to the decoder). In plain DETR the content input starts at zero and only the position is
   learned. In MapTR both are learned (see `_build_hierarchical_queries`, where the query
   embedding is split into query_pos and query_content).

   This may help because the learned content gives each point query a starting hint
   about which point it is and how it relates to the other points in the same line,
   instead of starting from a zero vector.


## Converting the polyline loss into meters

- The polyline loss is an average L1 error per coordinate, taken over every point of
  every matched line. Points are normalized to [0,1] by the padded image size
  (pad_to - 1 = 1023), so multiplying the loss by 1023 turns it back into pixels.
  A loss of 0.004 is therefore about 4 pixels of error per coordinate, which at
  25 cm per pixel is about 1 meter on the ground. Note this is per coordinate, with
  x and y averaged separately, so the real distance from a predicted point to its
  target is somewhat larger, up to about 1.4 times.


## A zigzag early in training that fixes itself is normal

Seen in the working UNet runs: the lines look like a zigzag in the first epochs, then
become clean later. This is expected and is not a problem.

- At the start, all 20 points of a line begin near the same reference point, in no real
  order, so the first predictions look tangled.
- The ordered point loss plus the per-layer refinement then sort the points into order,
  and the zigzag goes away.

This is different from the old broken zigzag, which never went away. That one was held in
place because the box and length losses were happy with a zigzag. In these runs all the
geometric losses are set to 0 (only class + point L1 + aux are on), so nothing holds a
zigzag, and any early one clears up.

So the length loss idea is not the cause. The early zigzag is just the starting
state before the order is learned. A zigzag only becomes a real problem when a loss that
does not care about point order is strong enough to keep it.

## Why ordered L1 is preferable to a tolerance loss

With approximately 100,000 polylines whose ground-truth noise is mostly unbiased, L1 regression learns the conditional median. Random annotation errors therefore tend to average out, and the model may predict the true visible hedge location more accurately than some individual labels.

A tolerance band, where the loss becomes zero within ±ε, does not correct a systematic offset in the ground truth. It only declares a range of positions acceptable. It also removes the gradient near the target, exactly where the model is refining point placement and point ordering. Since ordered point loss is the main signal that teaches each predicted point its correct position in the sequence, weakening it may leave small zigzags, swapped points, or imprecise geometry.

Therefore, keep the standard ordered L1 loss for training. During inference and evaluation, compare predictions using buffered precision and recall at several tolerances, such as 1 m, 5 m, 10 m, and 15 m. This allows the evaluation to account for annotation uncertainty without weakening the training signal.

## A closed ring can have a near-duplicate point right after the start too

Opening a closed ring by dropping only the last point assumes the ring is
otherwise clean. In pdok_dataset3, 14 of 302 closed rings (4.6%) also have a
near-duplicate point right after the start (or before the end), e.g.
`[p0, p0+eps, ..., pN-eps, p0]`. Dropping only the last point still leaves a
near-zero-length first segment, which behaves like the box/length-loss
failure mode: a degenerate segment with an almost-undefined direction.

Fix: collapse consecutive near-duplicate points (within closed_eps_px) before
checking whether the ring is closed, then drop the last point
(`open_closed_ring` in `scripts/data/convert_pdok_polylines_to_detr_polyline.py`).
Verified on the full dataset: 0 rings left with a near-zero segment after the
fix, versus 12 before.

## Why block_m = 5000 m for the geographic split

The block size only has to satisfy two things: much larger than the 250 m
crop, so blocks and overlap groups do not fight each other, and small enough
that there are many blocks to hash, so the achieved val fraction lands near
the target. At 30k crops spread over the Netherlands, 5 km blocks give 19.1%
val against a 20% target, which is close enough. There is no optimum here; if
the exact fraction ever matters, the fix is to retry seeds until the achieved
fraction is within a tolerance, not to tune the block size.

## Length filters must be checked on the stored polyline

The converter first filters raw polylines by length, then resamples them to
20 equidistant points. `resample_polyline_equidistant` rounds coordinates to
0.1 px, so a polyline sitting right on the threshold (40.006 px) can end up a
hair below it (39.9997 px) after resampling. Two polylines in pdok_dataset3
did exactly this and were stored despite being under the threshold.

Fix: re-check the length after resampling, since the resampled polyline is
what training actually sees. Border clipping was verified not to change
lengths (0 of 13,416 polylines), so the check after resampling is enough.

## Regenerating a split must delete the old outputs

The converter writes `polylines/{train,val}/*.npz`. Re-running it with
different split settings (for example after enabling `avoid_label_dirs`) can
move a crop from val to train, but the stale copy in the other directory
survives, so the same image ends up in both splits. This is silent: 30,004
files were written for 30,000 images, with 4 stems duplicated.

Fix: delete existing NPZs in both split directories before writing. This was
caught by `verify_no_split_overlap` in `exps/quantify_geographic_crop_overlap_pdok_dataset3.py`,
which is a good argument for keeping such checks as asserts in the analysis
scripts rather than as one-off manual checks.

## The largest batch that fits is not the batch size to use

Probed on an A100-40GB with `exps/probe_batch_size.py` (real train step:
forward, loss, backward, optimizer, since backward dominates peak memory):

| batch | peak GB | s/step | s/image |
|---|---|---|---|
| 4 | 1.55 | 0.123 | 0.0307 |
| 16 | 5.76 | 0.445 | 0.0278 |
| 32 | 11.41 | 0.861 | 0.0269 |
| 64 | 22.67 | 1.678 | 0.0262 |

Memory is linear in batch size (about 0.354 GB per 1024x1024 image), so 40 GB
would fit roughly 100 images. But time per image is already flat at batch 16:
going from 16 to 64 saves 6% per image while cutting optimizer steps per epoch
by 4 (313 to 78 for a 5,000 image epoch). The GPU is saturated well before it
is full, so the extra memory buys steps, not speed.

Consequences:
- Pick the batch size where s/image stops improving, not the one that fills
  the GPU. Here that is 16, which also matches the effective batch size DETR's
  lr=1e-4 is tuned for.
- Headroom left over is useful for other things: a larger model, a finer
  feature stage (`up3` -> stride 8), or unfreezing the backbone.
- num_workers has to account for persistent workers on both the train and the
  eval loader: 8 workers means 16 processes, which is right for the 18 CPUs
  that come with one A100.

## A slurm log that stops updating is usually just buffering

In the first cluster run the .out file stopped at "Epoch 005 starting" while
nvtop showed the GPU busy, tensorboard showed 26 epochs, and best_1.pt was an
hour newer than the log. Nothing was wrong: Python block buffers stdout (8 KB)
when it is a file rather than a terminal, and one epoch prints about 330
bytes, so roughly 25 epochs accumulate before each flush. Tensorboard is
unaffected because SummaryWriter flushes on its own timer, and checkpoints are
written directly.

Fix: run the training script with `python -u` in the slurm script. Before
concluding a cluster job is stuck, compare the log timestamp against the
checkpoint mtime and the job state, not the log alone.

## A frozen run with the GPU idle is a DataLoader worker hang

Symptom: the log stops, but the main process spins at ~100% CPU while the GPU
sits at 0%, and (in a laptop run) the machine still feels busy. This is not
slow training, it is a hang. It froze the 5000-image laptop run at epoch 11,
right after the first eval (eval was epoch 10).

Cause: to load images fast, the DataLoader starts a few worker processes. There
are different ways to start them: fork (old, simple) and forkserver (newer).
Python 3.14 changed the default to forkserver, which deadlocks with persistent
workers across the eval to train transition: after the eval epoch the workers
respawn and the main process gets stuck coordinating with them.

Fix: force the fork start method for the DataLoader
(`multiprocessing_context="fork"`, done via `_mp_context` in
`scripts/train_detr_unet_polyline.py`). fork is safe here because the workers do
no CUDA work. The cluster was never affected (its Python defaults to fork, and
the cluster run passed many eval transitions), but the fix protects both.

How to tell a hang from slow training: check GPU utilization (nvtop), not just
the log. 0% GPU with a busy CPU means a worker hang; a busy GPU means it is just
slow. `exps/smoke_test_train_detr_unet_polyline.py` reproduces the eval to
train transitions quickly and fails if this regresses.

## No cheap measure can rank two checkpoints, so build the real one

Cluster run 1, two checkpoints, same cluster-val crops at threshold 0.95. Three
cheap rankings gave three answers: eval loss picked `best_1.pt` (epoch 65),
mean lines per image picked `1_150.pt`, mean error per image picked `best_1.pt`
again. The mean count is the trap: per image `1_150.pt` predicts nothing on two
crops that have hedges and 16 lines on a crop that has 3, and those cancel.

Buffered length over all 3,098 val crops settled it. `1_150.pt` wins at every
buffer, F1 0.640 against 0.614 at 10 m. So the checkpoint the eval loss calls
overfit is the better detector, and its advantage is recall.

Two rules follow. Do not report eval loss as a result, and do not early-stop on
it. Aggregate per image and then average, never as a ratio of two totals,
because errors of opposite sign cancel in a total.

## Chamfer plus Hungarian is the wrong metric for polylines

The obvious detection metric is to pair each prediction with one GT line, the
way a box detector pairs by IoU, and call the pair a hit if the chamfer
distance is under 5, 10 or 15 m. It gives F1 0.41 where buffered length gives
0.64, on the same predictions, and the figures agree with 0.64.

Two ordinary cases break the pairing:

- A hedge that turns a corner is one L-shaped GT line. A prediction covering
  one leg is half right, but chamfer averages over the whole GT line, so the
  far leg pushes the average to about 17 m and the pair fails at every
  threshold. The correct half is thrown away.
- A hedge stored as three overlapping GT lines can be paired with only one of
  them. The other two count as misses even though the hedge was covered.

Buffered length never pairs anything. It asks how much of the drawn length lies
within r of a real hedge, and how much of the real hedge length has something
drawn within r of it. Both cases then score honestly. Use `buffered_length_pr`
in `hedge_seg/metrics.py`; `matched_pr` is kept there only to reproduce this.

## Recall is limited by the score head, not by perception

Sweeping `infer_score_thresh` down to 0.05 on run 1 gives recall 0.75 at 10 m,
with precision falling to 0.41 and 8.4 predictions per image against 2.1 GT
lines. So the model already draws a line within 10 m of three quarters of all
GT hedge length. Those lines exist and are then discarded.

They are discarded because the scores cannot rank them. Deciles on 400 crops
are 0.775, 0.909, 0.961, 0.983, 0.990, 0.997, so nearly every prediction scores
above 0.9 and good and bad lines look alike. F1 goes 0.632 at 0.90, 0.640 at
0.95, 0.494 at 0.98: a knife edge, where 0.03 of threshold costs a third of the
score. A calibrated score would fade gradually.

This is not about images with no hedges; every crop has at least one. It is
about the 58 of 60 queries that must be labelled "no object" in every image.
`eos_coef=0.05` makes calling one of them a hedge cheap, so nothing pushes
their scores down. Raising `eos_coef`, or replacing the softmax class head with
a focal sigmoid head as Deformable-DETR and MapTRv2 do, is the change to try.

The threshold is not a constant. It was optimal at 0.95 for exp 1 and moved to
0.90 for exp 2, worth F1 0.699 against 0.685. Re-sweep it after any change to
the model or the data, and record the value with the run.

## What 5.4x the data actually bought

Exp 2 is exp 1 with the full 26,902 train crops instead of 5,000, nothing else
changed. F1 at 10 m went 0.640 to 0.685, and at 5 m 0.489 to 0.547.

The split of that gain is the useful part. Precision rose about 0.08 in every
density bucket. Recall rose about 0.02 and did not move at all on the crowded
crops, where it is worst (0.372 on crops with 7+ hedges). So more data makes
the model surer about the lines it already draws, and does not make it find
more lines in a busy crop. Do not expect a third dataset increase to fix
recall; that is the score head's job.

Overfitting did vanish. The train-eval gap went from 0.29 to 0.03 and eval loss
was still falling at the last epoch, so 45 epochs was short rather than long.

One regression came with it. Predicted straightness rose to 0.942 against 0.909
in the labels, and the strongly bent share halved to 0.101 against 0.220. Exp 1
matched the labels almost exactly. An L1 loss under uncertainty about where a
corner sits is minimised by cutting the corner, and more data appears to have
sharpened that bias rather than removed it. Watch it in the next run.

## A checkpoint has to be evaluated on the split it was trained against

Run 1 trained on the cluster conversion of pdok_dataset3 (26,902 / 3,098). The
local conversion is 24,257 / 5,743, because `avoid_label_dirs` sees 10
pdok_dataset2 labels locally and 5,000 on the cluster. Same seed and blocks, so
the cluster val set is a subset: all 3,098 stems were found locally. About 46%
of the local val crops were cluster training images, so scoring a cluster
checkpoint on the local val directory shows training images half the time.

The split is part of the run, like the weights. Keep the val stem list with the
checkpoint, and do not assume two conversions of the same dataset agree.

## The "nothing found" branch is the one that breaks

Raising `infer_score_thresh` to 0.95 made some crops return no prediction, and
inference stopped with "can't convert cuda:0 device type tensor to numpy".

Predictions are made on the GPU. numpy only reads CPU memory, so they must be
copied over with `.cpu()` before saving. The normal path does that. The
separate early-return path for "nothing passed the threshold" builds empty
tensors with `new_zeros` and returns straight away, and we forgot the `.cpu()`
there. `new_zeros` keeps the device of the tensor it came from, which is its
documented job, so those empty tensors stayed on the GPU.

Ours, not PyTorch's. PyTorch never moves data between GPU and CPU by itself; a
hidden copy would be slow and would hide exactly this kind of mistake.

At threshold 0.5 every crop had a prediction, so the branch never ran. When a
change makes empty results possible, look there first.

## Output directories should name themselves after what produced them

Comparing two checkpoints used to mean editing two paths, running, renaming the
old ground-truth directory, re-running a symlink loop by hand, then editing
paths in the plotting code. A figure could not be traced back to what made it,
and re-running overwrote the previous result.

Now `infer` writes to `<root>/<ckpt stem>_<split dir>_t<threshold>/` and links
the ground truth of exactly those crops into `gt/` beside the predictions. Runs
cannot collide, and the two figures cannot disagree about which images they
show.

Settings stay in the committed cfg block. Command-line overrides were tried and
removed: they make a run depend on something the file does not show.

## Two label artifacts, one real and one that turned out not to matter

Both are visible by eye in the run 1 figures, but only one survives measurement.

- One hedge split into several overlapping polylines: **not a problem**.
  Merging GT lines within 10 m of each other removes 0.8% of GT lines and moves
  F1 by 0.001. Buffered length is immune to it anyway, since it measures length
  covered rather than lines matched. Do not spend more time on it.
- Hedges missing from the labels: **still open**. The model draws them and the
  metric counts them as false positives, so the reported precision of 0.70 is a
  lower bound. Nothing can be fixed in the labels; the useful work is to
  quantify it by classing the unmatched predictions in the lowest-precision
  crops, so the result can be stated honestly.

The general rule stands: measure the artifact before designing around it. The
first one cost a paragraph of worry and was worth 0.001.

## The worst crops were campsites, and filtering them out is not worth it

Ranking val crops by false-positive length put the same thing on top every
time: rows of identical chalets on small plots, each plot ringed by a clipped
hedge. Not housing estates, which is what they look like. Top10NL lists 6
building footprints in one whole 250 m crop and calls the ground grassland,
because a chalet is not a registered building, so a built-up or building
filter misses them entirely. The layer that finds them is
`functioneel_gebied_vlak`, field `typefunctioneelgebied`, values camping,
vakantiepark, bungalowpark, caravanpark. On 13 crops sorted by eye the
separation was clean, 8 of 8 bad and 0 of 5 good.

Then measuring killed the idea. Those crops are 315 of 3,098, 10.2%. Excluding
them moves F1 from 0.640 to 0.652. They really are harder, 0.542 against 0.652,
but 10% of crops cannot move a per-image average much.

Two lessons. A crop that looks urban may be a campsite, and the layer that
identifies a land use is not always the obvious one. And an exclusion can
always be tested by dropping crops at scoring time
(`exps/probe_recreation_crops.py` writes the stem list) before anyone touches
the dataset. That test cost an afternoon and saved a reconversion.

## Done

Phase A — data + split:

- Converter opens closed rings (with the near-duplicate-point fix above) and
  raises min_length_px to 40 px (10 m).
- Geographic train/val split from center_world: crops grouped into 5 km
  blocks, blocks hashed into train/val, connected components of overlapping
  crops assigned as a whole, KDTree-verified zero overlap. Rationale for
  needing this: pdok_dataset3 crops are sampled per polyline, so 72.6% of
  crops overlap at least one other crop, and a naive random 80/20 split lets
  68.5% of val crops overlap a training crop (`exps/quantify_geographic_crop_overlap_pdok_dataset3.py`).
- Training script reads train/val from the split directories instead of an
  internal random split.
- Converted dataset3 (30k images). The operative cluster split (avoid_label_dirs
  sees all 5,000 pdok_dataset2 labels) is train=26,902, val=3,098 (10.3%),
  103,414 polylines, 302 rings opened, 0 val-train overlap; enforcing backbone
  avoidance halves the val set because 40% of crops sit near semseg areas. The
  local dry-run (only 10 pdok_dataset2 labels available) was 24,257 / 5,743.
  Stats reproducible with `exps/dataset_stats.py` and
  `exps/quantify_geographic_crop_overlap_pdok_dataset3.py`; per-run numbers in
  `docs/experiment_log.md`.
- `hedge_seg/paths.py` resolves DATA_ROOT / EXP_ROOT / CLUSTER_EXP_ROOT from
  a filesystem marker, so scripts no longer need path edits when moving
  between the local machine and the cluster.

Phase B — training half:

- Cluster job 24799874 (exp 1) finished, 150 epochs in 7 h on a 5k
  spatially-blocked train subset, frozen backbone up3, augment on, scheduler
  on, eval_every=5. First run whose predictions sit on real hedgerows in
  images the model never saw.
- Inference re-run on the cluster val split only, using `polylines/val_cluster`
  (commands in `docs/experiment_log.md`). The local val directory is a
  different split, about 46% of it was in the cluster's training set, so
  figures taken from it were not the baseline.
- `infer_score_thresh` raised from 0.5 to 0.95 on line count, then confirmed by
  sweep: 0.95 is the optimum for `1_150.pt`.
- Inference made reproducible: self-naming run directories, ground truth
  linked next to the predictions, no manual steps.

Phase B — measurement half. All of it, and it needed no training run:

- `hedge_seg/metrics.py` and `exps/probe_polyline_pr.py`: buffered-length
  precision and recall per image at 5, 10 and 15 m, plus the stratification,
  the threshold sweep, the GT merge test and the geometry check.
- Baseline on the 3,098 cluster val crops at 10 m: `1_150.pt` 0.70 precision,
  0.59 recall, F1 0.640. At 15 m 0.78 / 0.67. It beats `best_1.pt` at every
  buffer, so the ranking is settled and the eval loss was wrong.
- Rough placement against YOLO-seg (80% precision, 70% recall): close on
  precision at 15 m, behind on recall. Not the same measurement, so it is a
  placement and not a parity claim.
- Chamfer plus Hungarian rejected, with the reason recorded.
- GT merging measured and dropped. Campsite exclusion measured and dropped.
- Straightness and length checked: predictions match the labels, so the
  "the model only draws straight lines" worry from the figures was wrong.
- Recall traced to the score head rather than to perception.

## TODO

Phase C — exp 2 is done (F1 0.640 -> 0.685 at 10 m, see above). Next, in order:

- Exp 3: the score head. Raise `eos_coef` from 0.05, or swap the softmax class
  head for a focal sigmoid head. This is now clearly the biggest lever. Exp 2
  reaches recall 0.840 at threshold 0.05 and only 0.608 at 0.95, and the extra
  data moved recall by 0.02 while moving precision by 0.08, so the lines are
  being found and then thrown away by the ranking. Train on the full data so it
  compares against exp 2. A cheap read first: resume from `best_2.pt` with the
  new coefficient for a few epochs and watch whether the scores spread out.
- Exp 4: longer. Exp 2's eval loss fell at every eval including the last, so 45
  epochs was short. Only worth spending after exp 3, since the score head is
  the larger effect.

Then, results-driven:

- Quantify the missing-label rate by classing unmatched predictions in the
  lowest-precision crops. It sets how much of the 0.30 false-positive length is
  real error, and it is needed before any precision number is published.
- Tree lines as a second class, from
  `Top10NL2023_inrichtingselementen_lijn_bomenrij.shp`. Worth doing for the
  class itself, not for the hedgerow score: precision is already 0.70 to 0.83
  and the worst false positives were campsites, not tree rows. The crop world
  extents already exist, so it is clip to each crop, append with `label=1`, set
  `num_classes=2`.
- Backbone unfreezing with low lr, and MapTRv2 decoupled self-attention,
  regression-tested with the one-image overfit.
- LiDAR height (`ahn4_10m_perc_95_normalized_height.tif`) last. At 10 m per
  pixel it cannot localise a 3 m hedge, only say that tall vegetation is
  present, which helps the part that is least broken. If tried, fuse it as a
  side branch so the semseg-pretrained RGB backbone stays untouched.

Note: the spatial split makes val numbers look worse than a random split would.
That is expected; they are the real baseline to improve from.


