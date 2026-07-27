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

## Eval loss is not detection quality

In cluster run 1 the eval loss bottomed near epoch 65 and rose to the end, so
`best_1.pt` was saved at 65. But the final epoch 150 checkpoint draws better
polylines: 4.16 lines per image against 3.53 in GT, versus 2.91 for best_1 on
the same crops. best_1 is cleaner but misses whole hedges. By the loss, best_1
wins. By the pictures, the last epoch wins.

The reason is what the loss is made of. Most of it is classification over 60
queries against about 3 real lines. Overfitting first ruins the confidence
calibration on unseen images, which raises the cross-entropy term, while the
geometry of the lines keeps improving or stays flat. The number that goes up is
not the number we care about.

Consequences:
- Do not pick a checkpoint by eval loss on this task, and do not report eval
  loss as the result. Always look at the predictions.
- The buffered precision/recall metric is not an extra for comparing against
  YOLO-seg. It is required, because without it there is no way to say which of
  two checkpoints is better. That is why it moved to the front of the queue.
- Keep saving the final checkpoint, not only `best_*.pt`. Same conclusion as
  the small overfit runs reached, now for a real 5k run.

## The detection score threshold has to be tuned, 0.5 is not a default

The model's scores are squashed into the top of the range: measured on 400 val
crops, the deciles are 0.775, 0.909, 0.961, 0.983, 0.990, 0.997. The median
prediction scores 0.96. So a 0.5 threshold keeps nearly every query and the
output looks flooded: 6.1 lines per image against 3.4 in GT. At 0.95 the counts
almost match, 3.50 against 3.39.

This is worth knowing before blaming the model for over-detecting. Much of the
apparent duplication is a threshold that was never set. It costs nothing to fix
and needs no retraining.

Two cautions:
- Matching the count is not the same as matching the location. Only buffered
  precision/recall says whether the kept lines are the right ones.
- The right threshold is a property of a trained model, not a constant. Sweep it
  with the metric once the metric exists, and record the value with the run.

## A checkpoint has to be evaluated on the split it was trained against

Cluster run 1 was trained on the cluster conversion of pdok_dataset3
(train 26,902 / val 3,098). The local conversion of the same images is
train 24,257 / val 5,743, because `avoid_label_dirs` only sees the 10
pdok_dataset2 labels that exist locally, while the cluster sees all 5,000.

Both use the same seed and block hashing, so the cluster val set is roughly a
subset of the local one. That means about 46% of the local val crops were in
the cluster's training set. Running a cluster checkpoint over the local val
directory therefore shows training images about half the time, and the result
looks better than it is.

The split is part of the run, like the weights. Copy the cluster's val stem
list next to the checkpoint and filter with it, do not assume two conversions
of the same dataset agree.

## An empty result is a code path, and it is the one that breaks

Raising `infer_score_thresh` to 0.95 made some crops produce no prediction at
all, and inference crashed with "can't convert cuda:0 device type tensor to
numpy". The normal path in `detr_polyline_inference` ended with
`.detach().cpu()`, but the early-return branch for "nothing passed the
threshold" built its empty tensors with `new_zeros`, which inherits the CUDA
device, and skipped the `.cpu()`. It went unnoticed because at threshold 0.5
every crop had something to return.

The general point: the empty case usually gets written once and never exercised,
so it drifts away from the main path. When a change makes empty results possible
(a higher threshold, a stricter filter, a smaller subset), that branch is the
first place to look. Here the fix is one `.cpu()` per line; the cost was an
afternoon of thinking the checkpoint was corrupt.

## Output directories should name themselves after what produced them

Comparing two checkpoints used to mean editing `infer_out_dir` and
`infer_ckpt`, running, renaming the previous ground-truth directory, re-running
a symlink loop, then editing paths in the plotting code. Every step was manual,
so a figure could not be traced back to the checkpoint, split and threshold that
made it, and re-running quietly overwrote the previous result.

Two changes remove all of it:

- `infer` writes to `<root>/<ckpt stem>_<split dir name>_t<threshold>/`, so
  `1_150_val_cluster_t0.95` says exactly what it is and nothing collides.
- It writes the matching ground truth as symlinks into `gt/` inside that same
  directory, for exactly the crops that were run. The prediction grid and the
  GT grid then cannot disagree about which images they show.

Plus command-line overrides via `OmegaConf.from_cli()`, so a comparison is two
commands and no file edits. Keep that for inference and quick checks only.
Training runs should still edit and commit the config, because the committed
script is the record of what ran.

## Ground-truth artifacts that will distort precision and recall

Two label problems are visible by eye in the run 1 figures, and both will show
up as errors that are not the model's fault:

- Missing hedges. Some clear tree rows are not labelled at all. The model draws
  them, and a naive metric counts them as false positives.
- One hedge split into several polylines. In one crop, 3 overlapping GT lines
  cover a single field boundary while the model predicts 1. A naive one-to-one
  match counts 1 hit and 2 misses.

So when the buffered metric is built, do not stop at the first number. Look at
the worst false positives and false negatives before believing them. Merging
overlapping GT polylines that lie within the buffer of each other is worth
testing as a preprocessing step.

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

## TODO

Phase B — baseline on a 5k spatially-blocked train subset, full val, frozen
backbone up3, augment on, scheduler on, eval_every=5. Cluster job 24799874
(exp 1) finished, 150 epochs in 7 h. The training half is done and the result
is the first one where predictions sit on real hedgerows in unseen images
(details in `docs/experiment_log.md`). The measurement half is not. Remaining,
in order:

- Build the buffered precision/recall metric (match predicted to GT polylines
  within 5/10/15 m). This is now blocking, not optional: eval loss disagrees
  with the pictures about which checkpoint is better, so there is currently no
  way to rank two runs. It is also what makes the result comparable to YOLO-seg
  (80% precision, 70% recall).
- Re-run inference on the cluster val stems only. The local val directory is a
  different split and about 46% of it was in the cluster's training set, so the
  current figures are indicative only.
- Sweep `infer_score_thresh` with that metric. 0.95 matches the GT line count,
  but count is not location.
- Check the worst false positives and negatives against the labels before
  trusting the number, given the GT artifacts above.

Note: the spatial split makes val numbers look worse than a random split would.
That is expected; they are the real baseline to improve from.

Phase C — decoupled self-attention (MapTRv2) as a clean A/B, regression-tested
with the one-image overfit.

Phase D — results-driven: full 30k run, backbone unfreezing with low lr, and
tolerance-loss/border-filtering only if the error analysis points at them.


