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

## Most of the recall at a low threshold is clutter, not lost hedges

This corrects the central claim of the last two months of work. "Recall is 0.84
at t=0.05, so the model finds the hedges and the score throws them away" was
wrong, and it cost an 11 hour run (exp 3) to find out.

The hole in it: at t=0.05 the model draws about 13 lines per image against 2.1
labelled lines. Buffered recall asks how much labelled length has *any*
prediction within 10 m, and with 13 lines on a 250 m crop a lot of that happens
by luck. `exps/probe_recall_null_model.py` measures the luck by scoring each
crop's predictions against a *different* crop's labels, so the number, length
and orientation of the lines are unchanged and only the link to the image is
broken:

| threshold | pred/img | real recall | null recall | skill |
|---|---|---|---|---|
| 0.05 | 12.6 | 0.840 | 0.379 | 0.461 |
| 0.40 | 7.9 | 0.805 | 0.334 | 0.471 |
| 0.90 | 3.2 | 0.696 | 0.266 | **0.431** |
| 0.95 | 1.9 | 0.608 | 0.228 | 0.380 |

Nearly half of the celebrated 0.84 is what another crop's predictions would
score. Skill peaks at 0.471 and is already 0.431 at the operating point, so a
perfect score head is worth about 0.04 of recall, not the 0.23 the naive
reading suggested. Exp 2's t=0.90 is close to the best the current predictions
can support.

Two rules. Any metric that rewards drawing more lines needs a null model before
it is used to justify a run. And a threshold sweep is not evidence of headroom:
falling precision as the threshold drops is the same fact as rising recall.

The threshold is still not a constant. It was 0.95 for exp 1, 0.90 for exp 2
and 0.40 for exp 3. Re-sweep after any change and record the value.

## The score ranks lines by how straight they are, not by whether they are right

Measured on exp 2 `best_2.pt`, all 3,098 val crops from the t=0.05 run. A
prediction counts as correct if at least 80% of its length is within 10 m of a
label. 7,956 of 39,104 predictions are correct.

| straightness | median score, correct | median score, wrong |
|---|---|---|
| < 0.85 (bent) | **0.453** | 0.216 |
| 0.85 to 0.95 | 0.883 | 0.690 |
| 0.95 to 0.99 | 0.951 | 0.868 |
| > 0.99 (straight) | 0.974 | **0.957** |

A correct bent line scores 0.45 and a wrong straight one scores 0.96, so no
single threshold can keep both. At t=0.95 only 9% of correct bent predictions
survive against 74% of correct straight ones. The score's AUC for correct
against wrong is 0.767 overall but only 0.618 inside the bent group, so most of
its apparent skill is straightness acting as a proxy.

Two earlier readings were wrong because of this, and both are corrected above:

- "Exp 2 draws straighter lines than the labels" was a thresholding artifact.
  The model draws bends. At t=0.80 predicted straightness is 0.908 against
  0.909 in the labels, an almost exact match. Only the 0.95 cut makes it 0.942.
- The bends problem and the recall problem are one problem, not two.

Exp 3 tried to fix this with a focal sigmoid head and failed. The ranking did
not improve: AUC 0.742 against exp 2's 0.767 overall, and 0.648 against 0.618
inside the bent group. The scores did not spread, they clumped: a mass near 0.2
and a spike above 0.98 where correct and wrong straight lines sit together at
0.983 and 0.979.

So the classification *loss* is not the problem. The class head reads the mean
of the 20 point features (`class_embed(hs_poly.mean(dim=3))`) and never sees how
well those points fit the image, so no reweighting of that head's loss can make
its output track geometric quality. Changing what the head is told, rather than
how it is scored, is the only version of this idea left, and after the null
model above it is worth much less than it looked.

Reproduce with `exps/probe_score_quality.py`, which prints exp 2 and exp 3 side
by side.

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

The apparent straightness regression was not one. Predicted straightness at
t=0.95 rose to 0.942 against 0.909 in the labels, but that is the threshold
selecting straight lines, not the model losing bends. At t=0.80 it is 0.908.
See "The score ranks lines by how straight they are" above.

## Check an overlay with a shift test, not with your eyes

Before lidar can be an input, the crop-to-raster maths has to be right, and a
one-cell error is 10 m. Looking at the overlay is not enough: the first crops
tried were the ones with the most labelled hedge, which are so cluttered that
something bright sits near every line whether the maths is right or wrong.

Two changes made it decisive.

- **Pick uncluttered crops to look at.** One long hedge across an empty field
  shows a shift immediately. `pick_ids(..., max_lines=2)`.
- **Measure it.** Sample the metric in the cells the hedges pass through, minus
  the cells they do not, then repeat at whole-cell offsets. On `perc_95` over
  40 crops the peak is at offset (0, 0) at 1.35 m, with every neighbour lower
  (0.89 to 1.00). So the maths is right, and this now regression-tests itself.

`exps/probe_lidar_crop_alignment.py`. It also writes `crop_footprints.geojson`
so the same crops can be checked in QGIS against the uncropped layers.

Two things this turned up in passing. Drawing the patch with nearest-neighbour
interpolation matters, because smoothing hides exactly the half-cell shift being
looked for. And in some crops the bright band follows the woody line visible in
the photo while the label sits a little to one side of it, which is the 2014
digitising accuracy showing up directly.

## Test an augmentation by making two code paths agree

The second half of the lidar check was the dangerous one. The dataset flips and
rotates the image and the polylines before padding, so a lidar array added
there has to go through the same steps, and if it misses one **nothing raises**:
the shapes stay right and only the content is wrong. It would have shown up as
a slightly worse run, months later, with no obvious cause.

What makes it testable is that the polylines and the lidar are augmented by
separate lines of code. So instead of checking either against the truth, check
them against each other: the presence channel says where the laser found
vegetation, and it should be high in the cells the hedges pass through and low
elsewhere, whether or not augmentation is on.

| | under hedges | elsewhere | gap |
|---|---|---|---|
| augment off | 0.920 | 0.456 | +0.463 |
| augment on | 0.917 | 0.456 | **+0.461** |

The gap survives, so both paths agree. And the control matters more than the
result: deliberately skipping the lidar rotation drops the gap to **+0.187**,
which is what shows the test can fail. A passing test nobody has seen fail is
not evidence.

`augmentation_test` in `exps/probe_lidar_crop_alignment.py`.

## Nodata in the AHN4 metrics means no vegetation, not missing data

24 of the 25 metrics are computed from vegetation returns only, so a cell with
nothing woody in it has no value at all. That is 48.4% of all cells, but only
7.2% of the cells a hedge passes through, against 51.8% elsewhere.

Two consequences. Filling the metrics with 0 is right rather than a fudge: no
vegetation is zero density and ground-level height. And the validity channel is
not bookkeeping, it is the single most informative channel of the six, since
"is anything growing here" is
most of what a 10 m grid can say about a 3 m hedge.

## The images are three years newer than the labels, not ten

Two facts, both checked:

- `Actueel_ortho25` is bit-identical to `2025_ortho25`: 0.00 mean pixel
  difference on a test crop, against 29.51 for `2022_ortho25`, the year the
  labels come from. `pdok_dataset3` was downloaded 2026-04-29, so the crops are
  **2025** imagery.
- Top10NL2023 was "herzien op basis van luchtfoto 2022"
  (`BRT_Actualiteitskaart_april_2023.pdf`), so the labels are **2022**.

So the gap is about three years.

An earlier note here called it ten years, from `bronactual` sitting in 2014 or
2015 for 75.3% of the 62,415 hedge features. That read the field wrong.
`bronactualiteit` is the currency of the source used for the last edit of that
feature, not the last time the map was checked. Top10NL is revised for the
whole country every year, and changes are trigger-based: two years of photos
are compared and only a detected difference causes an edit. A hedge with
`bronactual` 2015 was therefore re-confirmed as unchanged many times since. The
residual worry is positional rather than existential: those geometries have not
been redrawn in a decade, so they carry 2014 digitising accuracy.

The cheap fix is not a re-download. Top10NL2025 was revised on 2024 and 2025
photos, so taking the 2025 labels against the existing 2025 crops closes the
gap to about zero and touches no imagery. Re-downloading crops at
`2022_ortho25` to match the 2023 labels is the other direction and costs 47 GB.

## Read the label definition before choosing a feature

Top10NL does not separate hedges from tree rows by height. It separates them by
whether the vegetation blocks the view at about eye level
(https://kadaster.github.io/imbrt/):

- bomenrij: at least 3 trees in a row, spaced so that "tot manshoogte geen
  zichtbelemmering" (up to man-height it does **not** block the view).
- heg, haag: a row of trees, with or without shrubs, spaced or under-grown so
  that "tot minstens manshoogte het zicht belemmerd wordt" (up to at least
  man-height it **does** block the view).

So a tall row of trees is a heg when it has understory and a bomenrij when it
does not. Height was never the criterion, and a first pass here that tested
`perc_95` height alone concluded "lidar cannot separate them" at balanced
accuracy 0.638. That conclusion was wrong, and it was wrong because the metric
was chosen before the definition was read.

## LiDAR structure does separate hedges from tree rows

All 25 AHN4 metrics, 2,000 features per layer, split by 5 km geographic blocks
so nearby features cannot straddle train and test:

| features used | balanced accuracy | AUC |
|---|---|---|
| height p95 only, 1 metric | 0.601 | 0.649 |
| all **7** height metrics | 0.709 | 0.771 |
| the other **18**, no height at all | **0.765** | **0.844** |
| all 25, gradient boosting | **0.778** | **0.852** |

(7 height metrics, not 8, as an earlier version of this table said. The paper
lists max, mean, median and the 25th, 50th, 75th and 95th percentiles, and
median and perc_50 are the same quantity in two identical rasters, so there are
really only 6 distinct ones.)

Structure alone beats every height metric put together. The best single metric
is the share of vegetation returns between 1 and 2 m, which is a literal
measurement of the rule above:

| metric | heg | bomenrij | best single cut |
|---|---|---|---|
| BR_1_2 (returns 1-2 m) | 0.079 | 0.008 | 0.739 |
| BR_2_3 (returns 2-3 m) | 0.066 | 0.007 | 0.701 |
| BR_above_3 | 0.686 | 0.933 | 0.696 |
| height p95 | 8.0 m | 12.1 m | 0.635 |

A hedge puts ten times as much of its return profile in the 1-2 m layer as a
tree row does. Two details worth keeping:

- `pulse_penetration_ratio` is nearly useless here (0.526) despite measuring
  openness, because it is ground returns over **all** returns in a 10 m cell,
  and a linear feature covers only part of a cell, so the surrounding field
  dominates. The band ratios use vegetation returns only, which normalizes that
  away. Pick the metric whose denominator matches the thing being measured.
- Each raster carries its own nodata value, and the band ratios use +3.4e38
  while the height rasters use -3.4e38 or -99999. A single `value < -100` filter
  silently keeps the positive sentinels.

This is per-feature classification given that a line is already there, so it is
an upper bound on what a 10 m grid can contribute, not a detector result. At
10 m a 250 m crop is only 25x25 cells, so the right role is a side branch that
informs the class, never the geometry.

Reproduce with `exps/probe_lidar_hedge_vs_tree.py`.

## Exp 3: the focal head made it worse, and that was informative

Exp 2 with the softmax class head swapped for a focal sigmoid, MapTR weights,
nothing else changed. Best F1 at 10 m fell from **0.699 to 0.664**.

Each run at its own best threshold, so neither is handicapped. Exp 3's better
checkpoint is the last epoch (`3.pt`, F1 0.664); `best_3.pt` gives 0.660.

| | exp 2 `best_2.pt` | exp 3 `3.pt` |
|---|---|---|
| best F1 at 10 m | **0.699** at t=0.90 | 0.664 at t=0.40 |
| P / R there | 0.701 / 0.696 | 0.779 / 0.578 |
| pred per image there | 3.2 | 1.6 |
| recall at t=0.05 | 0.840 | 0.845 |
| score AUC, correct vs wrong | 0.767 | 0.739 |
| same, inside the bent group | 0.618 | 0.642 |

(The AUC rows are `3.pt`. For `best_3.pt` they are 0.742 and 0.648.)

Perception did not change: recall at t=0.05 is the same to three decimals, so
the model draws the same lines. The head just became more decisive about the
wrong thing. Precision rose to 0.84 and recall collapsed to 0.51, and the score
went bimodal rather than calibrated.

Three things to take from it:

- A more expressive loss on a head that cannot see the evidence does not make
  it better informed. The class head averages the 20 point features and never
  compares them to the image.
- Focal's prior at p=0.01 plus normalisation by matched-target count makes the
  head very reluctant. With 60 queries and 2.1 real lines, "predict almost
  nothing" is a good place to sit.
- The A/B was worth running even though it lost, because it is what forced the
  null model above, and that overturned a bigger and older claim.

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
- Woody lines missing from the labels: **measured, and a quarter of it is tree
  rows**. See the next section. The reported precision is a lower bound.

The general rule stands: measure the artifact before designing around it. The
first one cost a paragraph of worry and was worth 0.001.

## Top10NL says itself that both layers are incomplete

The specification (https://kadaster.github.io/imbrt/, extracted into
`/home/fatemeh/Downloads/hedge/Top10NL/Top10NL Metadata.docx`) marks both heg
and bomenrij "Volledigheid: Beperkt", limited completeness, and says in the heg
collection criteria:

> "Wordt niet opgenomen binnen bebouwd gebied en tussen tennisbanen. Een heg,
> haag op of rondom een erf wordt niet ingewonnen, tenzij deze zich voortzet
> voorbij het erf."

Plainly: a hedge is not recorded inside a built-up area, or between tennis
courts, and a hedge on or around an *erf* is not recorded unless it continues
past it. An *erf* is the plot or grounds belonging to a house or farm, not a
farmyard specifically, so this excludes ordinary garden and property
boundaries as well as agricultural ones.

The nominal minimum length is 100 m for both classes, but 18.0% of heg features
(11,208 of 62,415) and 16.9% of bomenrij features (45,028 of 266,783) are
shorter, because segments split at a road or a watercourse are exempt. Measured
with `geometry.length` on the shapefiles in EPSG:28992, where the unit is
metres.

This is the authoritative version of a thing already measured from the other
side: reported precision is a lower bound, because the model is penalised for
finding woody lines that the map excludes by rule.

It is tempting to go one step further and say this is *why* built-up and
campsite crops top the false-positive ranking. Do not. Two things are known
separately, that the rules exclude those areas and that those crops score
worst, and nothing yet connects them. Sparser labels there could equally come
from the mapper skipping cluttered scenes, or from those crops being genuinely
harder. Testing it means checking whether the unmatched predictions in those
crops sit on woody lines the rules exclude, the way
`exps/probe_treeline_overlap.py` did for tree rows. Until then this is
consistent, not shown.

## A quarter of the false positives are tree rows, not mistakes

Top10NL splits linear woody features into two layers. Training uses only
`inrichtingselementen_lijn_heg`. Tree rows are a separate layer,
`inrichtingselementen_lijn_bomenrij`, with 266,783 features against 62,415
hedges, and they are not in the labels at all. From above a tree row looks much
like a hedgerow, so the model draws it and every metre counts as a false
positive.

`exps/probe_treeline_overlap.py` measures it. On 300 random val crops, **24.3%
of the predicted length that matches no label is within 10 m of a tree line**,
and only 1.2% is near another hedge, so this is not the crop clipping losing
labels. The sanity check passes: 100% of the training GT sits on the heg layer.

Exp 2 precision at 10 m is 0.783, so 0.217 of predicted length is unmatched and
24.3% of that is 0.053. That is the measured size of the prize for adding tree
lines as a second class, and it is why the reported precision is a lower bound
on hedgerow performance rather than an honest error rate.

This reverses an earlier call. Tree lines were ranked low because the worst
crops looked like campsites rather than tree rows. The worst crops are not the
typical crops: campsite exclusion moved F1 by 0.011, while tree rows account
for a quarter of all unmatched length. Rank a fix by its share of the total,
not by how bad the worst examples look.

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
- Straightness and length checked: predictions match the labels below the
  reporting threshold, so the "the model only draws straight lines" worry from
  the figures is the threshold, not the model.

Phase C — the score head, closed:

- Recall was traced to the score head rather than to perception. **That was
  wrong.** Exp 3's focal head lost (F1 0.699 -> 0.664), and the null model then
  showed that half of the recall it was chasing is clutter. Both sections
  above. Perception, not ranking, is the limit.

## TODO

Phase C — exp 2 is done (F1 0.640 -> 0.685 at 10 m, see above). Next, in order:

Exp 3 is done and lost (0.699 -> 0.664). Exp 2 `best_2.pt` at t=0.90 remains
the baseline. The score head is closed as a line of work: the null model says
there is about 0.04 of recall behind it, not 0.23.

The bottleneck is that the model does not draw enough correct lines, worst
where crops are crowded (exp 2 recall 0.372 on crops with 7+ labelled lines
against 0.731 on crops with one). So the next runs should target perception.

- Exp 4: finer features. `feature_stage="up3"` is stride 16, so a 1024 px crop
  becomes a 64x64 grid and a 3 m hedge is 12 px, under one feature cell. Thin
  structures cannot survive that. `FEATURE_STAGES` only has `enc4` (stride 32)
  and `up3` (stride 16) today, so this means adding the next UNet decoder
  stage at stride 8. Memory is available: batch 16 uses 5.8 of 40 GB, and
  stride 8 is 4x the tokens. This is the most direct attack on the real limit.

- Exp 5, the data step, two things in one conversion: tree lines as a second
  class, and labels from Top10NL2025 instead of 2023. The second closes the
  three-year image/label gap at no imagery cost, since the crops are already
  2025. Both shapefiles are downloaded, in
  `/home/fatemeh/Downloads/hedge/Top10NL2025`. Tree lines are worth 0.053 of
  predicted length on their own.

- LiDAR as a side branch, once the second class exists. Structure metrics
  separate heg from bomenrij at balanced accuracy 0.778, so this is the partner
  to exp 5. Feed the band ratios (BR_1_2, BR_2_3, BR_above_3, BR_below_5) and a
  couple of variability metrics, not `perc_95` height and not all 25. Fuse at
  10 m so the semseg-pretrained RGB backbone stays untouched, and use it only
  for the class head, never for geometry.

- Backbone unfreezing with low lr. Only 5.8 M of the network trains today. Pair
  it with exp 4 rather than running it alone.

Then, results-driven:

- Longer training. Exp 2's eval loss fell at every eval including the last, so
  45 epochs was short. Cheap, but it buys less than the above.
- Image resolution. Everything so far is 25 cm. Downsampling to 50 cm or 1 m
  costs nothing to try and would say how much of the result depends on
  resolution, which matters for applying this outside PDOK coverage. Not urgent.
- Border filtering. Polylines are clipped to the crop bounds, so a hedge
  crossing the edge becomes a truncated line the model is asked to predict
  exactly. Nothing drops or down-weights them. Still untested rather than
  dismissed.

Dropped, with the reason:

- The score head, after exp 3. See above.
- Bends as a separate work item. They are the threshold's doing, not the
  geometry's.
- A date-matched rebuild on `2016_ortho25`. 2016 imagery matches nothing here;
  the labels are 2022 and the crops are 2025.

Note: the spatial split makes val numbers look worse than a random split would.
That is expected; they are the real baseline to improve from.


