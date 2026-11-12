# Lessons learned

Lessons that change what we do next. Each gives the reason, the number and
the script behind it. Where the project stands is in `docs/status.md`, and the
details of each run are in `docs/experiments_log.md`.

- Labels and data: what the Top10NL labels hold and miss, tree rows, the
  split, the amount of data.
- Model and losses: point order, the MapTR parts, training the backbone.
- Measuring: how to score a run, compare two runs and trust a gap.
- Lidar: what the laser data can tell, and how it is checked and fed in.
- Engineering pitfalls: bugs and traps outside the model itself.

## Labels and data

### The photos are from 2025, the labels from 2022

- The crops are 2025 photos. Stored crops differ from a fresh `2025_ortho25`
  download by 5.6 to 13 mean pixel value, against 21 to 41 for
  `2024_ortho25`.
- Top10NL2023 was revised on 2022 photos ("herzien op basis van luchtfoto
  2022", `BRT_Actualiteitskaart_april_2023.pdf`).

Both from `exps/probe_image_label_dates.py`. So the labels are three years
older than the photos.

Do not read the age of a label from the `bronactual` field. It sits in 2014 or
2015 for 75.3% of the 62,415 hedge features, but it is the date of the source
used for the last edit, not of the last check. Top10NL is revised for the whole
country every year, and a feature is only edited when a change is seen. An old
date means the line was drawn with 2014 accuracy, not that nobody looked at it
since.

The cheap fix is the 2025 labels. Top10NL2025 was revised on 2024 and 2025
photos, so it matches the existing crops and needs no new imagery (TODO 3 in
`docs/status.md`).

### Top10NL is incomplete by its own rules, so precision is a lower bound

The Top10NL specification (https://kadaster.github.io/imbrt/, extracted into
`/home/fatemeh/Downloads/hedge/Top10NL/Top10NL Metadata.docx`) marks both
heg and bomenrij "Volledigheid: Beperkt", limited completeness. By rule it
leaves out:

- hedges inside built-up areas, and between tennis courts
- hedges on or around an erf, the plot of a house or farm, unless they
  continue past it

The minimum length is 100 m, but 18.0% of heg features (11,208 of 62,415) and
16.9% of bomenrij features (45,028 of 266,783) are shorter, because pieces
split by a road or a watercourse are exempt. Measured with `geometry.length`
on the shapefiles in EPSG:28992, where the unit is metres.

So the model is penalised for woody lines the map leaves out on purpose, and
the reported precision is a lower bound. Do not go one step further and say
this is why built-up and campsite crops score worst. That is consistent, not
shown. Testing it means checking whether the unmatched lines there lie on
woody lines the rules exclude, the way `exps/probe_treeline_overlap.py` did for
tree rows.

One label artifact turned out not to matter: a hedge stored as several
overlapping lines. Merging labelled lines within 10 m of each other removes
0.8% of them and moves F1 by 0.001 (`merge_gt` in
`exps/probe_polyline_pr.py`). The length metric is immune to it anyway.
Measure an artifact before designing around it.

### A quarter of the false positives are tree rows

Training uses only the hedge layer, `inrichtingselementen_lijn_heg`. Tree rows
are a separate layer, `inrichtingselementen_lijn_bomenrij`, with 266,783
features against 62,415 hedges. From above a tree row looks much like a hedge,
so the model draws it, and every metre counts as wrong.

On 300 random val crops, 24.3% of the drawn length that matches no label lies
within 10 m of a tree row, and only 1.2% near another hedge
(`exps/probe_treeline_overlap.py`). As a check, 100% of the training labels
lie on the hedge layer.

What that is worth, for exp 2 at t=0.95: precision at 10 m is 0.783, so 0.217
of the drawn length counts as wrong. A quarter of that, 0.243 x 0.217 = 0.053,
lies on tree rows. Without it, precision would be about 0.836. Exp 5 tried to
win this by adding tree rows as a second class and did not: hedge precision
went to 0.777 (`exps/probe_polyline_pr.py`).

Rank a fix by its share of the total, not by how bad the worst examples look.
The worst crops were campsites (next section), but excluding them moves F1 by
about 0.01, while tree rows hold a quarter of all unmatched length.

### The worst crops were campsites, and filtering them out is not worth it

Ranking val crops by wrongly drawn length put the same scene on top every
time: rows of identical chalets on small plots, each plot ringed by a clipped
hedge. It looks like housing, but Top10NL has almost no buildings there,
because a chalet is not a registered building. So a built-up or building filter
misses them. The layer that finds them is `functioneel_gebied_vlak`, field
`typefunctioneelgebied`, values camping, vakantiepark, bungalowpark,
caravanpark.

These crops are 315 of 3,098 (10.2%). They really are harder, F1 0.542 against
0.652 for the other crops, but excluding them moves the overall F1 only from
0.640 to 0.652 (exp 1, 10 m, `exps/probe_recreation_crops.py` and
`exps/probe_polyline_pr.py`).

Two lessons. A crop that looks urban may be a campsite, and the layer that
names a land use is not always the obvious one. And test an exclusion at
scoring time, by leaving crops out of the score, before changing the dataset.

### Tree rows are free to add as a second class, and do not help the hedges

Adding Top10NL bomenrij as class 1 needs no new imagery. The converter clips
the layer to each existing crop with the same function that made the hedge
labels (`scripts/data/convert_pdok_polylines_to_detr_polyline.py`), in about a
minute for 30,000 crops.

|                       | hedges only | with tree rows   |
| --------------------- | ----------- | ---------------- |
| lines                 | 103,432     | 103,432 + 44,435 |
| lines per crop, mean  | 3.45        | 4.93             |
| max lines per crop    | 49          | 49               |
| crops with a tree row | 0           | 18,061 (60.2%)   |

All from `exps/probe_tree_dataset_stats.py`.

- The query budget does not change. The busiest crop already had 49 lines, so
  60 queries still suffice.
- The hedge labels are untouched: the class 0 arrays are bit-identical to
  `pdok_dataset3_polylines` on all 30,000 crops.
- The two classes blur at 10 m. 12.2% of tree row length lies within 10 m of
  the hedge layer, and 11.0% of hedge length within 10 m of the tree row layer,
  against 2.4% and 1.7% at 5 m (`exps/probe_tree_class_labels.py`). So the 5 m
  score shows the class separation more cleanly.
- Take the tree rows from the same Top10NL year as the hedges, 2023, or the
  run changes two things at once.

As a second class they did not help the hedges: hedge F1 at 10 m 0.689 against
0.699, a gap about the size of the luck of a new random start (exps 5 and 7 in
`docs/status.md`).

### One woody line class gives a better woody map, not a better hedge map

Exp 10 trains hedges and tree rows as one class, on the exp 9 recipe
(`merge_classes=True` in `scripts/train_detr_unet_polyline.py`). As a woody
line detector it is the best model, F1 0.722 at 10 m (results in
`docs/status.md`).

It does not find more hedges. Its hedge recall rose from 0.733 to 0.795, but
it draws 5.76 lines per crop against 3.33, and the null model takes all of that
back: skill 0.483 against 0.473 (`exps/probe_recall_null_model.py`, see "Most
of the recall at a low threshold is clutter" under Measuring).

Against two classes (exp 5), exp 10 is 0.13 more precise at the same recall,
0.76. Part of that comes from the frozen backbone of exp 5, which was worth
0.022 on hedges, so the class choice itself is worth nearer 0.05. That rests on
one run each.

So the choice is about which map we want, not about a better model: exp 10 for
woody lines, exp 9 for hedges. Woody recall also barely falls on crowded crops,
but no null model was run per crowding bucket, so that is only a hint (TODO 4
in `docs/status.md`).

### More data buys precision, not recall

Exp 2 is exp 1 with all 26,902 train crops instead of 5,000, nothing else
changed. F1 at 10 m went from 0.640 to 0.685, both at t=0.95. Precision rose
about 0.08 in every crowding bucket. Recall rose about 0.02, and not at all on
the most crowded crops, 0.372 on crops with 7 or more hedges
(`exps/probe_polyline_pr.py`). So more data makes the model surer of the lines
it draws, but it does not find more lines in a busy crop.

Overfitting did vanish. The gap between train and eval loss went from 0.29 to
0.03, and the eval loss still fell at the last epoch (training logs of exps 1
and 2, in `docs/experiments_log.md`).

### The split is part of the run

Score a checkpoint only on the split it trained against. A conversion from
other labels gives another split, even with the same seed and blocks, because
`avoid_label_dirs` keeps different crops out of val. Once, such a conversion
had about 46% of its val crops in a run's training set (counts from
`exps/dataset_stats.py`), so the run was shown its own training images.

Pin a new dataset to an existing run with `val_stems_file` in
`scripts/data/convert_pdok_polylines_to_detr_polyline.py`.
`exps/probe_split_reproduces.py` checks that the split still comes out the
same.

## Model and losses

### Only the ordered point loss teaches point order

The model predicts 20 points per line. Only the ordered point loss (point L1)
checks each point by its number: point 0 against target 0, point 1 against
target 1, and so on. The other geometric losses ignore the order, so a zigzag
through the right place pleases them as much as a clean line.

| loss                      | what it checks                     | effect on a zigzag                                                                                               |
| ------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ordered L1 (point)        | each point goes to its own target  | the only loss that sees the order                                                                                |
| box (L1, gIoU)            | the rectangle around the points    | any shape that fills the same rectangle scores the same. The main cause: removing it helped.                     |
| length                    | the spacing of neighbouring points | a zigzag with the right spacing scores zero. Early in training it pushes points in random directions.            |
| direction (1 - cos angle) | the angle of each piece            | fully pleased only by a straight line, so it does not hold a zigzag. Not a main cause: changing it did not help. |

The ordered loss could not win against the others, for two reasons:

- A line can be read from either end, so the loss compares the prediction with
  whichever reading is closer. A zigzag sits evenly between the two readings,
  so the choice keeps flipping. Each flip sends a point's target to the other
  end of the line, and the points drift to the middle.
- When two points nearly coincide, the direction loss gradient becomes very
  large. Gradient clipping then shrinks every gradient, the ordered loss
  included.

The rule: box, length and direction may only be small helpers on top of a
strong ordered loss. All three are set to 0 now, so training uses the class
loss, the ordered point L1 and the auxiliary losses of the decoder layers.
MapTR does the same: class loss, ordered point loss at weight 5.0, direction at
0.005, no box and no length loss.

A zigzag in the first epochs that clears up later is normal. All 20 points of
a line start near one reference point, in no order, and the ordered loss and
the per-layer refinement then sort them. A zigzag is a problem only when a
loss that ignores the order is strong enough to hold it.

Ordered L1 also beats a tolerance loss, one that is zero within a band around
the target. With about 100,000 labelled lines (`exps/dataset_stats.py`) and
label errors that mostly point in no particular direction, L1 learns the
middle position, so random label errors average out. A tolerance band would
remove the gradient exactly where the points are being placed and ordered.
Label uncertainty is handled at scoring time instead, with tolerances of 5, 10
and 15 m.

### What makes MapTR-style point queries work

Two parts of MapTR, both in `scripts/train_detr_unet_polyline.py`:

- Each decoder layer refines the points rather than starting over. It starts
  from the current reference points and predicts a small correction. The query
  position is rebuilt from those points at every layer, so the attention knows
  where each point is now. This is the main reason the point queries learn
  fast.
- The query content is learned, not only the query position. Each query has a
  learned position part (`query_pos`) and a learned content part
  (`query_content`), split in `_build_hierarchical_queries`. Plain DETR starts
  the content at zero. The learned content may give each point query a hint of
  which point it is in its line.

### Training the backbone is the first clear gain

The backbone is the part that reads the photo. Until exp 9 it was frozen: it
had learned from 4,000 crops, for another task (masks), and never changed
again. Exp 9 lets it learn too, 10 times more slowly than the rest
(`backbone_lr=1e-5`, `build_optimizer` in
`scripts/train_detr_unet_polyline.py`).

Hedge F1 went from 0.699 to 0.721 at 10 m, and from 0.553 to 0.581 at 5 m
(t=0.90). That is more than twice the luck of a new random start.

- Lines lie closer to the hedges. The gain is biggest at 5 m.
- There are far fewer wrong lines.
- It does not find more hedges. Hedges the model missed before, it still
  misses.

The backbone really changed: 66 of its 86 weight tensors moved, against none
in exp 2 (`exps/probe_backbone_trained.py`). *(exp 9,
`exps/probe_polyline_pr.py`)*

## Measuring

### A score is meaningless without the threshold it was read at

Exp 2 is on record as F1 0.699 at 10 m and 0.547 at 5 m. Those are two different
thresholds: 0.699 is at t=0.90 and 0.547 is at t=0.95. At its own best threshold
exp 2 scores 0.553 at 5 m, not 0.547. Comparing exp 4 against the pair as if it
were one operating point makes the 5 m loss look like 0.014 when it is 0.020.

The optimum also moves with the model, so it has to be re-swept every time
(exp 1 peaked at 0.95, exps 2 and 4 at 0.90, exp 3 at 0.40). Quote the
threshold beside every number, and compare two runs either at one shared
threshold or each at its own optimum, never one of each.
*(`exps/probe_polyline_pr.py`)*

### How big a gap has to be before it counts

A score can move by luck. There are two kinds:

- **Which crops are in the val set.** If we pick the 3,098 crops again at
  random, a gap between two runs moves by about 0.005.
- **The random start of training.** Exp 7 is exp 2 run again, with only the
  random start (the seed) changed. It scores 0.008 lower at 10 m and 0.013
  lower at 5 m.

The second kind is bigger. So a gap of about 0.01 can appear with no real
change at all. We ask for 0.02 before we call a run better. That bar was a
guess at first, and exp 7 shows it is about right. Exp 9 (+0.023 at 10 m)
passes it; exps 4 to 6 do not. It rests on one repeat, so it is rough.
*(exps 2 and 7, `exps/probe_f1_gap_bootstrap.py`)*

### No cheap measure can rank two checkpoints, so build the real one

Cluster run 1, two checkpoints, same cluster-val crops at threshold 0.95. Three
cheap rankings gave three answers: eval loss picked `best_1.pt` (epoch 65),
mean lines per image picked `1_150.pt`, mean error per image picked `best_1.pt`
again. The mean count is the trap: per image `1_150.pt` predicts nothing on two
crops that have hedges and 16 lines on a crop that has 3, and those cancel.

Buffered length over all 3,098 val crops settled it (`exps/probe_polyline_pr.py`).
`1_150.pt` wins at every buffer, F1 0.640 against 0.614 at 10 m. So the checkpoint the eval loss calls
overfit is the better detector, and its advantage is recall.

Two rules follow. Do not report eval loss as a result, and do not early-stop on
it. Aggregate per image and then average, never as a ratio of two totals,
because errors of opposite sign cancel in a total.

### Chamfer plus Hungarian is the wrong metric for polylines

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

### Most of the recall at a low threshold is clutter, not lost hedges

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

| threshold | pred/img | real recall | null recall | skill     |
| --------- | -------- | ----------- | ----------- | --------- |
| 0.05      | 12.6     | 0.840       | 0.379       | 0.461     |
| 0.40      | 7.9      | 0.805       | 0.334       | 0.471     |
| 0.90      | 3.2      | 0.696       | 0.266       | **0.431** |
| 0.95      | 1.9      | 0.608       | 0.228       | 0.380     |

Nearly half of the celebrated 0.84 is what another crop's predictions would
score.

How to read the table. Exp 2 runs at t=0.90. The score head cannot draw a new
line, it only keeps or drops lines the model already drew, so the best it could
ever do is move exp 2 to another row of this table. Judged by real recall,
going from t=0.90 to t=0.05 gains 0.840 - 0.696 = 0.14. Judged by skill, going
from t=0.90 to t=0.40 gains 0.471 - 0.431 = 0.04. The 0.04 is the real gain.
The 0.14 counts luck as a gain.

What this means. The model did learn to find hedges. Shuffle the picture and
the same lines score 0.379 instead of 0.840, and that gap can only come from
having seen the image. What it did not do is find as many as it appeared to.
The hedges it misses are missing because it never drew a line there, not
because the score buried the line. So the score head is closed as a line of
work, worth 0.04, and everything after this targets perception: draw more of
the right lines in the first place.

Two rules. Any metric that rewards drawing more lines needs a null model before
it is used to justify a run. And a threshold sweep is not evidence of headroom:
falling precision as the threshold drops is the same fact as rising recall.

The threshold is still not a constant. It was 0.95 for exp 1, 0.90 for exp 2
and 0.40 for exp 3. Re-sweep after any change and record the value.

### The score ranks lines by how straight they are, not by whether they are right

Measured on exp 2 `best_2.pt`, all 3,098 val crops from the t=0.05 run. A
prediction counts as correct if at least 80% of its length is within 10 m of a
label. 7,956 of 39,104 predictions are correct.

| straightness      | median score, correct | median score, wrong |
| ----------------- | --------------------- | ------------------- |
| < 0.85 (bent)     | **0.453**             | 0.216               |
| 0.85 to 0.95      | 0.883                 | 0.690               |
| 0.95 to 0.99      | 0.951                 | 0.868               |
| > 0.99 (straight) | 0.974                 | **0.957**           |

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

### Exp 3: the focal head made it worse, and that was informative

Exp 2 with the softmax class head swapped for a focal sigmoid, MapTR weights,
nothing else changed. Best F1 at 10 m fell from **0.699 to 0.664**.

Each run at its own best threshold, so neither is handicapped. Exp 3's better
checkpoint is the last epoch (`3.pt`, F1 0.664); `best_3.pt` gives 0.660.

|                             | exp 2 `best_2.pt`   | exp 3 `3.pt`    |
| --------------------------- | ------------------- | --------------- |
| best F1 at 10 m             | **0.699** at t=0.90 | 0.664 at t=0.40 |
| P / R there                 | 0.701 / 0.696       | 0.779 / 0.578   |
| pred per image there        | 3.2                 | 1.6             |
| recall at t=0.05            | 0.840               | 0.845           |
| score AUC, correct vs wrong | 0.767               | 0.739           |
| same, inside the bent group | 0.618               | 0.642           |

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

## Lidar

### Check an overlay with a shift test, not with your eyes

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

### Why the lidar is not upsampled to the image size

The obvious way to add lidar is to blow the 25x25 patch up to 1000x1000, then
flip, rotate and pad it exactly like the image. Same code, nothing new to think
about. It was not done that way, for one reason:

| how the lidar is carried                 | size per crop | per batch of 16 |
| ---------------------------------------- | ------------- | --------------- |
| upsampled to the padded image, 1024x1024 | 28.7 MB       | **470 MB**      |
| on the token grid, 64x64                 | 112 KB        | 1.8 MB          |

470 MB per batch would go through the DataLoader, which sends every batch
between worker processes, and it buys nothing. Upsampling adds no information:
one lidar cell is 40x40 image pixels, so each value would simply be stored
1,600 times. The model then reduces it to the token grid anyway, so the whole
round trip is up and straight back down.

The cost of not doing it is one function, `_lidar_on_padded_grid`. It samples
each output cell by where its centre falls rather than resizing, which is what
keeps the mapping exact: 1024/16 is 64 output cells over a 25 cell patch that
covers only 1000 of the 1024 pixels, so no whole-number scale factor exists.

One simpler option does exist and was rejected: pad the patch to 26x26, since
26 cells of 40 pixels covers 1040 and the canvas is 1024. That is exact too and
needs no sampling, but it leaves the model to work out that the last cell is
mostly padding. Doing it in the dataset keeps that knowledge in one place.

### Test an augmentation by making two code paths agree

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

|             | under hedges | elsewhere | gap        |
| ----------- | ------------ | --------- | ---------- |
| augment off | 0.924        | 0.467     | +0.457     |
| augment on  | 0.921        | 0.467     | **+0.453** |

The gap survives, so both paths agree.

Then break it on purpose. Take out the lidar rotation and the gap falls to
**+0.187**. That is the important step. Without it, the test might be passing
because it cannot fail.

`augmentation_test` in `exps/probe_lidar_crop_alignment.py`.

### Nodata in the AHN4 metrics means no vegetation, not missing data

24 of the 25 metrics are computed from vegetation returns only, so a cell with
nothing woody in it has no value at all.

|                              | nodata share |
| ---------------------------- | ------------ |
| all cells                    | 46.2%        |
| cells a hedge passes through | **6.7%**     |
| every other cell             | 49.6%        |

So an empty cell is telling you something, not hiding something. Two things
follow. Filling the metrics with 0 is right rather than a fudge, because no
vegetation is zero density and ground-level height. And the presence channel is
not bookkeeping. It is probably the most useful of the seven channels, since
"is anything growing here" is most of what a 10 m grid can say about a 3 m
hedge.

`exps/probe_lidar_nodata_vs_hedge.py`.

### Read the label definition before choosing a feature

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

### LiDAR structure does separate hedges from tree rows

All 25 AHN4 metrics, 2,000 features per layer, split by 5 km geographic blocks
so nearby features cannot straddle train and test:

| features used                      | balanced accuracy | AUC       |
| ---------------------------------- | ----------------- | --------- |
| height p95 only, 1 metric          | 0.601             | 0.649     |
| all **7** height metrics           | 0.709             | 0.771     |
| the other **18**, no height at all | **0.765**         | **0.844** |
| all 25, gradient boosting          | **0.778**         | **0.852** |

(7 height metrics, not 8, as an earlier version of this table said. The paper
lists max, mean, median and the 25th, 50th, 75th and 95th percentiles, and
median and perc_50 are the same quantity in two identical rasters, so there are
really only 6 distinct ones.)

Structure alone beats every height metric put together. The best single metric
is the share of vegetation returns between 1 and 2 m, which is a literal
measurement of the rule above:

| metric                 | heg   | bomenrij | best single cut |
| ---------------------- | ----- | -------- | --------------- |
| BR_1_2 (returns 1-2 m) | 0.079 | 0.008    | 0.739           |
| BR_2_3 (returns 2-3 m) | 0.066 | 0.007    | 0.701           |
| BR_above_3             | 0.686 | 0.933    | 0.696           |
| height p95             | 8.0 m | 12.1 m   | 0.635           |

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
10 m a 250 m crop is only 25x25 cells, so the natural role is a side branch,
which is what exp 4 does. Whether it may also touch the geometry is decided in
"Where the lidar is allowed to act" below.

Reproduce with `exps/probe_lidar_hedge_vs_tree.py`.

### Where the lidar is allowed to act

The backbone turns the image into a 64x64 grid of feature vectors, one per
token, and the dataset puts the lidar patch on that same grid. So the lidar can
enter in two places:

- **In the tokens.** Add a learned lidar vector to each image token. Everything
  downstream sees image and lidar mixed, so the lidar can change the class and
  the point positions.
- **In the class head only.** Sample the lidar under each predicted line and
  give it only to the head that says hedge or tree. It can rename a line but
  never move one.

Exp 4 uses the first, although an earlier note here argued for the second on the
grounds that a 10 m grid could blur a thin line. Three reasons:

- It is far less code: an addition after `input_proj`, against sampling at each
  predicted point plus a wider head.
- `LidarTokenEncoder`'s last layer is zero-initialised, so at step 0 the model
  is exactly the no-lidar model. The lidar only moves the geometry if the
  gradient says that helps, and a lidar run and a no-lidar run start from the
  same place.
- The limit is that too few correct lines are drawn at all. A class-only branch
  cannot change which lines are drawn, so it cannot touch that.

**The old worry looked right, but the runs cannot show it.** Before exp 4 we
wrote down what a blurring lidar would look like: the 5 m score falls while
the 10 m score holds. Exp 4 showed exactly that (5 m 0.533 against 0.553).
But exp 5, without lidar, fell just as far (0.536), so the lidar is not the
cause. And a new random start alone moves the 5 m score by 0.013 (exp 7), so
the drop may be mostly luck.

The lidar helps hedges very little: +0.009 at 10 m with tree rows (exp 4
against exp 5), +0.004 without (exp 6 against exp 2). Both are below the 0.02
we need. What it does every time is remove some wrong lines. It helps the
tree row class more. So it is not worth moving it to the class head, and it
is not a hedge win either.

The lesson: a result that matches a prediction is still not proof when the
run changed two things at once, or when the gap is as small as the luck.
*(exps 4 to 7, `exps/probe_polyline_pr.py`)*

Two details that are not cosmetic. BatchNorm comes first because the six metrics
are in different units, five band ratios in [0, 1] against `perc_95` reaching
56 m with a standard deviation of 6, so without it one channel dominates the
sum. And `lidar_stride` has to equal the feature stride, or the lidar grid and
the token grid are different sizes; `main` asserts it.

## Engineering pitfalls

### A closed ring can have a near-duplicate point right after the start too

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

### Length filters must be checked on the stored polyline

The converter first filters raw polylines by length, then resamples them to
20 equidistant points. `resample_polyline_equidistant` rounds coordinates to
0.1 px, so a polyline sitting right on the threshold (40.006 px) can end up a
hair below it (39.9997 px) after resampling. Two polylines in pdok_dataset3
did exactly this and were stored despite being under the threshold.

Fix: re-check the length after resampling, since the resampled polyline is
what training actually sees (`scripts/data/convert_pdok_polylines_to_detr_polyline.py`). Border clipping was verified not to change
lengths (0 of 13,416 polylines), so the check after resampling is enough.

### The largest batch that fits is not the batch size to use

Probed on an A100-40GB with `exps/probe_batch_size.py` (real train step:
forward, loss, backward, optimizer, since backward dominates peak memory):

| batch | peak GB | s/step | s/image |
| ----- | ------- | ------ | ------- |
| 4     | 1.55    | 0.123  | 0.0307  |
| 16    | 5.76    | 0.445  | 0.0278  |
| 32    | 11.41   | 0.861  | 0.0269  |
| 64    | 22.67   | 1.678  | 0.0262  |

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

### A slurm log begins with the source code, so do not grep the whole file

The slurm script cats `paths.py`, the data scripts and the whole training
script into the `.out` file, so the log documents the run on its own. That means
the first few thousand lines are Python, not output. Grepping the file for
`Error` or `Traceback` matches error-handling code in the source and reports a
failure on a healthy job. It happened on exp 5, on a run that was training fine.

Cut to the run output first, then search:

```
sed -n '/^start training/,$p' <n>_<jobid>.out | grep ...
```

The same trap hides real information. A config that is off shows up in the
dumped source as well as in the output, so a grep for a setting returns both
and the source copy is the one that is always there.

### A frozen run with the GPU idle is a DataLoader worker hang

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

### The "nothing found" branch is the one that breaks

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

At threshold 0.5 every crop had a prediction, so the branch never ran. The empty
branch is exercised by running `mode="infer"` at a threshold above every score.
When a change makes empty results possible, look there first.

### A GPU run does not come back bitwise months later

Re-inferring exp 2 with the current code gave the same number of predictions on
every crop, but coordinates up to 0.19 px different from the run stored in July,
which is 4.7 cm on the ground. That looks like a regression and is not one.

The code is equivalent. With lidar off, the new model produces bitwise identical
logits, polylines and boxes to the version at the previous commit, on CPU, with
the real best_2.pt weights. The GPU is deterministic within a session, two runs
came out bitwise equal, but GPU and CPU differ by 0.64 px on the same weights
and input, because cuDNN TF32 is on. A driver or library change since July is
enough to move the last bits.

So to check that a refactor left a model alone, run the two code paths on CPU
with the same weights and compare the tensors
(`exps/probe_refactor_equivalence.py`, which imports the old script straight
from a git revision). Comparing NPZs from two GPU runs on different days
answers a different question.

### A reference test must be rebuilt when the output changes on purpose

`tests/test_training_data.py` compares 10 freshly built lidar crops with a
saved copy, `test_mini_gt`, made on 2026-03-03. Commits from 2026-03-11 to
2026-03-18 changed the labels on purpose: points are floored instead of
rounded, lines under 10 px are dropped, and `n_lines` counts stored lines (31
became 19 on one crop). The copy was not rebuilt, so the test failed from then
on. The March code still reproduces the copy exactly, so those commits were
the only cause.

The test matters more than its name says. `make_polylines_for_chip` in
`hedge_seg/pdok_training_data.py` makes every hedge and tree row label with the
same helpers. Rebuild the copy in the commit that changes the output:
`PYTHONPATH=. python tests/test_training_data.py`.

OSM chips had a second, silent fault. The OpenStreetMap tile server blocks
contextily's default User-Agent, the name a program sends to say who it is
(`contextily-<random hex>`, which names no app). It still answers HTTP 200,
with an image that says "Access blocked", so nothing raised and OSM chips
built after the block were the notice. The same tile asked for under a name for
this project comes back as the map, so the block is on the name, not the address.
Fixed: `save_osm_chip` in `hedge_seg/training_data.py` now sends `OSM_USER_AGENT`.
No account is needed. *(`exps/probe_osm_user_agent.py`)*

OSM stays off in the test anyway. Volunteers edit the map every day, so tiles
drift away from any saved copy, and the test would need the network.
