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


## Lessons

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

## TODOs

Phase A — data + split, one batch of small changes:

Converter: open closed rings, raise min_length_px to ~40 px (10 m).
Spatial split script from center_world (block split + drop residual overlaps + KDTree verification).
Training script: accept explicit split files instead of the internal random split.
Convert dataset3 (30k), spot-check ~20 overlays.
Phase B — baseline on a ~5k spatially-blocked subset of dataset3 (frozen backbone, augment on, scheduler enabled, eval_every=5), plus the buffered precision/recall metric so results are comparable to the YOLO-seg result (80% precision, 70% recall).

Phase C — decoupled self-attention as a clean A/B while the baseline trains (regression-tested with the one-image overfit).

Phase D — results-driven: full 30k run, backbone unfreezing with low lr, and tolerance-loss/border-filtering only if the error analysis points at them.

For the baseline, use a subset of dataset3 with the blocked split rather than dataset2: it is consistent with the eventual 30k run, and the backbone was trained on dataset2-derived data, so dataset2 val images also leak through the frozen backbone features. (Dataset3 likely overlaps dataset2 geographically too, but that is a second-order effect.)

Note: the spatial split will make the first val numbers look worse than a random split would have. That is expected; they are the real baseline to improve from.


