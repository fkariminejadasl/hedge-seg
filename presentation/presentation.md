---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- Slides. See README.md for where the figures come from and how to remake
them, including which probe script writes each crop list (probe_worst_crops.py
for worst_fn / worst_fp / missing_labels, probe_recreation_crops.py for
recreation). Captions cite the source of the number, not of the crops.
Render with Marp: see "How to present this" at the end of README.md.
Figures are at threshold 0.90, the operating point the reported numbers use.
Keep image height at 420 on the paired slides; taller pushes the caption off
the bottom of the slide, and so does a second caption line. -->

# Mapping hedgerows from aerial photos

Automatic delineation of hedgerows as **lines**, not coloured pixels

Aerial imagery at 25 cm per pixel, the Netherlands

---

## The result in one sentence

# We can now draw hedgerows automatically.

# About **70% of what we draw is right**, and we find about **70% of what is there**.

Measured by length, on areas the model has never seen, at a 10 m tolerance.

*(exp 9: precision 0.71, recall 0.73, `exps/probe_polyline_pr.py`)*

---

## Why lines and not a coloured mask

A **mask** tells you which pixels look like hedge.

A **line** has a length, a direction and two ends.

That is what goes into a landscape map: hedgerow length per parcel,
connectivity between patches, change over time.

Drawing lines is the harder task. Earlier attempts learned roughly *where
hedges tend to be*, but never *where a particular hedge was*.

---

## How to read the numbers

All scores are by **length**, on 3,098 crops in areas the model never saw.

- **Precision**: of the line length we draw, the share near a mapped hedge
- **Recall**: of the mapped hedge length, the share with a drawn line nearby
- **F1**: one number from the two, high only when both are high
- **Tolerance**: how near counts as near. 10 m unless a slide says 5 or 15 m
- **Cut-off**: each drawn line has a confidence score; we keep lines above
  0.90 unless a slide says otherwise

A difference under 0.02 is too small to trust *(exps 2 and 7)*.

*(`hedge_seg/metrics.py`, `exps/probe_polyline_pr.py`)*

---

## How good is it

The current best model, exp 9:

| tolerance | precision | recall | F1 |
|---|---|---|---|
| 5 m | 0.57 | 0.59 | 0.58 |
| **10 m** | **0.71** | **0.73** | **0.72** |
| 15 m | 0.76 | 0.79 | 0.78 |

Earlier box and mask models scored about 0.5 to 0.8 by their own tools' scores,
on an easier random split. Not comparable yet.

*(exp 9 at cut-off 0.90, `exps/probe_polyline_pr.py`; earlier models: `docs/descriptions.md`)*

---

## Where it works well

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_good_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_good_best_2_val_cluster_t.90.png) |

Crops with 500 to 900 m of hedge, in photos the model has never seen *(exp 2)*

---

## A quarter of our "mistakes" are not mistakes

The reference map keeps woody lines in **two separate layers**.

We trained on the **hedge** layer only.
**Tree rows** are a different layer the model never saw.

From above a tree row looks much like a hedgerow. The model draws it, and our
score counts every metre as an error.

**24.3%** of the length we are penalised for lands on a mapped tree row.

*(`exps/probe_treeline_overlap.py`, exp 2 at cut-off 0.95, 300 random test crops)*

---

## Tree rows: right model, incomplete labels

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_trees_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_trees_best_2_val_cluster_t.90.png) |

So we added tree rows to the labels: first as a second class, then as one class *(exp 2)*

---

## What actually makes a hedge a hedge

The reference map does not separate them by height. It separates them by
whether you can **see through at eye level**.

- **Tree row**: trees in a line, open underneath
- **Hedge**: trees or shrubs, closed underneath by undergrowth

So a tall line of trees is a hedge if it has undergrowth. We had assumed
height, but height separates them poorly.

*(Top10NL object definitions, https://kadaster.github.io/imbrt/)*

---

## Laser scanning measures exactly that

Airborne laser *(AHN4, 10 m grid)* counts how many returns come back from each
height layer.

| what we measure, median | hedge | tree row |
|---|---:|---:|
| returns between **1 and 2 m** | 7.9% | 0.8% |
| height (95th percentile) | 8.0 m | 12.1 m |

A hedge sends **ten times** as much back from eye level. Height separates
them far less.

*(`exps/probe_lidar_hedge_vs_tree.py`)*

---

## So laser data is worth adding

Train a classifier to say hedge or tree row from the laser data alone, and test
it on **regions it never saw**. 0.50 is a coin flip, 1.00 is perfect:

| what it is allowed to use | balanced accuracy |
|---|---:|
| one height measure (95th percentile) | 0.60 |
| all 7 height measures | 0.71 |
| the 18 non-height measures | 0.77 |
| all 25 measures | **0.78** |

The useful part is the layer-by-layer density, not the height.

*(`exps/probe_lidar_hedge_vs_tree.py`, balanced accuracy on held-out 5 km
blocks)*

---

## Six more runs, one change each

| what we changed | hedge F1 at 10 m | did it help? |
|---|---|---|
| nothing: our first good model | 0.70 | starting point |
| + tree rows + laser | 0.70 | no |
| + tree rows | 0.69 | no |
| + laser | 0.70 | no |
| nothing, only a new random start | 0.69 | shows the luck |
| train twice as long | 0.71 | a little |
| let the image part learn too | **0.72** | **yes** |

A gap below 0.02 is too small to trust: the same model trained twice already
differs by 0.01 F1. *(exps 2, 4 to 9, `exps/probe_polyline_pr.py`)*

---

## We tried both, and the hedges did not improve

Tree rows as a second class and the laser data, together in one run.

**Hedges did not move.** F1 0.699 to 0.698 at 10 m. At the tightest tolerance,
5 m, a little worse: 0.553 to 0.533.

**Tree rows:** the model now draws them as their own class, but misses many.
Our first score for them was too high, and is being fixed.

That 5 m drop looked like what we had predicted: a 10 m laser grid blurring a
3 m hedge.

*(exp 4, `exps/probe_polyline_pr.py`)*

---

## Tree rows and laser: no help for hedges

We then tried tree rows alone, and laser alone.

**Neither helps hedges.** All hedge F1 scores stay within 0.02 of the first
model at 10 m.

**The 5 m drop may be luck.** The same model, trained again from a new random
start, moved F1 by 0.013 at 5 m by itself.

A result that matches our prediction is still no proof when one run changes
two things at once.

*(exps 4 to 7, `exps/probe_polyline_pr.py`)*

---

## Luck, and training longer

**Luck.** We trained the first model again and changed only the random start.
F1 moved by 0.01 at 10 m. So we trust a gain only above 0.02 *(exp 7)*.

**Training twice as long:** F1 0.699 to 0.712 at 10 m. A little *(exp 8)*.

*(exps 7 and 8, `exps/probe_polyline_pr.py`)*

---

## Letting the image part learn helps

Until now the part that reads the photo was frozen. It learned from 4,000
crops, for masks. Now it trains too, 10 times more slowly than the rest
*(backbone learning rate 1e-5)*.

**The first clear gain.** F1 0.699 to 0.721 at 10 m, 0.553 to 0.581 at 5 m.

- Lines sit closer to the hedges, and there are far fewer wrong ones
- It does not find more hedges: what it missed before, it still misses

A new random start alone moves F1 by about 0.01. This gain is twice that or
more.

*(exp 9 against exps 2 and 7, `exps/probe_polyline_pr.py`)*

---

## Hedges and tree rows as one class

The map tells them apart by what grows underneath, which a photo from above
barely shows. So we stopped asking: one class, a *woody line*.

**The best woody line map so far.** Woody line F1 0.72 at 10 m, against 0.68
for the hedge-only model and 0.65 for the older model with two classes, each
at its best cut-off.

**Hedges: no better, no worse.** Hedge recall rises from 0.73 to 0.80, but only
because it draws more lines. Recall corrected for luck *(null model)*: +0.01.

*(exp 10 against exps 5 and 9, `exps/probe_polyline_pr.py`,
`exps/probe_recall_null_model.py`)*

---

## Hedges the map is missing entirely *(high recall, low precision)*

![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_missing_labels_gt_val_cluster_t.90.png) ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_missing_labels_best_2_val_cluster_t.90.png)

**The map is incomplete, so our 0.70 precision is a floor.** *(exp 2)*

---

## What we miss *(false negatives)*

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fn_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fn_best_2_val_cluster_t.90.png) |

Recall: 73% with 1 hedge, 37% with 7+ *(exp 2, cut-off 0.95, `probe_polyline_pr.py`)*

---

## Worst false alarms *(false positives)*

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fp_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fp_best_2_val_cluster_t.90.png) |

Built-up areas and holiday parks, where every plot has a clipped hedge. *(exp 2)*

---

## Campsites and holiday parks

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_recreation_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_recreation_best_2_val_cluster_t.90.png) |

Rows of chalets, each ringed by a hedge, 315 of 3,098 crops. *(`probe_recreation_crops.py`)*

---

## What we did: data

- Aerial photos, 25 cm per pixel, 30,000 crops, about 103,000 mapped hedge lines
- Train and test split **by geography**, never at random
  - the crops overlap, so a random split would have put 68.5% of the test
    crops on top of a training crop
- Cleaned labels: closed loops opened into lines, hedges under 10 m dropped

*(`exps/dataset_stats.py`, `exps/quantify_geographic_crop_overlap_pdok_dataset3.py`)*

---

## Know your reference map

Our labels are the national topographic map *(Top10NL, Kadaster)*.

**It is not a survey of every hedge.** It says so itself: completeness is
*"Beperkt"*, limited. By rule it leaves out

- hedges **inside built-up areas**
- hedges **on or around a property**, unless they carry on past it
- anything under about **100 m** long

So some of what we score as a false alarm is a real hedge the map is not meant
to contain. **Our precision is a floor, not a true error rate.**

How much of our false alarm rate this explains is not yet measured.

*(Top10NL collection criteria, https://kadaster.github.io/imbrt/)*

---

## Know your dates

| | year |
|---|---|
| our aerial photos | **2025** |
| the map we score against *(Top10NL 2023)* | drawn from **2022** photos |

Three years apart. Hedges are planted and removed in three years.

The map is redrawn every year for the whole country, so this is fixable: use
the 2025 map instead. Both are downloaded.

**The shape of the labels**, for reference: a typical mapped hedge line is 120 m
long (median), and about one in five bends noticeably *(`exps/probe_polyline_pr.py`)*

---

## What we did: model

- A **U-Net** first trained to colour hedge pixels, then reused to draw lines
  *(ResNet18-UNet)*. Frozen at first; letting it learn too gave the best model (exp 9)
- **One query per hedge, one per point along it**, so the model knows which
  point belongs to which hedge *(hierarchical queries, MapTR)*
- The starting guess for each hedge is **learned from the data**
  *(learned content queries)*
- Each layer **refines** its guess rather than starting over
  *(per-layer reference-point refinement)*

---

## What we did: evaluation

- Precision and recall by **length within a buffer**, per crop, then averaged
- **Automatic lists** of the worst misses, worst false alarms, and probable
  label errors, so failures are looked at rather than guessed at
- **Cross-checks** against the tree-row layer and the campsite layer

---

## What helped

- **Higher resolution.** On 25 cm aerial photos the model learned where each hedge is;
  on low-resolution height data it never did
- **A pretrained segmentation backbone.** Colour the pixels first, then draw
- **Removing losses, not adding them.** Extra geometric terms made the lines
  zigzag *(length, direction, smoothness losses, all set to zero)*
- **The MapTR pieces.** *Hierarchical queries, learned content queries,
  per-layer reference-point refinement*
- **More data.** 5,000 to 27,000 crops removed overfitting completely
  *(exp 1 to exp 2)*
- **Cleaner labels.** Loops opened, short lines dropped
- **Training the image backbone too.** F1 0.699 to 0.721 at 10 m *(exp 9)*
- **One class for hedges and tree rows**, for a woody line map: F1 0.72 *(exp 10)*

---

## One thing we tried, and it did not work

We rebuilt the confidence score, believing the model already found the hedges
and then threw them away *(focal sigmoid class head)*.

**It got worse**: hedge F1 0.70 to 0.66 at 10 m, each at its best cut-off *(exp 3)*.

Then we checked the belief and it was wrong: most of what we thought we had
found was clutter. Eleven hours of computing to learn that, and worth it, since
we were about to spend far more on the same idea.

---

## What is next

1. **Look at finer detail.** A hedge is about 3 m wide, 12 pixels. One grid
   cell is 4 m, 16 pixels. The hedge is thinner than a single cell, so its
   shape is gone before the head sees it *(feature stride 16 to 8)*
2. **Use the 2025 map**, so photos and labels are the same year
3. **Lower resolution**, to see how much depends on 25 cm imagery

---

## Backup: the runs, 1 to 5

<style scoped>table { font-size: 22px; }</style>

| exp | what changed | best cut-off | precision | recall | F1 |
|---|---|---|---|---|---|
| 1 | 5,000 crops, 150 epochs | 0.95 | 0.702 | 0.589 | 0.640 |
| 2 | all 26,902 crops, 45 epochs | 0.90 | 0.701 | 0.696 | 0.699 |
| 3 | focal sigmoid score head | 0.40 | 0.779 | 0.578 | 0.664 |
| 4 | tree rows as class 2, plus laser | 0.90 | 0.693 | 0.703 | 0.698 |
| 5 | tree rows as class 2, no laser | 0.90 | 0.692 | 0.686 | 0.689 |

At 10 m on the same 3,098 crops; gaps under 0.02 do not count *(`exps/probe_polyline_pr.py`)*

- **More data buys precision, not recall.** Precision +0.08 in every crowding
  bucket, recall +0.02, nothing on crowded crops *(exp 1 to exp 2)*

---

## Backup: the runs, 6 to 10

<style scoped>table { font-size: 22px; }</style>

| exp | what changed | best cut-off | precision | recall | F1 |
|---|---|---|---|---|---|
| 6 | laser, no tree rows | 0.90 | 0.698 | 0.709 | 0.703 |
| 7 | exp 2 again, new random start | 0.90 | 0.687 | 0.694 | 0.691 |
| 8 | exp 2 for 90 epochs | 0.90 | 0.709 | 0.715 | 0.712 |
| 9 | image backbone trained too | 0.90 | 0.710 | 0.733 | **0.721** |
| 10 | hedges and tree rows as one class | 0.90 | 0.683 | 0.765 | 0.722 *(woody lines)* |

At 10 m on the same 3,098 crops; gaps under 0.02 do not count *(`exps/probe_polyline_pr.py`)*

Exp 10 is scored on hedges and tree rows together, the others on hedges only.

---

## Backup: precision, recall and F1

Each cell: precision / recall / F1, at cut-off 0.90, on the same 3,098 held-out crops.

| what we changed | 5 m | 10 m | 15 m |
|---|---|---|---|
| our first good model | .549 / .557 / .553 | .701 / .696 / .699 | .767 / .767 / .767 |
| + tree rows + laser | .523 / .545 / .533 | .693 / .703 / .698 | .767 / .779 / .773 |
| + tree rows | .532 / .540 / .536 | .692 / .686 / .689 | .762 / .759 / .761 |
| + laser | .534 / .556 / .545 | .698 / .709 / .703 | .767 / .782 / .774 |
| new random start only | .530 / .550 / .540 | .687 / .694 / .691 | .756 / .766 / .761 |
| train twice as long | .563 / .575 / .569 | .709 / .715 / .712 | .769 / .781 / .775 |
| image part learns too | .569 / .593 / .581 | .710 / .733 / .721 | .764 / .794 / .779 |

*(exps 2, 4 to 9, `exps/probe_polyline_pr.py`)*

---

## Backup: measurement traps we hit

- **Half of the hedges we thought we found were never found.** At the low
  cut-off the model draws 13 lines per picture where 2 hedges exist. Score one
  picture's lines against a *different* picture's map and recall is still 0.38
  of the 0.84. Only 0.46 is real *(`exps/probe_recall_null_model.py`)*
- **How you measure changes the answer by half.** F1 is 0.41 with one-to-one
  matching *(chamfer)* and 0.64 by length, on identical predictions
  *(exp 1 at cut-off 0.95, `exps/probe_polyline_pr.py`)*
- **Validation loss cannot pick the best model.** It chose epoch 65; the metric
  and the pictures both prefer epoch 150 *(exp 1)*
