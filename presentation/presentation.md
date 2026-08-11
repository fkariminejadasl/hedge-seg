---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- Slides. See README.md for where the figures come from and how to remake
them. Render with Marp: see "How to present this" at the end of README.md.
Figures are at threshold 0.90, the operating point the reported numbers use.
Keep image height at 420 on the paired slides; taller pushes the caption off
the bottom of the slide. -->

# Mapping hedgerows from aerial photos

Automatic delineation of hedgerows as **lines**, not coloured pixels

Aerial imagery at 25 cm per pixel, the Netherlands

---

## The result in one sentence

# We can now draw hedgerows automatically.

# About **70% of what we draw is right**, and we find about **70% of what is there**.

Measured on areas the model has never seen, at a 10 m tolerance.

---

## Why lines and not a coloured mask

A **mask** tells you which pixels look like hedge.

A **line** has a length, a direction and two ends.

That is what goes into a landscape map: hedgerow length per parcel,
connectivity between patches, change over time.

Drawing lines is the harder task. Earlier attempts learned roughly *where
hedges tend to be*, but never *where a particular hedge was*.

---

## How good is it

Precision: of what we draw, how much is really a hedge.
Recall: of the hedges that exist, how many we find.
Both at a 10 m tolerance.

| method | output | precision | recall |
|---|---|---|---|
| box detection | boxes | 0.70 | 0.60 |
| pixel segmentation | mask | 0.60 | 0.40 |
| shape segmentation | blobs | 0.80 | 0.70 |
| **this model** | **lines** | **0.70** | **0.70** |

Other numbers came from a different measurement on an easier split.

---

## Where it works well

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_good_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_good_best_2_val_cluster_t.90.png) |

Crops with 500 to 900 m of hedge, in photos the model has never seen.

---

## A quarter of our "mistakes" are not mistakes

The reference map keeps woody lines in **two separate layers**.

We trained on the **hedge** layer only.
**Tree rows** are a different layer the model never saw.

From above a tree row looks much like a hedgerow. The model draws it, and our
score counts every metre as an error.

**24.3%** of the length we are penalised for lands on a mapped tree row.

---

## Tree rows: right model, incomplete labels

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_trees_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_trees_best_2_val_cluster_t.90.png) |

Adding tree rows as a second class is the obvious next data step.

---

## What actually makes a hedge a hedge

The reference map does not separate them by height. It separates them by
whether you can **see through at eye level**.

- **Tree row**: trees in a line, open underneath
- **Hedge**: trees or shrubs, closed underneath by undergrowth

So a tall line of trees is a hedge if it has undergrowth. We had assumed
height, and height cannot tell them apart.

---

## Laser scanning can tell them apart, once you measure the right thing

Airborne laser *(AHN4, 10 m grid)* counts how many returns come back from each
height layer.

| what we measure | hedge | tree row |
|---|---:|---:|
| returns between **1 and 2 m** | 7.9% | 0.8% |
| height (95th percentile) | 8.0 m | 12.1 m |

A hedge sends **ten times** as much back from eye level. Height barely
separates them at all.

Now train a classifier to say "hedge or tree row" from the laser data alone,
and test it on **regions it never saw**. A score of 0.50 is a coin flip and
1.00 is perfect:

| what it is allowed to use | score |
|---|---:|
| height only | 0.60 |
| the 18 non-height measures | 0.77 |
| all 25 measures | **0.78** |

Height barely beats a coin flip. Structure gets most of the way there.

So laser data is worth adding as a second input, and the useful part is the
layer-by-layer density, not the height. *(`exps/probe_lidar_hedge_vs_tree.py`)*

---

## Hedges the map is missing entirely (High R, low P)

![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_missing_labels_gt_val_cluster_t.90.png) ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_missing_labels_best_2_val_cluster_t.90.png)

Found automatically: **The map is incomplete, so true precision is above the 0.70 we report.**

---

## What we miss (FN)

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fn_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fn_best_2_val_cluster_t.90.png) |

Crowded scenes. One hedge in a crop: we find 73%. Seven or more: 37%.

---

## Worst false alarms (FP)

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fp_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_worst_fp_best_2_val_cluster_t.90.png) |

Built-up areas and holiday parks, where every small plot has a clipped hedge.

---

## Campsites and holiday parks

| reference map | model |
|:---:|:---:|
| ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_recreation_gt_val_cluster_t.90.png) | ![h:420](/home/fatemeh/Downloads/hedge/screenshots/detr_unet_polyline_recreation_best_2_val_cluster_t.90.png) |

Rows of chalets, each plot ringed by a hedge. 10% of the test area.

---

## What we did: data

- Aerial photos, 25 cm per pixel, 30,000 crops, about 103,000 hedgerows
- Train and test split **by geography**, never at random
  - the crops overlap, so a random split would have let the model see 68.5% of its own test area during training
- Cleaned labels: closed loops opened into lines, hedges under 10 m dropped

---

## Know your reference map

Our labels are the national topographic map *(Top10NL, Kadaster)*.

**It is not a survey of every hedge.** It says so itself: completeness is
*"Beperkt"*, limited. By rule it leaves out

- hedges **inside built-up areas**
- hedges **on a farmyard**, unless they carry on past the yard
- anything under about **100 m** long

So some of what we score as a false alarm is a real hedge the map is not meant
to contain. **Our precision is a floor, not a true error rate.**

---

## Know your dates

| | year |
|---|---|
| our aerial photos | **2025** |
| the map we score against *(Top10NL 2023)* | drawn from **2022** photos |

Three years apart. Hedges are planted and removed in three years.

The map is redrawn every year for the whole country, so this is fixable: use
the 2025 map instead. Both are downloaded.

**The shape of the labels**, for reference: an average hedge runs 120 m, and
one in four bends noticeably.

---

## What we did: model

- A **U-Net** first trained to colour hedge pixels, then **frozen** and reused
  to draw lines *(ResNet18-UNet, semantic-segmentation pretraining)*
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

- **Higher resolution.** 25 cm aerial beats low-resolution height data by a lot
- **A pretrained segmentation backbone.** Colour the pixels first, then draw
- **Removing losses, not adding them.** Extra geometric terms made the lines
  zigzag *(length, direction, smoothness losses, all set to zero)*
- **The MapTR pieces.** *Hierarchical queries, learned content queries,
  per-layer reference-point refinement*
- **More data.** 5,000 to 27,000 crops removed overfitting completely
- **Cleaner labels.** Loops opened, short lines dropped

---

## One thing we tried, and it did not work

We rebuilt the confidence score, believing the model already found the hedges
and then threw them away *(focal sigmoid class head)*.

**It got worse**, 0.70 to 0.66.

Then we checked the belief and it was wrong: most of what we thought we had
found was clutter. Eleven hours of computing to learn that, and worth it, since
we were about to spend far more on the same idea.

---

## What is next

1. **Look at finer detail.** The model works on a grid where a 3 m hedge is
   smaller than one cell. This is the real limit *(feature stride 16 to 8)*
2. **Tree rows as a second class**, with laser structure to tell them apart
3. **Use the 2025 map**, so photos and labels are the same year
4. **Train the image backbone**, which is frozen today
5. **Lower resolution**, to see how much depends on 25 cm imagery

---

## Backup: numbers

- Exp 2: 26,902 training crops, 45 epochs, 11 h on one A100
- F1 0.685 at 10 m, 0.750 at 15 m
- **More data buys precision, not recall.** Precision +0.08 in every crowding
  bucket, recall +0.02, nothing on crowded crops
- **Half of the "hedges we found but discarded" were never found.** At the low
  cut-off the model draws 13 lines per picture where 2 hedges exist. Score one
  picture's lines against a *different* picture's map and you still score 0.38
  of the 0.84. *(`exps/probe_recall_null_model.py`)*
- **The confidence score ranks lines by how straight they are.** A correct bent
  hedge scores 0.45, a wrong straight one 0.96
  *(`exps/probe_score_quality.py`)*
- **How you measure changes the answer by half.** Chamfer matching gives 0.41,
  buffered length 0.64, on identical predictions
- **Validation loss cannot pick the best model.** It chose epoch 65; the metric
  and the pictures both prefer epoch 150
