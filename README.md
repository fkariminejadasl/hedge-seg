# hedge-seg

Draw hedgerows as lines from aerial photos of the Netherlands.

The model reads a 250 m x 250 m crop of a 25 cm per pixel aerial photo and
draws each hedgerow as a line of 20 points. A line, unlike a coloured mask,
has a length, a direction and two ends, which is what a landscape map needs:
hedgerow length per parcel, connectivity, change over time.

## Model and data

- Model: a ResNet18-UNet that first learned to colour hedge pixels, followed by
  a DETR-style head that draws the lines, with the MapTR parts (one query per
  line and per point, per-layer refinement of the points). All in
  `scripts/train_detr_unet_polyline.py`.
- Images: PDOK aerial photos (Beeldmateriaal Nederland), 30,000 crops.
- Labels: the hedge layer of Top10NL (Kadaster), and its tree row layer for
  the woody line model.
- Train and val are split by location, because the crops overlap.

## Result

The best hedge model, exp 9, on 3,098 crops in areas it never saw, measured by
line length within 10 m of a mapped hedge: precision 0.71, recall 0.73, F1
0.72, at score cut-off 0.90 (`exps/probe_polyline_pr.py`). All runs and how
they are measured: `docs/status.md`.

## Install

```
conda create -n hedge python=3.15
conda activate hedge
pip install -e ".[test]"
```

The `hf` extra adds `albumentations`, needed only by the old
`scripts/train_detr_hf.py`: `pip install -e ".[hf]"`. `hedge_seg/paths.py`
finds the data and experiment folders on the laptop and on the Snellius
cluster, so scripts need no path edits.

## Use

Each script is configured in the cfg block at its bottom: edit it and run.

```
python scripts/train_detr_unet_polyline.py   # mode: train, infer or preview
python scripts/show_polyline_results.py      # draw inference runs on the photos
python exps/probe_polyline_pr.py             # score inference runs
```

The datasets are built by `scripts/data/build_pdok_wms_dataset.py` and
`scripts/data/convert_pdok_polylines_to_detr_polyline.py`.

## Documentation

- `docs/status.md`: where the project stands, the results and the next steps.
- `docs/descriptions.md`: every script and dataset.
- `docs/lesson_learned.md`: what we learned and why.
- `docs/experiments_log.md`: notes on each run, and how each dataset was made.
- `docs/workflow.md`: copying between machines, inference and figures.
- `docs/literature.md`: papers and data sources.
- `presentation/presentation.md`: the talk, for ecologists.
