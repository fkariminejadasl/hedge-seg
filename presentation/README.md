# Presentation figures

The talk itself is `presentation.md`, and `update.md` is the meeting-update
deck. This file is only about the figures: where they are, what each set
shows, and how to remake them. Both decks render the same way.

Figures live in `/home/fatemeh/Downloads/hedge/screenshots/`, named

    detr_unet_polyline_<set>_<what>_val_cluster_t.90.png

`<what>` is `gt` (the reference map) or `best_2` (the current model). Six sets,
two figures each, 16 crops per figure, drawn at score threshold **0.90**, which
is exp 2's own operating point and the one the reported numbers use.

| set | what it shows | list it comes from |
|---|---|---|
| `good` | 500 to 900 m of hedge per crop, nearly all found | `best_*.txt` |
| `worst_fn` | the hedges the model misses | `worst_fn_*.txt` |
| `worst_fp` | lines drawn that are not labelled hedges | `worst_fp_*.txt` |
| `missing_labels` | extra lines that look like real hedges the map lacks | `missing_labels_*.txt` |
| `trees` | extra lines that are mapped tree rows | `treeline_crops.txt` |
| `recreation` | campsites and holiday parks, the hardest crops | `recreation_crops_val_cluster.txt` |

The lists are written next to the dataset, in
`pdok_dataset3_polylines/`, one crop id per line. Nothing is hand-picked.

## Remaking them

### `good`, `worst_fn`, `worst_fp`, `missing_labels`

1. `scripts/train_detr_unet_polyline.py` with `mode="infer"` and
   `infer_score_thresh=0.05`. Infer once at 0.05; every higher threshold is
   then just a filter, so no second inference run is ever needed.
2. `exps/probe_worst_crops.py` writes all four lists.
3. In `scripts/show_polyline_results.py`: paste a list into `ids`, set `tag` to
   the set name, `score_thresh=0.90`, `save=True`, and run.

### `trees`

Same, but step 2 is `exps/probe_treeline_overlap.py`, which writes
`treeline_crops.txt`.

### `recreation`

Same, but step 2 is `exps/probe_recreation_crops.py`, which writes
`recreation_crops_val_cluster.txt`. That file holds `pos_XXXXXX` stems rather
than bare ids, so drop the `pos_` prefix when pasting.

### Gotchas

- `tag` is what puts `worst_fn` into the file name. Without it every set writes
  to the same name and overwrites the last one.
- `run_dirs` should list **only** `best_2` for the talk. A second run directory
  adds a third figure per set.
- `score_thresh` must match whatever the talk claims. 0.90 is exp 2's best;
  0.95 is exp 1's.
- `n=16` gives the 4x4 grid. `seed` is ignored when `ids` is set.

## How to present this

`presentation.md` is written for **Marp**, which turns markdown into slides.
`---` starts a new slide and `![h:520](...)` sets image height in pixels. It is
not installed yet.

Easiest, inside VS Code:

1. Install the extension **Marp for VS Code** (`marp-team.marp-vscode`).
2. Open `presentation.md`. The preview pane now shows slides.
3. `Ctrl+Shift+P` then **Marp: Export Slide Deck** gives PDF, **PPTX** or HTML.

In Ubuntu’s Document Viewer (Evince), Press F5 to start presentation mode.

The PPTX export means you can keep writing markdown and still hand over a
PowerPoint file. Images are embedded on export, so the absolute paths in the
markdown do not have to travel with it.

From the command line instead:

    npm install -g @marp-team/marp-cli
    marp presentation.md --pptx      # or --pdf, or --html

Alternatives, if Marp does not suit: `pandoc -t pptx` (simpler, less control
over layout) or Quarto (`quarto render`, heavier but handles figures and
citations well). Neither is installed here either.
