"""
What did adding tree rows do to the dataset?

Compares the hedges-only polyline dataset with the one that also has Top10NL
tree rows as class 1, over the same crops. Three questions, each of which could
have stopped the run:

- Do the queries still fit? The model has num_polylines=60 instance queries. If
  the extra class pushed the busiest crop past that, the second class would cost
  hedge recall on exactly the crops where recall is already worst.
- Is the hedge half untouched? If the class 0 arrays are not identical to the
  old dataset, then a change in the hedge score could come from the data rather
  than from the model, and the comparison with exp 2 would mean nothing.
- How much did the dataset actually grow?

Result (2026-09-16, pdok_dataset3_polylines against pdok_dataset3_tree_polylines):

    crops                          30,000 in both, same stems
    lines                          103,432 -> 147,867 (44,435 tree rows)
    lines per crop, mean           3.45 -> 4.93
    lines per crop, max            49 -> 49
    crops with a tree row          18,061 (60.2%)
    crops over 60 lines            0
    hedge arrays identical         30,000 of 30,000

So the busiest crop was already at 49 lines before tree rows were added, and 60
queries are still enough. Nothing about the hedges changed.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_tree_dataset_stats.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402


def load_index(root: Path) -> dict:
    """{stem: path} over both splits, so the split itself does not matter here."""
    return {
        p.stem: p
        for split in ("train", "val")
        for p in (root / "polylines" / split).glob("*.npz")
    }


def main(cfg):
    old = load_index(Path(cfg["old_root"]))
    new = load_index(Path(cfg["new_root"]))
    print(f"crops: old {len(old)}, new {len(new)}, same stems: {set(old) == set(new)}")

    old_counts, new_counts, tree_counts = [], [], []
    n_identical = n_compared = 0
    for stem, path in sorted(new.items()):
        new_npz = np.load(path)
        labels = new_npz["labels"]
        polylines = new_npz["polylines"]
        new_counts.append(len(labels))
        tree_counts.append(int((labels == cfg["tree_class_id"]).sum()))

        if stem in old:
            old_polylines = np.load(old[stem])["polylines"]
            old_counts.append(len(old_polylines))
            hedges = polylines[labels == cfg["hedge_class_id"]]
            n_compared += 1
            n_identical += int(
                hedges.shape == old_polylines.shape
                and np.array_equal(hedges, old_polylines)
            )

    old_counts = np.asarray(old_counts)
    new_counts = np.asarray(new_counts)
    tree_counts = np.asarray(tree_counts)

    print(
        f"lines: {old_counts.sum()} -> {new_counts.sum()} ({tree_counts.sum()} trees)"
    )
    print(
        f"lines per crop, mean: {old_counts.mean():.2f} -> {new_counts.mean():.2f}, "
        f"max: {old_counts.max()} -> {new_counts.max()}"
    )
    with_tree = int((tree_counts > 0).sum())
    print(f"crops with a tree row: {with_tree} ({with_tree / len(new):.1%})")
    print(
        f"crops over {cfg['num_polylines']} lines (the query budget): "
        f"{int((new_counts > cfg['num_polylines']).sum())}"
    )
    print(f"hedge arrays identical to the old dataset: {n_identical} of {n_compared}")


if __name__ == "__main__":
    cfg = dict(
        old_root=DATA_ROOT / "pdok_dataset3_polylines",
        new_root=DATA_ROOT / "pdok_dataset3_tree_polylines",
        hedge_class_id=0,
        tree_class_id=1,
        num_polylines=60,  # instance queries in the training cfg
    )
    main(cfg)
