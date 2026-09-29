"""
How much does an F1 gap between two runs move just because of which crops are
in the val set?

Paired bootstrap: draw 3,098 val crops with replacement, score both runs on
the same draw, take the F1 difference, repeat 2,000 times. Both runs see the
same crops in every draw, so a crop that is hard for both does not count as a
difference.

This measures one source of noise only. The other, a different random start
for training, needs a second training run with another seed and is not
measured yet.

Result (2026-09-29, hedge against hedge, t=0.90):

    runs           buffer  gap     95% range         sd of the gap
    exp 6 - exp 2   5 m   -0.008   -0.014 .. -0.002  0.003
    exp 6 - exp 2  10 m   +0.004   -0.001 .. +0.010  0.003
    exp 6 - exp 2  15 m   +0.007   +0.002 .. +0.012  0.002
    exp 5 - exp 2   5 m   -0.017   -0.022 .. -0.011  0.003
    exp 5 - exp 2  10 m   -0.010   -0.015 .. -0.006  0.002
    exp 5 - exp 2  15 m   -0.006   -0.010 .. -0.002  0.002

- Picking the val crops again moves a gap by about 0.003 (one sd), so about
  0.005 either way covers 95% of draws.
- Exp 6 against exp 2 at 10 m could be zero even before seed noise: the range
  crosses 0.
- The tree rows really cost hedges at 5 m (exp 5, -0.017): the whole range is
  below zero, far outside this noise.
- The 0.02 bar used since exp 4 is a chosen bar, not a measured one. It is
  about seven times this noise. How much seed noise adds is still unknown.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_f1_gap_bootstrap.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_polyline_pr import load_run, score  # noqa: E402

from hedge_seg.metrics import f1  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402


def per_crop(run_dir, score_thresh):
    """Precision and recall per crop, hedge against hedge, keyed by stem."""
    rows = score(load_run(run_dir), score_thresh=score_thresh, class_id=0)
    return {r["stem"]: r for r in rows}


def main(cfg):
    rows = {
        name: per_crop(run, cfg["score_thresh"]) for name, run in cfg["runs"].items()
    }
    stems = sorted(next(iter(rows.values())))
    assert all(sorted(r) == stems for r in rows.values()), "runs differ in crops"

    rng = np.random.default_rng(cfg["seed"])
    draws = rng.integers(0, len(stems), size=(cfg["n_boot"], len(stems)))
    everything = np.arange(len(stems))
    print(f"{len(stems)} crops, t={cfg['score_thresh']}, {cfg['n_boot']} draws\n")

    for a, b in cfg["pairs"]:
        for buf in cfg["buffers"]:
            pa, ra, pb, rb = (
                np.array([rows[run][s][f"{k}{buf}"] for s in stems])
                for run, k in ((a, "p"), (a, "r"), (b, "p"), (b, "r"))
            )

            def gap(idx):
                return f1(pa[idx].mean(), ra[idx].mean()) - f1(
                    pb[idx].mean(), rb[idx].mean()
                )

            gaps = np.array([gap(idx) for idx in draws])
            lo, hi = np.percentile(gaps, [2.5, 97.5])
            print(
                f"{a} - {b} @{buf:>2}m: gap {gap(everything):+.3f}, "
                f"95% range {lo:+.3f} .. {hi:+.3f}, sd {gaps.std():.3f}"
            )


if __name__ == "__main__":
    hedge = DATA_ROOT / "pdok_dataset3_polylines/inference"
    tree = DATA_ROOT / "pdok_dataset3_tree_polylines/inference"
    cfg = dict(
        # All hold the same 3,098 val stems; t0.05 runs, so any threshold works.
        runs={
            "exp 2": hedge / "best_2_val_cluster_t0.05",
            "exp 5": tree / "best_5_val_t0.05",
            "exp 6": hedge / "best_6_val_cluster_t0.05",
        },
        pairs=[("exp 6", "exp 2"), ("exp 5", "exp 2")],
        buffers=(5, 10, 15),
        score_thresh=0.90,  # the best threshold of exps 2, 5 and 6
        n_boot=2000,
        seed=0,
    )
    main(cfg)
