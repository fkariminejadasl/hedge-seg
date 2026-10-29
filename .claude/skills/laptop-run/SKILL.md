---
name: laptop-run
description: Run a training script locally on the laptop as a throwaway experiment. Use when the user wants to train on their own machine (not the cluster), e.g. "run it on my laptop", "train locally while the cluster is down".
---

# Run a training job on the laptop

For a throwaway or debugging run on the local machine. The laptop GPU
(RTX PRO 3000, 12.3 GB) is only good for smoke tests and small runs. Its val
split is leakier than the cluster's (avoid_label_dirs only sees the few local
pdok_dataset2 labels), so a laptop run's val numbers are not the honest
baseline. Use it to watch the learning curve, not to report a final number.

Paths come from `hedge_seg/paths.py` and resolve to the local machine
automatically. Always `conda activate hedge` first, never base.

## Config overrides (edit in the cfg, do not commit)

The committed config is the cluster baseline. For a laptop run, edit these in
the training script cfg and leave the change uncommitted:

- `exp="<n>_laptop"` so it never collides with the cluster's `<n>` and the
  leaky-val run is obvious from the name.
- `num_workers=4` (4 train + 4 eval = 8 processes) so the machine stays usable
  while you work. Do not raise it just because the laptop has more cores.
- `eval_every=10` and `n_val_subset=1000` to keep eval cheap. The leaky val is
  not the honest number anyway, so scoring all of it every few epochs is waste.

These are visible edits, not a hidden override. The committed script always
shows the value that runs.

## Steps

1. Apply the overrides above.
2. Launch in the background with `python -u` (without it the log block-buffers
   and looks stuck), logging into the experiment dir:

   ```
   LOGDIR=<save_path>/<exp>       # e.g. .../training/detr_unet_polyline/1_laptop
   mkdir -p "$LOGDIR"
   PYTHONPATH=. nohup python -u scripts/<script>.py > "$LOGDIR/train.log" 2>&1 &
   ```

3. Confirm it started: `grep -E "Dataset:|Traceback" "$LOGDIR/train.log"`.
4. Watch it: `tail -f "$LOGDIR/train.log"`, or tensorboard on `<save_path>/<exp>/tensorboard`.

## Gotchas learned the hard way

- The run is a local `nohup` process. It dies if the laptop sleeps or powers
  off. Keep the machine on. Unlike a cluster job, nothing survives a shutdown.
- Epoch time is CPU-bound (data loading + Hungarian matching), not GPU-bound.
  A GPU-only s/image measurement underestimates it by a lot. Extrapolate from
  the first ~10 real epochs instead.
- Before relaunching, delete the old `<save_path>/<exp>/` directory. Multiple
  runs writing the same tensorboard dir overlay their curves and pollute it.
- If the log freezes with the GPU at 0% but a process spinning at ~100% CPU,
  it is a DataLoader worker hang, not slow training. The fork start method
  (already set in train_detr_unet_polyline.py via `_mp_context`) avoids the
  Python 3.14 forkserver deadlock; if a script lacks it, that is the fix.
