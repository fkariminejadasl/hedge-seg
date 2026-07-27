# Working notes for Claude

## How to write

- Use simple language. Short sentences. No em dashes.
- This is the most important rule, in chat and in documentation. Explain things
  so they are easy to understand. Do not use a hard word where an easy one
  works. Do not pack several ideas into one sentence. If an explanation needs a
  term like "block buffering" or "start method", say what it means in plain
  words the first time it appears.
- Do not add formatting flourishes to documentation. Plain text, plain lists.
- Give full paths so they can be clicked, for example
  `scripts/train_detr_unet_polyline.py`, not "the training script".
- When asked to diagnose, diagnose first and do not change code until asked.
- Say which files were inspected before answering a question about the repo.

## Committing

- Never run `git commit` on your own. Make the change, run the formatter, say
  what changed and why, and propose the commit message. The user reviews it
  first and then says when to commit.
- Never run `git push` unless the user says to push.
- This holds even when the change was asked for and looks finished. Being asked
  to make a change is not permission to commit it.

## Before every commit

Run the formatter and linter:

```
for i in hedge_seg exps scripts; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
```

Fix what pyflakes reports in the code being committed. Leave unrelated
pre-existing warnings alone and mention them instead.

## Commit messages

- No Claude Code attribution, no Co-Authored-By line, no tool names.
- Explain why, not only what. Numbers are useful, for example
  "0.354 GB per image, so 40 GB fits about 100".
- Wrap at about 72 characters.

## Documentation layout

- `docs/descriptions.md`: overview of every script and dataset. Keep it
  polished and current.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind them. Has a Done and a TODO section for the phase plan.
- `docs/experiment_log.md`: raw notebook of individual runs, in a terse style.
  Held to a lower bar than the other two.

After a change, update the docs it affects, in the same commit:
- New or changed behavior of a script: its top docstring and `descriptions.md`.
- A bug or surprise worth remembering, or a design decision: `lesson_learned.md`,
  with the reason.
- A finished or planned run: `experiment_log.md`, and move the phase plan's
  Done/TODO in `lesson_learned.md` if it changed.
Do not leave a doc describing the old behavior.

## Environments

Local:
- conda env `hedge` has geopandas, shapely, rasterio. The base env does not.
  Run scripts with `PYTHONPATH=. python ...` from the repo root.
- GPU is an RTX PRO 3000 with 12.3 GB, so it is only good for smoke tests and
  one image overfit runs.

Cluster (Snellius, `ssh me`):
- code `~/dev/hedge-seg`, data `/projects/prjs1025/data/hedge`,
  experiments `~/exps/hedge`.
- conda env `hedge` there too.
- Do not hardcode paths. `hedge_seg/paths.py` resolves DATA_ROOT, EXP_ROOT and
  CLUSTER_EXP_ROOT from a filesystem marker, so the same script runs on both
  machines with no edits. CLUSTER_EXP_ROOT locally points at
  `/home/fatemeh/Downloads/hedge/snellius`, which mirrors `~/exps/hedge`.
- Limits are fine so far but worth checking: `myquota prjs1025` for disk and
  inodes, `accinfo` for the GPU budget.
- A100 gives 18 CPUs per GPU, H100 gives 16. A100 has a shorter queue, H100 is
  faster.

## Cluster run workflow

1. Commit and push locally, then `ssh me` and pull in `~/dev/hedge-seg`.
2. Copy `slurm/snellius_<model>.sh` to `~/exps/hedge/<model>/<n>.sh`, set the
   `-o` line and the `exp` in the training config to the same `<n>`.
3. `sbatch <n>.sh`.
4. Outputs go to `~/exps/hedge/<model>/<n>/best_<n>.pt`, next to `<n>.sh` and
   `<n>_<jobid>.out`. This matches the semseg_unet layout.

The slurm script prints the git hash and the content of the data scripts, the
training script and `hedge_seg/paths.py`, so the log alone documents the run.
Keep hyperparameters in the script config, not in `paths.py`, so they end up in
that log.

Ask before submitting a job. Slurm jobs cost budget and run for hours.

## Bringing a cluster run back to the laptop

- Copy the whole run directory, not the files inside it, so the local mirror
  keeps the `<n>/` level and matches `~/exps/hedge`:

  ```
  scp -r me:~/exps/hedge/detr_unet_polyline/1 \
      /home/fatemeh/Downloads/hedge/snellius/detr_unet_polyline/
  ```

- Point checkpoint configs at `CLUSTER_EXP_ROOT`, never `EXP_ROOT`. It resolves
  to `~/exps/hedge` on the cluster and to the local mirror on the laptop, so
  the same line works on both machines with no editing. This applies to
  `infer_ckpt` and `backbone_ckpt`.
- Copy the run's val stem list too. See the split note under project specifics.

## Looking at an inference result

Settings live in the cfg block at the bottom of each script. Edit it and run.
Do not add command-line overrides: a run must be readable from the file alone,
otherwise the committed script stops being the record of what ran.

1. In `scripts/train_detr_unet_polyline.py`, set `mode="infer"` and pick
   `infer_ckpt`. Alternatives are commented out right under it.
2. Run it. Output goes to
   `<infer_out_dir>/<ckpt stem>_<split dir name>_t<threshold>/`, which holds
   `polylines/` and a `gt/` of links to the ground truth of exactly those
   crops. Runs never overwrite each other and nothing needs linking by hand.
3. List the run directories in `scripts/show_polyline_results.py` and run it.
   It draws one ground-truth figure and one figure per run, over the same
   crops, so the panels line up.

`infer_polyline_dir` should be `polylines/val_cluster`, not `polylines/val`.
See the split note under project specifics.

## Skills

`.claude/skills/` holds the two run procedures, invoked by name:

- `/cluster-run`: commit, push, pull on Snellius, copy and edit the slurm
  script, sbatch, watch.
- `/laptop-run`: the laptop config overrides and the nohup launch.

They exist because both are multi-step and easy to get half right. The config
overrides they list are edits to make in the script, not hidden settings, so
the committed script always shows the values that actually ran.

## Stopping a local training run

- Stop it gently first: `pkill -TERM -f <script>` (or Ctrl-C if foreground) so
  the process releases the GPU cleanly. Only hard-kill (`pkill -9`) if it does
  not exit within ~10 s.
- Hard-killing a process while it is using the GPU can leave the driver wedged:
  the GPU stays at high power (P1, boosted clocks, 100% util stuck) and stays
  hot with the fan loud, even though nothing is training. `nvidia-smi` shows no
  compute process, and `power.draw`/`clocks.sm`/`pstate` reveal the real state
  (idle is ~5-15 W and P8).
- To un-wedge it, open and cleanly close a small CUDA context, which makes the
  driver re-check and drop to idle. Do NOT `nvidia-smi --gpu-reset` while Xorg
  uses the GPU; it crashes the display.

  ```
  python -c "import torch; x=torch.zeros(8,device='cuda'); torch.cuda.synchronize(); del x; torch.cuda.empty_cache()"
  ```

- After any kill, confirm no orphaned workers remain (`pgrep -af <script>`);
  persistent DataLoader workers do not always die with the parent.

## Reading a running job

- A log that stops updating is usually stdout buffering, not a hung job. Use
  `python -u` in the slurm script. Check the checkpoint mtime and `squeue`
  before concluding anything is wrong.
- Claude's background monitoring dies when this session ends. For a
  notification that survives a laptop shutdown, enable the `--mail-type` lines
  in the slurm script.
- Tensorboard on a cluster run, over a forwarded port. `--logdir_spec <name>:<dir>`
  labels the run, so `1:1` means show directory `1` under the name `1`:

  ```
  ssh -X -L 4004:localhost:4004 me
  cd ~/exps/hedge/detr_unet_polyline/
  conda activate hedge
  tensorboard --port 4004 --logdir_spec 1:1
  ```

  Then open http://localhost:4004 locally.

## Project specifics that are easy to get wrong

- Crops are sampled per polyline, so they overlap geographically. Train and val
  must be split by location, never randomly. `verify_no_split_overlap` in
  `hedge_seg/utils.py` checks this and should stay an assert.
- Regenerating a split must delete the old NPZs first, or a crop that moves
  between splits leaves a stale copy in both.
- `loss_poly` is mean L1 per coordinate on points normalized by `pad_to - 1`.
  Multiply by 1023 for pixels, then by 0.25 for meters.
- Only the ordered point loss knows about point order. Box, length and
  direction losses do not, and they stabilize a zigzag if their weight is high.
  Keep them at 0 unless there is a reason.
- For small overfit runs, inspect the final checkpoint, not `best_*.pt`. Best
  is chosen by eval loss, which stops falling after a few epochs when there is
  not enough data to generalize.
- This holds for real runs too. In cluster run 1 the final epoch 150 checkpoint
  draws better polylines than `best_1.pt` from epoch 65, even though its eval
  loss is higher. Never rank checkpoints or report results by eval loss on this
  task. See "Eval loss is not detection quality" in `docs/lesson_learned.md`.
- `infer_score_thresh` is a real knob, not a formality. Scores sit near 1, so
  0.5 keeps almost everything. Cluster run 1 needed 0.95 to match the GT line
  count.
- The local and cluster conversions of pdok_dataset3 do NOT produce the same
  train/val split, because `avoid_label_dirs` sees 10 labels locally and 5,000
  on the cluster. About 46% of the local val crops were cluster training
  images. Evaluate a checkpoint only on the val stems of the run that produced
  it.
