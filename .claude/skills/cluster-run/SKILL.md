---
name: cluster-run
description: Run a training script on the Snellius cluster. Use when the user wants to launch, submit, or monitor a training job on the cluster (sbatch), e.g. "run the polyline baseline on the cluster", "submit a cluster job".
---

# Run a training job on the cluster

Procedure for launching a training script on Snellius (`ssh me`). The committed
config in each training script is the cluster baseline, so no value edits are
needed for a standard run. Paths come from `hedge_seg/paths.py`, which resolves
to the cluster automatically.

Ask before `sbatch`. It costs GPU budget and runs for hours.

## Steps

1. Set the run in the training script cfg, locally. `exp="<n>"`, with the next
   free `<n>` from `ls ~/exps/hedge/<model>/` on the cluster. Set
   `infer_score_thresh=0.05` for the inference that follows (see "Looking at an
   inference result" in `docs/workflow.md`). Hyperparameters belong in this
   cfg, not in `hedge_seg/paths.py`, so the log records them.

2. If the training script or anything it imports changed, run the smoke test.
   It is a few real epochs on tiny subsets and takes under a minute, and it
   catches a broken config key, a shape bug or a DataLoader hang before they
   cost hours of queue and budget:

   ```
   PYTHONPATH=. python exps/smoke_test_train_detr_unet_polyline.py
   ```

   Its cfg is a copy of the training cfg, so a new cfg key has to be added
   there too or the test fails with a missing key. That is the point: it is the
   cheapest place to find out.

   If inference also changed, run one in `mode="infer"` on a handful of crops
   and check both branches: predictions above the threshold, and a threshold
   high enough that a crop returns nothing. The empty branch is the one that
   breaks (see `docs/lesson_learned.md`).

3. Format and lint, and fix what pyflakes reports in the changed code:

   ```
   for i in hedge_seg exps scripts tests; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
   ```

4. Commit and push. Claude never runs `git commit`; propose the message and let
   the user commit. The push is `git push origin <branch>`, with `-u` on the
   first push of a new branch.

5. On the cluster, pull, and check it really moved:

   ```
   ssh me
   cd ~/dev/hedge-seg && git pull --ff-only origin <branch> \
     && git log --oneline -1 && git status --porcelain
   ```

   Compare that hash against the local one. The cluster silently sits on an old
   commit otherwise, and the run then documents the wrong code. `git status`
   matters too: a modified tracked file on the cluster means the job runs code
   that no commit describes, and the log's git hash then lies. Untracked junk
   like `.vscode/` or downloaded `*.pt` weights is fine and can be ignored.

6. Copy the slurm script, and set its `-o` line to the same `<n>` as `exp`:

   ```
   cp ~/dev/hedge-seg/slurm/snellius_<model>.sh ~/exps/hedge/<model>/<n>.sh
   # edit the -o line to .../<model>/<n>_%j.out
   ```

   The slurm script prints the git hash, `hedge_seg/paths.py`, the data scripts
   and the whole training script before starting, so the `.out` log alone
   documents the run.

7. Choose the GPU line in `<n>.sh`: `gpu_a100` (18 CPUs per GPU, shorter queue)
   or `gpu_h100` (16 CPUs per GPU, faster). Set `--time` with margin over the
   expected run time. For a notification that survives a laptop shutdown,
   enable the `--mail-type` lines; monitoring by Claude ends with the session.

8. Submit from the experiment dir:

   ```
   cd ~/exps/hedge/<model> && sbatch <n>.sh
   ```

9. Monitor. Outputs go to `~/exps/hedge/<model>/<n>/best_<n>.pt`, next to
   `<n>.sh` and `<n>_<jobid>.out`.

   - `squeue -j <jobid>` for state, `sacct -j <jobid>` after it ends.
   - A log that stops updating is usually stdout buffering, not a hung job; the
     slurm script runs `python -u` against it. Check the checkpoint mtime and
     `squeue` before concluding anything is wrong.
   - The log starts with the dumped source code, so cut to the run output before
     searching it (see "A slurm log begins with the source code" in
     `docs/lesson_learned.md`).
   - Tensorboard over a forwarded port. `--logdir_spec <name>:<dir>` labels the
     run, so `<n>:<n>` shows directory `<n>` under the name `<n>`:

     ```
     ssh -X -L 4004:localhost:4004 me
     cd ~/exps/hedge/<model>/ && conda activate hedge
     tensorboard --port 4004 --logdir_spec <n>:<n>
     ```

     Then open http://localhost:4004 locally.

10. Copy the run to the laptop, with its script and log, as in "Runs" in
    `docs/workflow.md`. Rerun the copy to update a run still going.

11. Score and record it: inference and figures as in `docs/workflow.md`, then
    `docs/experiments_log.md`, and the results table, Done and TODO in
    `docs/status.md`.

## Before the first big run of a model

- Run `exps/probe_batch_size.py` in a short slurm job on the target GPU to
  pick `batch_size`. The largest batch that fits is usually not the fastest;
  pick where s/image stops improving.
- Check limits: `myquota prjs1025` (disk, inodes), `accinfo` (GPU budget).

## Notes

- Machine-dependent run knobs (num_workers, and whether to subset val) stay
  explicit in the cfg; there is no hidden per-machine override. The committed
  values are the cluster ones.
- For a laptop run instead, see the laptop-run skill.
