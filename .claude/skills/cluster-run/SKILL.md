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

1. If the training script or anything it imports changed, run the smoke test
   first. It is a few real epochs on tiny subsets and takes under a minute, and
   it catches a broken config key, a shape bug or a DataLoader hang before they
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
   breaks (see lesson_learned.md).

2. Format and lint locally, fix what pyflakes reports in the changed code:

   ```
   for i in hedge_seg exps scripts; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
   ```

3. Commit and push (`git push origin <branch>`). No Claude Code attribution in
   the message. Claude never runs `git commit`; propose the message and let the
   user commit.

4. On the cluster, pull, and check it really moved:

   ```
   ssh me
   cd ~/dev/hedge-seg && git pull --ff-only origin <branch> && git log --oneline -1
   ```

   Compare that hash against the local one. The cluster silently sits on an old
   commit otherwise, and the run then documents the wrong code.

5. Copy the slurm script and set `<n>` in three places so they match: the file
   name `<n>.sh`, the `-o` line, and `exp` in the training script cfg. Pick the
   next free `<n>` from `ls ~/exps/hedge/<model>/`.

   ```
   cp ~/dev/hedge-seg/slurm/snellius_<model>.sh ~/exps/hedge/<model>/<n>.sh
   # edit the -o line to .../<model>/<n>_%j.out
   ```

   `exp` lives in the committed cfg, so it is edited locally and pushed in step
   3, not on the cluster. Set `infer_score_thresh` for the run that follows too:
   0.05 keeps everything and can be swept both ways, while a run saved high can
   only be swept upward and has to be re-inferred.

6. Choose the GPU line in `<n>.sh`: `gpu_a100` (18 CPUs/GPU, shorter queue) or
   `gpu_h100` (16 CPUs/GPU, faster). Set `--time` with margin over the ETA.

7. Submit from the experiment dir:

   ```
   cd ~/exps/hedge/<model> && sbatch <n>.sh
   ```

8. Monitor. Outputs go to `~/exps/hedge/<model>/<n>/best_<n>.pt`, next to
   `<n>.sh` and `<n>_<jobid>.out`.
   - `squeue -j <jobid>` for state, `sacct -j <jobid>` after it ends.
   - The `.out` log can lag if `python -u` is missing; check the checkpoint
     mtime and `squeue`, not the log alone.
   - Tensorboard over a forwarded port:

     ```
     ssh -X -L 4004:localhost:4004 me
     cd ~/exps/hedge/<model>/ && conda activate hedge
     tensorboard --port 4004 --logdir_spec <n>:<n>
     ```

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
