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

1. Format and lint locally, fix what pyflakes reports in the changed code:

   ```
   for i in hedge_seg exps scripts; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
   ```

2. Commit and push (`git push origin <branch>`). No Claude Code attribution in
   the message.

3. On the cluster, pull:

   ```
   ssh me
   cd ~/dev/hedge-seg && git pull --ff-only origin <branch>
   ```

4. Copy the slurm script and set `<n>` in three places so they match: the `-o`
   line, and the `exp` in the training script cfg. Pick the next free `<n>`.

   ```
   cp ~/dev/hedge-seg/slurm/snellius_<model>.sh ~/exps/hedge/<model>/<n>.sh
   # edit the -o line to .../<model>/<n>_%j.out
   ```

5. Choose the GPU line in `<n>.sh`: `gpu_a100` (18 CPUs/GPU, shorter queue) or
   `gpu_h100` (16 CPUs/GPU, faster). Set `--time` with margin over the ETA.

6. Submit from the experiment dir:

   ```
   cd ~/exps/hedge/<model> && sbatch <n>.sh
   ```

7. Monitor. Outputs go to `~/exps/hedge/<model>/<n>/best_<n>.pt`, next to
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
