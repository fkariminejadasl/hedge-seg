# Workflow

How to keep the laptop in step with the cluster, run inference, and look at
the result. Training runs have their own procedures: `/cluster-run` and
`/laptop-run` in `.claude/skills/`.

## Keeping the laptop in step with the cluster

The laptop holds a full copy of the cluster data and runs. Copy each new run
and dataset over as below, so it stays that way. Use rsync, not scp: run it
again and it copies only what changed, so it also updates a run still going.

### Runs

`/home/fatemeh/Downloads/hedge/snellius` mirrors `~/exps/hedge`, with one
difference: the slurm script and the log sit beside the run directory on the
cluster, and inside it on the laptop, so one directory holds the whole run.

```
M=detr_unet_polyline; N=11; L=/home/fatemeh/Downloads/hedge/snellius
rsync -av --exclude='[0-9]*_[0-9]*.pt' me:exps/hedge/$M/$N $L/$M/
rsync -av me:exps/hedge/$M/$N.sh "me:exps/hedge/$M/${N}_*.out" $L/$M/$N/
```

- The exclude leaves the epoch snapshots `<n>_<epoch>.pt`, 80 MB each, on the
  cluster.
- The log is not optional. It records the git hash and the full training
  script, so it is the only proof of what ran.
- Copy one run at a time, never all of `~/exps/hedge`. A few laptop files have
  a cluster file of the same name with other content, and a whole-tree copy
  would overwrite them.

To list what the laptop lacks, without copying anything:

```
rsync -avn --ignore-existing --exclude='[0-9]*_[0-9]*.pt' \
    --exclude='*.sh' --exclude='*.out' \
    me:exps/hedge/ /home/fatemeh/Downloads/hedge/snellius/
```

### Data

Datasets are made by code, but build each one once, where its inputs are, and
copy it to the other machine. Do not build it again there, because a rebuild
does not give the same files: some download requests fail, and at different
places each time. pdok_dataset3 was once downloaded on both machines, and each
copy lacked two crops the other had. A dataset built from the laptop copy then
named crops the cluster did not have, and the first exp 4 job stopped on "No
image for pos_029989" (`exps/probe_dataset_stem_mismatch.py`). The lidar
patches are built on the laptop, which alone has the AHN4 rasters.

```
rsync -av me:/projects/prjs1025/data/hedge/<dataset> \
    /home/fatemeh/Downloads/hedge/results/
```

- Swap the two paths to send a laptop file to the cluster.
- After a rebuild, delete the old copy first, or files that changed split stay
  in both.
- Add the dataset to the table in `docs/experiments_log.md`, then run
  `exps/probe_dataset_stem_mismatch.py` on both machines.

### Checking a copy

- One copy: run the same rsync with `-n --checksum`. It reads every file on
  both sides and lists only those whose content differs, so an empty list means
  the copy is exact.
- Everything: `exps/probe_laptop_matches_cluster.py`, about 5 minutes. Add a
  new dataset to its list first.

### Backing up the laptop

Use `rsync -aH` to an ext4 disk. `-H` keeps a hard link as one file:
`pdok_dataset_semseg3` shares its images with `pdok_dataset3`, so a plain copy
stores them twice. The `gt/` links of the inference runs are absolute paths,
so restore to the same place.

## Looking at an inference result

1. In `scripts/train_detr_unet_polyline.py`, set `mode="infer"` and pick
   `infer_ckpt`. Alternatives are commented out right under it.
1. Run it. Output goes to
   `<infer_out_dir>/<ckpt stem>_<split dir name>_t<threshold>/`, which holds
   `polylines/` and a `gt/` of links to the labels of exactly those crops. Runs
   never overwrite each other and nothing needs linking by hand.
1. List the run directories in `scripts/show_polyline_results.py` and run it.
   It draws one label figure and one figure per run, over the same crops, so
   the panels line up.

- Infer at `infer_score_thresh=0.05`. That keeps every line, so the cut-off
  can be swept both ways afterwards; a run saved at 0.95 can only be swept
  upward. The directory name records the inference cut-off, so
  `best_2_val_cluster_t0.05` scored at 0.95 is normal.
- `best_<n>.pt` and `<n>.pt` hold the same weights when the best epoch is the
  last one. Check before running inference twice.

## Screenshots

Save figures in `/home/fatemeh/Downloads/hedge/screenshots/`, named
`<model>_<what it shows>`, so they can be found without being told. The model
prefix matters, because the run directory alone does not say which model it
came from: `detr_unet_polyline_1_150_val_cluster_t.95.png`,
`detr_unet_polyline_1_gt_cluster_t.95.png`,
`detr_unet_polyline_exp1_tensorboard.png`.
