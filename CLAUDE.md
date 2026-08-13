# Working notes for Claude

## Rules
- At the start of a session, read `docs/description.md` (the Quick reference
  section first), `docs/lesson_learned.md` and `docs/experiment_log.md` before
  proposing or running anything. They are not loaded automatically, and they
  hold what has already been tried, measured and ruled out.
  
## How to write

- Use simple language. Short sentences. No em dashes.
- This is the most important rule, in chat and in documentation. Explain things
  so they are easy to understand. Do not use a hard word where an easy one
  works. Do not pack several ideas into one sentence. If an explanation needs a
  term like "block buffering" or "start method", say what it means in plain
  words the first time it appears.
- Keep it short. Docs, commit messages and chat all tend to grow. Before
  finishing any text, cut it. Say the finding, the reason, and the number, then
  stop. Drop background the reader already has, restatements of the same point,
  and sentences that only lead into the next one.
- Editing a doc means rewriting the section, not appending to it. If a new
  paragraph overlaps an old one, merge them. A lesson is one short section, not
  a history of what we thought over time.
- A commit message is a few lines plus the numbers that matter. Not a report.
- Do not add formatting flourishes to documentation. Plain text, plain lists.
- Give full paths so they can be clicked, for example
  `scripts/train_detr_unet_polyline.py`, not "the training script".
- When asked to diagnose, diagnose first and do not change code until asked.
- Say which files were inspected before answering a question about the repo.

## Committing

- Never run `git commit` on your own. Make the change, run the formatter, say
  what changed and why, and propose the commit message. The user reviews it
  first and then says when to commit.
- **Always end with a commit message, unprompted.** Every time the working tree
  is left dirty, finish the reply with a ready-to-paste message covering
  everything uncommitted, not only the last edit. Do not wait to be asked. If
  the work splits into unrelated concerns, propose one message per commit and
  say which files belong to each.
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
  0.354 GB per image, so 40 GB fits about 100.
- Wrap at about 72 characters.
- **The message must paste straight into `git commit -m "<message>"`.** So it
  may not contain any of these four characters:

  | character | why it breaks |
  |---|---|
  | `"` | closes the quote early |
  | `'` | breaks the shell if the user quotes with it instead |
  | `` ` `` | runs as a command inside double quotes |
  | `$` | expands as a variable inside double quotes |

  Write `cls_loss=focal`, not `` `cls_loss="focal"` ``. Say "the 0.05
  threshold" as: the 0.05 threshold. Blank lines between paragraphs are fine
  inside the quotes, so a multi-paragraph message still works with one `-m`.
- Give it in a copyable block as the full command, `git commit -m "..."`, not
  as bare text the user has to wrap themselves.

## Documentation layout

- `docs/descriptions.md`: overview of every script and dataset. Keep it
  polished and current.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind them. Has a Done and a TODO section for the phase plan.
- `docs/experiment_log.md`: raw notebook of individual runs, in a terse style.
  Held to a lower bar than the other two. It opens with a **"How each dataset
  was made"** table: which script built which directory, from what input, and
  anything about rebuilding it that the script docstring does not say. Add a
  row whenever a dataset is created, and edit the row whenever the script that
  builds it changes. This is the one part of that file held to the same bar as
  the other docs, because a deleted dataset can only be rebuilt from it.
- `presentation/presentation.md`: the talk. Highlights only, short and
  itemised, written for ecologists rather than for engineers. Where
  `lesson_learned.md` gives the reasoning, this gives the conclusion in one
  line. Keep a technical name in italic parentheses after the plain-language
  version, so the audience follows and a specialist can still place it.
- `presentation/README.md`: which figures the talk uses, what each set shows,
  and the exact steps to remake them. No findings here, only mechanics.

## Every number needs code in the repository

This applies to all four documents, not only the talk.

- **If a number, table or finding is worth writing down, the code that produced
  it is worth committing.** Put it in `exps/` as `probe_<what it asks>.py`, with
  the result in its top docstring so nobody has to run it to learn the answer.
- **Name that script wherever the number appears.** In `lesson_learned.md` and
  `descriptions.md` write the path in the sentence; in `presentation.md` put it
  in italic parentheses at the end of the line, for example
  `*(exp 2, exps/probe_polyline_pr.py)*`.
- **This includes the quick check done in a scratch file.** If its answer ends
  up in a document, the file moves to `exps/` and gets a docstring. A number
  whose code was thrown away cannot be rechecked when the data changes, and
  every number here has to survive the next dataset.
- A probe that only confirmed something and changed no document can be deleted.
  The test is whether a document depends on it.

One rule for `presentation.md` specifically:

- **A slide must fit one page.** Marp does not warn, it silently cuts off the
  bottom. Budget for the default 16:9 theme: about 13 lines of body text, or
  about 5 lines plus one `h:420` image, and no body line over about 95
  characters. Marp runs as a VS Code extension here, not on the command line,
  so Claude cannot render to check. Keep to the budget, do not change a tested
  image height blind, and say when a slide should be exported and looked at.

After a change, update the docs it affects, in the same commit:
- New or changed behavior of a script: its top docstring and `descriptions.md`.
- A bug or surprise worth remembering, or a design decision: `lesson_learned.md`,
  with the reason.
- A finished or planned run: `experiment_log.md`, and move the phase plan's
  Done/TODO in `lesson_learned.md` if it changed.
- A result that changes the headline numbers, the figures, or the next steps:
  `presentation/presentation.md` too. It goes stale silently, because nothing
  breaks when it is wrong.
Do not leave a doc describing the old behavior.

## Environments

Local:
- Always the conda env `hedge`, never base:

  ```
  conda activate hedge
  PYTHONPATH=. python scripts/<script>.py
  ```

  Dependencies belong in `pyproject.toml` and get installed into `hedge`.
- Claude must call the interpreter by absolute path:

  ```
  /home/fatemeh/miniconda3/envs/hedge/bin/python <script>.py
  ```

  Not `conda run -n hedge`. If the editor was launched from a shell with
  another env active, `VIRTUAL_ENV` and `PATH` are inherited by every tool
  call and win over `conda run`, so `conda run -n hedge python` silently runs
  the other env's interpreter. That happened once and produced a wrong claim
  that geopandas was missing. The absolute path cannot be shadowed.
- Scratch files go in `/home/fatemeh/Downloads/hedge/cluade/`, never `/tmp`,
  which does not survive a reboot. A scratch file may only be deleted, never
  merely abandoned: either its answer went into a document, in which case it
  moves to `exps/` first (see "Every number needs code in the repository"), or
  it answered nothing and goes in the bin.
- Claude's tool permissions belong in this project's own `.claude`, that is
  `hedge-seg/.claude/settings.local.json`, not in whichever directory the
  session happened to start in.
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
- Check `myquota prjs1025` before generating a dataset and `accinfo` before
  submitting a job, because either can run out. Watch inodes as well as bytes:
  a dataset is two files per crop, so a 30,000-crop dataset is 60,000 inodes.
  Report what the commands say; do not copy the numbers into the docs, they
  change constantly.
- To free space, delete intermediate checkpoints first: the `<n>_<epoch>.pt`
  snapshots, once a run has been scored. Keep `best_<n>.pt` and `<n>.pt`,
  because the last epoch has beaten the best-eval one three times. Do not
  delete `pdok_dataset3` or any other dataset unless there is no alternative
  and the user has agreed; regenerating one is slow and changes the split.
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
  keeps the `<n>/` level and matches `~/exps/hedge`. The slurm script and the
  log sit *beside* that directory on the cluster, so `scp -r` of the directory
  alone silently leaves them behind. Take all three and put the two loose files
  inside `<n>/` locally, so one directory holds the whole run:

  ```
  scp -r me:exps/hedge/detr_unet_polyline/2 \
      /home/fatemeh/Downloads/hedge/snellius/detr_unet_polyline/
  scp me:exps/hedge/detr_unet_polyline/2.sh \
      me:exps/hedge/detr_unet_polyline/2_<jobid>.out \
      /home/fatemeh/Downloads/hedge/snellius/detr_unet_polyline/2/
  ```

  The log is not optional. It records the git hash and the full text of the
  training script, so it is the only proof of what actually ran.
- Only the checkpoints being analysed are worth mirroring. `best_<n>.pt` and
  `<n>.pt` are usually the same weights when the best epoch is the last one;
  check before running inference twice. Delete the `<n>_<epoch>.pt` copies
  locally once they are not needed, they are 80 MB each.
- Inference at `infer_score_thresh=0.05` keeps everything, so the reporting
  threshold can be swept in both directions afterwards. A run saved at 0.95 can
  only be swept upward. The directory name records the *inference* cutoff, so
  `best_2_val_cluster_t0.05` scored at 0.95 is normal and not a mistake.

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

## Screenshots

`/home/fatemeh/Downloads/hedge/screenshots/`. Name a figure `<model>_<what it
shows>`, so it can be found without being told. The model prefix matters, the
run directory alone does not say which model it came from:
`detr_unet_polyline_1_150_val_cluster_t.95.png`,
`detr_unet_polyline_1_gt_cluster_t.95.png`,
`detr_unet_polyline_exp1_tensorboard.png`.

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
- Never rank checkpoints or report results by eval loss, and do not trust a
  mean over images either. Both gave the wrong answer on cluster run 1. Inspect
  the final checkpoint as well as `best_*.pt`. See "No cheap measure can rank
  two checkpoints" in `docs/lesson_learned.md`.
- `infer_score_thresh` is a real knob, not a formality. Scores sit near 1, so
  0.5 keeps almost everything. Cluster run 1 uses 0.95.
- The local and cluster conversions of pdok_dataset3 do NOT produce the same
  train/val split, because `avoid_label_dirs` sees 10 labels locally and 5,000
  on the cluster. About 46% of the local val crops were cluster training
  images. Evaluate a checkpoint only on the val stems of the run that produced
  it.
