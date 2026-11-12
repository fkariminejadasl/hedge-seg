# Working notes

## Start of a session

- Read `docs/status.md` first: where the project stands, the results and the
  next steps. Before proposing a run or a change, search
  `docs/lesson_learned.md` and `docs/experiments_log.md` for what was already
  tried, measured and ruled out, and `docs/descriptions.md` for the scripts
  involved. Read a whole file when the task needs it.
- Answer from this repository only. The other repos in this VS Code workspace
  (bird-behavior, ...) are not context unless I say so.

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
  `scripts/train_detr_unet_polyline.py`, not "the training script". Name the
  exact function or class too.
- When asked to diagnose, diagnose first and do not change code until asked.
- Say which files were inspected before answering a question about the repo.

## Reporting a result

- **Report files touched outside `git status`.** Finish any task that ran,
  copied or scored something with a list of what changed outside the repo:
  directories created, with sizes; and any tracked file edited in place and
  then restored, naming the fields changed. A file that was edited and restored
  still has to be named, because it is how the numbers were produced.
- **Report the commands that produced a result, and what was checked.** Give
  the commands that actually ran, in order, in one code block, with a short
  comment above each saying what it does. Long paths may be shortened. Then
  list what was checked and what it showed, for example that the job exited 0,
  that two checkpoints hold the same weights, or that an old run reproduces its
  logged numbers. Say which tests ran, and if none ran, say so and why.

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

```
for i in hedge_seg exps scripts tests; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
mdformat docs/
```

Fix what pyflakes reports in the code being committed. Leave unrelated
pre-existing warnings alone and mention them instead.

`mdformat` (with `mdformat-gfm`, in the `test` extra of `pyproject.toml`)
formats the markdown and aligns the pipe tables. Run it on `docs/` only, never
on `presentation/`: those are Marp decks where a reflow can silently push
content off a slide. It does not rewrap paragraphs, so keep edited prose lines
near 80 characters yourself. If it adds a backslash escape, the markdown was
ambiguous; put that filename or expression in backticks instead.

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
- Give one copyable block with three commands in order: `git add` with the
  files of that commit, `git commit -m "..."`, and `git push origin <branch>`
  for the current branch. Name the files; never `git add -A` or `git add .`,
  which would also take untracked files like `.vscode/`. Writing the push
  command for the user to run is not running it.

## Documentation layout

- `docs/status.md`: where the project stands. The best models, how the numbers
  are measured, the results table, Done, TODO and Dropped.
- `docs/descriptions.md`: overview of every script and dataset. Keep it
  polished and current.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind them.
- `docs/experiments_log.md`: raw notebook of individual runs, terse, held to a
  lower bar. Except its opening table, "How each dataset was made": which
  script built which directory, from what input, and what a rebuild needs that
  the script docstring does not say. Add a row for every new dataset and edit
  it when its script changes, because a deleted dataset can only be rebuilt
  from it.
- `docs/workflow.md`: copying runs and data between the machines, checking a
  copy, backups, inference and figures.
- `docs/literature.md`: papers, datasets and data sources read for the project.
- `docs/qgis.md`: how to look at the data in QGIS, and how the Top10NL hedge
  layer was exported.
- `presentation/presentation.md`: the talk. Highlights only, short and
  itemised, written for ecologists rather than for engineers. Where
  `lesson_learned.md` gives the reasoning, this gives the conclusion in one
  line. Keep a technical name in italic parentheses after the plain-language
  version, so the audience follows and a specialist can still place it.
- `presentation/update.md`: the meeting updates, one slide per meeting, newest
  first. The user writes it. Do not add to it, reorder it or fill it in; only
  correct the wording.
- `presentation/README.md`: which figures the talk uses and the exact steps to
  remake them. No findings here, only mechanics.

Each result is written in full in one place only: the details of a run in
`experiments_log.md`, the results table in `status.md`. Other docs name the
result and point there, so two copies cannot drift apart. The talk is the
exception: it repeats numbers, each with its measure and its source.

## Every number needs code in the repository

This applies to every document listed above, not only the talk.

- **If a number, table or finding is worth writing down, the code that produced
  it is worth committing.** Put it in `exps/`, with the result in its top
  docstring so nobody has to run it to learn the answer. Name a one-off probe
  `probe_<what it asks>.py`; a standing tool keeps its own name, like
  `exps/dataset_stats.py`.
- **Name that script wherever the number appears.** In `lesson_learned.md`,
  `status.md` and `descriptions.md` write the path in the sentence; in
  `presentation.md` put it in italic parentheses at the end of the line, for
  example `*(exp 2, exps/probe_polyline_pr.py)*`.
- **This includes the quick check done in a scratch file.** If its answer ends
  up in a document, the file moves to `exps/` and gets a docstring. A number
  whose code was thrown away cannot be rechecked when the data changes.
- A probe that only confirmed something and changed no document can be deleted.
  The test is whether a document depends on it.

One rule for both Marp decks, `presentation.md` and `update.md`:

- **A slide must fit one page.** Marp does not warn, it silently cuts off the
  bottom. Budget for the default 16:9 theme: about 13 lines of body text, or
  about 5 lines plus one `h:420` image, and no body line over about 95
  characters. `update.md` sets 22px type and carries no figures, so more fits.
  Marp runs as a VS Code extension here, so you cannot render to check. Keep to
  the budget, do not change a tested image height blind, and say when a slide
  should be exported and looked at.

After a change, update the docs it affects, in the same commit:
- New or changed behavior of a script: its top docstring and `descriptions.md`.
- A bug or surprise worth remembering, or a design decision: `lesson_learned.md`,
  with the reason.
- A finished or planned run: `experiments_log.md`, and in `status.md` the
  results table, Done and TODO.
- A result that changes the headline numbers, the figures, or the next steps:
  `presentation/presentation.md` too. It goes stale silently, because nothing
  breaks when it is wrong.
Do not leave a doc describing the old behavior.

## Environments

Local:
- Always the conda env `hedge`, never base. Dependencies belong in
  `pyproject.toml` and get installed into `hedge`.
- Call the interpreter by absolute path,
  `/home/fatemeh/miniconda3/envs/hedge/bin/python`, with `PYTHONPATH=.` for
  scripts. Not `conda run -n hedge`: an editor started from a shell with
  another env active passes on `VIRTUAL_ENV` and `PATH`, which win over
  `conda run`, so it silently runs the other interpreter. The absolute path
  cannot be shadowed.
- Scratch files go in `/home/fatemeh/Downloads/hedge/claude/`, never `/tmp`,
  which does not survive a reboot. A scratch file may only be deleted, never
  merely abandoned: if its answer went into a document, it moves to `exps/`
  first; otherwise it goes in the bin.
- Tool permissions belong in this project's own
  `hedge-seg/.claude/settings.local.json`, not in whichever directory the
  session started in.
- The GPU is an RTX PRO 3000 with 12.3 GB, good only for smoke tests and small
  runs.

Cluster (Snellius, `ssh me`):
- Code `~/dev/hedge-seg`, data `/projects/prjs1025/data/hedge`, experiments
  `~/exps/hedge`, conda env `hedge` there too.
- Do not hardcode paths. `hedge_seg/paths.py` resolves DATA_ROOT, EXP_ROOT and
  CLUSTER_EXP_ROOT from a filesystem marker, so the same script runs on both
  machines. CLUSTER_EXP_ROOT is `~/exps/hedge` on the cluster and the mirror
  `/home/fatemeh/Downloads/hedge/snellius` on the laptop.
- Check `myquota prjs1025` before generating a dataset and `accinfo` before
  submitting a job. Watch inodes as well as bytes: a 30,000-crop dataset is
  60,000 inodes. Report what the commands say; do not copy the numbers into the
  docs, they change constantly.
- To free space, delete the epoch snapshots `<n>_<epoch>.pt` first, once a run
  has been scored. Keep `best_<n>.pt` and `<n>.pt`. Do not delete
  `pdok_dataset3` or any other dataset unless there is no alternative and the
  user has agreed; regenerating one is slow and changes the split.

## Running things

- A cluster job: the `/cluster-run` skill. Ask before submitting; slurm jobs
  cost budget and run for hours.
- A laptop run, and stopping one safely: the `/laptop-run` skill.
- Copying runs or data between the machines, checking a copy, backups,
  inference and figures: `docs/workflow.md`.
- Settings live in the cfg block at the bottom of each script. Edit it and run.
  Do not add command-line overrides: the committed script must stay the record
  of what ran. The overrides a skill lists are edits to that cfg.
- Use only this repository's skills. bird-behavior has skills with the same
  names; never run theirs.

## Project specifics that are easy to get wrong

- Crops are sampled per polyline, so they overlap geographically. Train and val
  must be split by location, never randomly. `verify_no_split_overlap` in
  `hedge_seg/utils.py` checks this and should stay an assert.
- Regenerating a split must delete the old NPZs first, or a crop that moves
  between splits leaves a stale copy in both.
- `loss_poly` times 1023 gives pixels, times 0.25 more gives metres (see the
  Model section of `docs/descriptions.md`).
- Only the ordered point loss knows about point order. Box, length and
  direction losses do not, and they hold a zigzag in place if their weight is
  high. Keep them at 0 unless there is a reason.
- Never rank checkpoints or report results by eval loss, and do not trust a
  mean over crops either. Score `<n>.pt` as well as `best_<n>.pt`. See "No
  cheap measure can rank two checkpoints" in `docs/lesson_learned.md`.
- `infer_score_thresh` is a real knob, not a formality. Scores sit near 1, so
  0.5 keeps almost everything. Exp 9 is scored at 0.90.
- Point checkpoint configs (`infer_ckpt`, `backbone_ckpt`) at
  `CLUSTER_EXP_ROOT`, never `EXP_ROOT`, so the same line works on both
  machines.
