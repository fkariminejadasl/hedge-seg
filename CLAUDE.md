# Working notes

## Start of a session

- Read `docs/status.md`, `docs/descriptions.md`, `docs/lesson_learned.md` and
  `docs/experiments_log.md` before proposing or running anything. They are not
  loaded automatically, and they hold what has already been tried, measured
  and ruled out.
- Answer from this repository only. The other repos in this VS Code workspace
  (bird-behavior, ...) are not context unless I say so.

## How to write

- Use simple language and short sentences. No em dashes. This is the most
  important rule, in chat and in documentation.
- Do not use a hard word where an easy one works, and do not pack several ideas
  into one sentence. Explain a term like "block buffering" in plain words the
  first time it appears.
- Keep it short. Before finishing any text, cut it: say the finding, the reason
  and the number, then stop. Drop background the reader already has, repeats,
  and sentences that only lead into the next one.
- Editing a doc means rewriting the section, not appending to it. Merge
  overlapping paragraphs. A lesson is one short section, not a history of what
  we thought over time.
- No formatting flourishes in documentation. Plain text, plain lists.
- Give full paths so they can be clicked, like
  `scripts/train_detr_unet_polyline.py`, and name the exact function or class.
- When asked to diagnose, diagnose first and change no code until asked.
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

## Commits

- Never run `git commit` or `git push` on your own, even when the change was
  asked for and looks finished. Make the change, run the checks below, say what
  changed and why, and propose the message. The user commits and pushes.
- **Always end with a commit message, unprompted,** whenever the working tree is
  left dirty. Cover everything uncommitted, not only the last edit. If the work
  splits into unrelated concerns, propose one commit per concern and say which
  files belong to each.
- Give one copyable block: `git add` with the files named,
  `git commit -m "..."`, and `git push origin <branch>`. Build the `git add`
  line from `git status --short` run just before, and leave out paths already
  staged: the old path of a staged rename makes `git add` fail. Never
  `git add -A` or `git add .`, which would take untracked files like
  `.vscode/`. On the first push of a new branch, use `git push -u`.
- The message is a few lines plus the numbers that matter, and says why, not
  only what (for example: 0.354 GB per image, so 40 GB fits about 100). Wrap at
  about 72 characters. No Claude Code attribution, no Co-Authored-By line, no
  tool names.
- **It must paste straight into `git commit -m "<message>"`,** so it may not
  contain a double quote (closes the quote early), a single quote (breaks
  single quoting), a backtick (runs a command) or a dollar sign (expands a
  variable). So write cls_loss=focal, not a quoted value. Blank lines between
  paragraphs are fine.

Before every commit:

```
for i in hedge_seg exps scripts tests; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
mdformat docs/
```

- Fix what pyflakes reports in the code being committed. Mention unrelated old
  warnings instead of fixing them.
- `mdformat` (with `mdformat-gfm`, in the `test` extra of `pyproject.toml`)
  formats the markdown and aligns the pipe tables. Run it on `docs/` only,
  never on `presentation/`: those are Marp decks where a reflow can silently
  push content off a slide. It does not rewrap paragraphs, so keep edited prose
  lines near 80 characters yourself. If it adds a backslash escape, the
  markdown was ambiguous; put that name or expression in backticks instead.

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
  Keep the slide list and order: each slide is one step of the story told at
  the meeting, so do not merge slides to cut repetition.
- `presentation/update.md`: the meeting updates, one slide per meeting, newest
  first. The user writes it. Do not add to it, reorder it or fill it in; only
  correct the wording.
- `presentation/README.md`: which figures the talk uses and the exact steps to
  remake them. No findings here, only mechanics.

Write the docs for the cluster. Where the laptop copy differs from the
cluster, for example in a file count or a split, give only the cluster number:
the runs happen there, so a laptop number only confuses. A probe's docstring
may still add a line on how to run it on the laptop.

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
- **Every number says what it is:** the measure (F1, precision, recall,
  balanced accuracy, ...), the run, and the tolerance and cut-off when they
  differ from the defaults in `docs/status.md`. A bare 0.72 is not a result.
- **This includes the quick check done in a scratch file.** If its answer ends
  up in a document, the file moves to `exps/` and gets a docstring. A number
  whose code was thrown away cannot be rechecked when the data changes.
- A probe that only confirmed something and changed no document can be deleted.
  The test is whether a document depends on it.

One rule for both Marp decks, `presentation.md` and `update.md`:

- **A slide must fit one page.** Marp does not warn, it silently cuts off the
  bottom. Budget for the default 16:9 theme: about 12 lines of body text, or
  about 5 lines plus one `h:420` image. Count lines as they show on the slide:
  a line over about 72 visible characters wraps and counts twice (markup like
  `**` and backticks does not show), and a caption under an image must stay on
  one line. Measured on the PDF export of 2026-11-12. `update.md` sets 22px
  type and carries no figures, so more fits.
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
