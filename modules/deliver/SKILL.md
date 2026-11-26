---
name: rsi-deliver
description: Use when the orchestrator dispatches the RSI deliver module: package a candidate model behind a prediction interface, run predictions from inputs only, measure latency and cost; not for general packaging requests.
---

# rsi-deliver: deliver module

Package the model and its prediction interface, run predictions on the evalset and the valset from their inputs only, measure latency and cost, collect execution feedback. Do only this module's work and dispatch no other subagent; how the module works inside is up to its owner, this skill fixes only the interface.

## Inputs

The brief is built by code and placed in the output directory: `brief.md` (for people) and `brief.json` (the same content, structured: `module`, `purpose`, `round`, `task_json`, `ids`, `inputs`, `outdir`, `budget_left`, `goal` and more; the fields are described in `skills/rsi-orchestrator/templates/brief.md` under the framework repository root). The brief's `inputs` are authoritative: read only the paths listed there. When sent back, the same session gets `feedback-<k>.md` (`<k>` is the launch number), which says why it was sent back and what to change: fix that, then write `handback.json` again. The same session also gets two kinds of `feedback-<k>.md` that are not a return: one starting with `# Continue: the second part of the hand-back` (the second half of a two-part hand-back, e.g. train first hands back the trial candidates and judge the evaluation; once they are registered, it gets their registered ids here and then writes the search or the plan), and one starting with `# The user answered your questions` (see "Outputs").

Read only the artifacts listed below; do not look for or read any artifact or directory not listed; do not read the workspace or the registry.

- `task`.
- The `candidate` sent for evaluation, with a prediction entry that runs as it is (the manifest's model.predict). When the evalset was refreshed, the brief may also give the current best version's `candidate`: package it the same way, run its predictions, and hand in a separate delivery artifact for it.
- `inputs.jsonl`: the input file of the `evalset`; the input file (unlabelled) of the `valset` the candidate sent for evaluation used, to reproduce the candidate's valset predictions. Both as file paths only. When the milestone release is approved, also the `inputs.jsonl` of the `testset`.
- This module's `record`s of earlier rounds (only its own): what was tried before and how it went.

## Outputs

- `delivery`: the delivery package (in the delivery directory, or an external entry with its sha256), the entry point, the runtime environment, the runtime metrics, and the predictions on the evalset and the valset; when using the test data is approved, test predictions too.
- Optional `feedback`: execution feedback collected in trial runs.
- `record`: one per improvement experiment.

Write each artifact as a subdirectory of the output directory, with null as the manifest's id and round, task_id and task_version as the brief says; fields in [delivery.md](../../docs/interfaces/delivery.md). `record` in [record.md](../../docs/interfaces/record.md).

The input and output format of the prediction interface (input: one sample's state; output: probs, decision, uncertainty; every call writes one execution feedback row) is fixed in the "Prediction interface" section of [delivery.md](../../docs/interfaces/delivery.md); the form of the interface (command line, function, HTTP) is up to the owner and is written in `package.entry`.

Hand-back: once every artifact is written and validated, write `handback.json` in the output directory: `status` (`ok`, `failed` or `long_job`), `artifacts` (subdirectory names in the output directory, one per artifact), `marker` (the absolute path of the long job's completion marker; required with `long_job`, else null), `notes` (gate numbers, cost, problems met; with `failed`, the reason) and `questions` (what the user must decide; `[]` when nothing). The hand-back counts only once the engine reads this file; a missing or malformed file counts as a failed hand-back, and writing it again correctly is enough. Reading is strict: no field other than those above, `artifacts` holds only subdirectory names (no path separator, no repeats), every item of `questions` is a non-empty string, `marker` is an absolute path or null, and with `failed` `notes` is not empty.

When the hand-back has `questions`, none of the artifacts of this hand-back is registered: the engine asks the user first; once the user answers, the answer goes back to the same session (`feedback-<k>.md` starts with `# The user answered your questions`, not a return). Change or confirm the artifacts by the answer, then write `handback.json` again, listing every subdirectory to register (those listed last time too). The `questions` of a long job (`long_job`) hand-back are asked only once the completion marker is there; when it lists no artifact, the answer comes in the feedback starting with `# Long job finished`, sent back together with the long job's results.

## Gates

- Gate: the delivery package loads; the delivery package reproduces the candidate. With the package's prediction interface, run `predictions_val` on the valset's inputs.jsonl (`valset_id` is the candidate's `valset_id`); it must match the candidate's `predictions_val` row by row: the same `decision`, and every probability within the task definition's `delivery.reproduction_tolerance` (1e-6 when not set) of the candidate's; rows near the boundary whose probabilities are within the tolerance but whose decision flipped are let through up to floor(`delivery.max_decision_mismatch_rate` × the valset's row count) rows (0 rows when not set), and the report names them. At registration the code checks row by row and refuses any mismatch, whatever `consistent_with_offline` says; set this flag to true only when the reproduction holds.
- The delivery package calls the candidate's prediction entry (`<candidate_dir>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`, see [models.md](../../docs/interfaces/models.md)), or loads the same model and gets the same results; never write a separate scoring from a description in words. A candidate's `policy.json` and `policy_apply.py` (the decision policy, present when the task has a `policy` section) are part of its prediction entry: copy them with the model and keep them beside `predict.py`, which applies them (calibrated probabilities and the policy's decisions); never drop them or decide with a threshold of your own.
- The delivery package is self-contained: `package.location` is a relative path inside the delivery directory (it enters the workspace with the directory at registration), or `external:<name>`, with its location and sha256 given in the manifest's `external` (a local entry must be a single file; pack a directory into a tar file first); never an absolute path, and never a path into your own working directory; at registration, symbolic links in the delivery directory are copied as the content they point to (e.g. a venv in the package), and links to files that do not exist are refused.
- Never read or ask for labels.jsonl or any labels. Having no labels is not a problem; do not ask for labels; if you get them by mistake, stop and report.
- Never touch the sealed testset's inputs without approval; a `delivery` with test predictions must give `testset_id`.
- Measure latency and cost as they are and write them into `runtime`: `latency_ms_p50` and `latency_ms_p95` are the quantiles, over the evalset inputs, of the wall-clock milliseconds of each call of the prediction interface (one sample), without the model loading time; `cost_per_1k` is the cost of 1000 predictions, with its unit in `cost_unit` (e.g. usd, gpu_seconds), null when unknown. The `id` of execution feedback rows is this module's choice; it only has to be unique within one trace.

## Running standalone

The owner runs it in their own sandbox directory, without the orchestrator:

1. Copy the `candidate`, the evalset's `inputs.jsonl` and the `inputs.jsonl` of the valset the candidate used from registered artifacts into the sandbox, paths given by hand; the task definition is `<task.json>`. No workspace is needed, and the registry is not read.
2. Write the output to an output directory you create in the sandbox, one subdirectory per artifact, with null as the manifest's `id` and a `round` of your choice.
3. Validate every artifact with `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`; only an artifact that passes counts as handed in. The only `rsi` commands you may run are this one and `rsi score` (for the metrics, baselines and deltas in the report); do not run `rsi register` or any other command.
4. The sandbox likewise holds only input files, no labels.

## Steps and strategy sub-skills

Filled in by the module owner: the packaging, the interface implementation, the latency and cost measurement, the collection of execution feedback. Write the repeatable steps as scripts; the skill covers only the parts that need judgement.

## Module metrics

Filled in by the module owner: latency, cost, whether packaging succeeds the first time, and how they are measured.

## Experiment records

Filled in by the module owner: hand in a `record` for every improvement experiment (fields in [record.md](../../docs/interfaces/record.md)), saying what was changed, the numbers before and after and the conclusion; `module` is `deliver`.

## Long jobs

Filled in by the module owner: large batch inference (run in the background, writing a completion marker). When it is expected to take longer than the brief allows, start it in the background, write a completion marker (first line ok or failed), and hand back the marker path, the log path and the expected duration at once; do not poll or wait.

Hand-back of a long job: when it is expected to take longer than the brief allows, start it in the background, write `handback.json` at once with `long_job` as `status` and the absolute path of the completion marker as `marker`, then end the session; do not poll or wait. If the completion marker is already there when you would hand back `long_job` (the long job ended before your hand-back, e.g. one a previous session started): do not hand back `long_job`. The engine only accepts a marker written after this launch; an earlier marker never counts as present and the run would wait forever. Read the results, finish the work, then hand back. When the job ends, succeeded or failed, it must write the completion marker, whose first line is `ok` or `failed`. When the marker is `ok`, the engine has the same session go on (`feedback-<k>.md` starts with `# Long job finished`): it reads the results, writes the artifacts and hands back with `ok`; when the marker is not `ok`, this dispatch is recorded as failed.

Before handing back `long_job`, write `<marker>.pid` next to the completion marker: the first line is the long job process's own pid, the second its start time (Unix seconds, `date +%s`). The pid must be written by the long job itself, e.g. in a shell `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <command>' > <log> 2>&1 &` (`exec` makes the command keep this pid), and in Python write `os.getpid()` and `int(time.time())` as the job starts; never use the background pid the shell gives instead, which may be a wrapper process and not the long job itself. Preferably run the long job in its own process group (e.g. Python's `subprocess.Popen(..., start_new_session=True)`). Write `handback.json` only after `<marker>.pid` is written. When the run is stopped, the engine checks by this file that the process is this long job, then terminates it (the whole process group); without this file it cannot terminate it and can only list it as possibly still running, for the user to end by hand.

Use a new completion marker path for every long job (e.g. `<name>-<k>.done`, `<k>` one more each time), and run `rm -f <marker> <marker>.pid` before starting it: the engine only accepts a marker whose modification time is not earlier than this launch; an earlier marker at the same path does not count as this long job's end.

## Owner and version

Filled in by the module owner: the owner, the current version, the change log, known limitations.

## Common mistakes

Filled in by the module owner: the mistakes this module has made, their consequences and the right way (a table). One is known: letting the evalset's labels into any part of the deliver module, e.g. by getting the whole evalset directory.
