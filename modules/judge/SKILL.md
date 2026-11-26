---
name: rsi-judge
description: Use when the orchestrator dispatches the RSI judge module: score a delivery with the evalset labels, give the promotion recommendation, the failure attribution and the next round's plan; not for general evaluation requests.
---

# rsi-judge: judge module

Keep the evalset's labels; score and compare, and give the promotion recommendation, the failure analysis and the next round's plan. Do only this module's work and dispatch no other subagent; how the module works inside is up to its owner, this skill fixes only the interface.

## Inputs

The brief is built by code and placed in the output directory: `brief.md` (for people) and `brief.json` (the same content, structured: `module`, `purpose`, `round`, `task_json`, `ids`, `inputs`, `outdir`, `budget_left`, `goal` and more; the fields are described in `skills/rsi-orchestrator/templates/brief.md` under the framework repository root). The brief's `inputs` are authoritative: read only the paths listed there. When sent back, the same session gets `feedback-<k>.md` (`<k>` is the launch number), which says why it was sent back and what to change: fix that, then write `handback.json` again. The same session also gets two kinds of `feedback-<k>.md` that are not a return: one starting with `# Continue: the second part of the hand-back` (the second half of a two-part hand-back, e.g. train first hands back the trial candidates and judge the evaluation; once they are registered, it gets their registered ids here and then writes the search or the plan), and one starting with `# The user answered your questions` (see "Outputs").

Read only the artifacts listed below; do not look for or read any artifact or directory not listed; do not read the workspace or the registry.

- `task` (metrics, baselines, promotion rule).
- The `candidate` sent for evaluation and this round's `delivery`; also the `search` when this round has one.
- `evalset`, with labels.jsonl; only this module gets labels.
- The rows.jsonl of the `trainset` the candidate sent for evaluation used, only to compute the baselines and for no other analysis.
- The current best version's name, with its `evaluation` and `delivery`; use the value of `best_before` the brief gives. When the evalset was refreshed, also the best version's `delivery` on the current evalset: compute the best version's side of the metrics and deltas from it, and write its id into the evaluation's parents.
- `feedback`, every `record`.
- The `plan` that opens this round (written last round; given for the evaluation only): judge against this round's evaluation whether that plan worked, and write it into the evaluation report.
- When the milestone release is approved, also the `testset` (with labels).
- With ③ Compare (report only, never a gate): the `comparison` of the candidate with the baselines on the valset (the brief field "strongest" names its strongest baseline), and each `baseline` to report as `baseline_eval_<id>` (the baseline directory: its manifest gives its name, its `predictions_eval` are on the evalset inputs); at the milestone each baseline scored on the testset inputs as `baseline_test_<id>` (its `predictions_test`). Read their predictions only: the judge never trains or runs a baseline.

## Outputs

- `evaluation`: each metric, the comparison with the current best version and with the baselines, with intervals, the promote or keep recommendation, the failed items and their attribution, `uses_testset`; the report file gives the results by group and slice. With ③ Compare also `baselines_eval`: every baseline the brief gives as `baseline_eval_<id>`, by id, with its name, its metrics on the evalset and the delta of the primary metric (candidate minus baseline); a milestone evaluation has `baselines_test` instead, for every `baseline_test_<id>`. They are a report: the recommendation never rests on them, but the failure analysis may use them and the comparison (e.g. the candidate behind a feature baseline such as gbdt points at the attribution model_capacity or config; behind the team's recipe, at what this round changed).
- `plan`: which module the next round enters at, the evidence and the change for each issue, the expected gain. When the candidate sent for evaluation used a trainset the data module produced this round, state in the evidence the conclusion on the candidate it trained against the current best version (promote or keep, the delta of the primary metric): the data module does not train, does not test a new trainset itself and does not get the `evaluation`, so the next round learns from the `plan` how the new trainset did. A closed round: state that the candidates on the new trainset did not pass the train gate.
- `record`: one per improvement experiment.

Write each artifact as a subdirectory of the output directory, with null as the manifest's id and round, task_id and task_version as the brief says; fields in [evaluation.md](../../docs/interfaces/evaluation.md). `record` in [record.md](../../docs/interfaces/record.md).

The `plan` refers to the registered id of the `evaluation`: first hand back the `evaluation` with `ok`; once the engine has registered it and decided, it has the same session go on (`feedback-<k>.md` starts with `# Continue: the second part of the hand-back`, not a return), with the registered id of the `evaluation` and the promotion decision in it; then write the `plan` and hand it back.

When the brief says this round is closed (no candidate passed the train gate, so this round has no `evaluation`), write only the `plan` and the `record`: the plan's evaluation_id is null, and the issues and evidence come from the train gate results and the experiment records of this round's candidates.

Milestone evaluation (when the brief says it is a milestone release): score the current best version's delivery; `uses_testset` is true, `candidate_id` is the current best version's candidate, `best_before` is the current best version; the metrics and deltas are still computed on the evalset, and the scores on the testset go into `metrics_test` (with the primary metric); the recommendation is keep: it is no promotion decision and takes no evalset use. No `plan`. Format in "Milestone evaluation" of [evaluation.md](../../docs/interfaces/evaluation.md).

Hand-back: once every artifact is written and validated, write `handback.json` in the output directory: `status` (`ok`, `failed` or `long_job`), `artifacts` (subdirectory names in the output directory, one per artifact), `marker` (the absolute path of the long job's completion marker; required with `long_job`, else null), `notes` (gate numbers, cost, problems met; with `failed`, the reason) and `questions` (what the user must decide; `[]` when nothing). The hand-back counts only once the engine reads this file; a missing or malformed file counts as a failed hand-back, and writing it again correctly is enough. Reading is strict: no field other than those above, `artifacts` holds only subdirectory names (no path separator, no repeats), every item of `questions` is a non-empty string, `marker` is an absolute path or null, and with `failed` `notes` is not empty.

When the hand-back has `questions`, none of the artifacts of this hand-back is registered: the engine asks the user first; once the user answers, the answer goes back to the same session (`feedback-<k>.md` starts with `# The user answered your questions`, not a return). Change or confirm the artifacts by the answer, then write `handback.json` again, listing every subdirectory to register (those listed last time too). The `questions` of a long job (`long_job`) hand-back are asked only once the completion marker is there; when it lists no artifact, the answer comes in the feedback starting with `# Long job finished`, sent back together with the long job's results.

## Gates

- Metrics, baselines, deltas and intervals are always computed with `rsi score`, and the numbers it prints are written into the evaluation as they are; never implement them yourself (definitions in "Metric definitions" of [task.md](../../docs/interfaces/task.md)). At registration the code recomputes them by the same definitions and refuses a difference over 1e-9:
  - `metrics.candidate`, `metrics.baselines` and the primary metric of `deltas`: `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --train <trainset>/rows.jsonl --clusters <evalset>/inputs.jsonl --compare <best-delivery>/predictions_eval.jsonl`; the `metrics`, `baselines` and `delta` it prints go into these three places respectively; when there is no best version yet, leave out `--compare`, `metrics.best` is null and `deltas` stays empty.
  - `metrics.best`: the same command with `--predictions` replaced by the best version's `predictions_eval.jsonl`, taking the `metrics` it prints. When the evalset was refreshed, the best version's side uses its delivery on the current evalset (the brief gives it), and that delivery's id goes into `parents`, otherwise registration is refused.
  - `metrics_test` of a milestone evaluation: `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl`.
  - `baselines_eval` (③ Compare, report only): add `--baselines-on evalset` and one `--baseline <dir>` per `baseline_eval_<id>` of the brief to the first command above (its `--clusters` is the evalset's inputs), and write the `baselines_eval` object it prints into the evaluation as it is; never assemble or copy the numbers by hand. Every baseline of the round's comparison must be there (`baselines_eval: missing <id> (<name>)` otherwise), and no other.
  - `baselines_test` of a milestone evaluation: the same on the testset, `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl --clusters <testset>/inputs.jsonl --baselines-on testset --baseline <dir>` (one `--baseline` per `baseline_test_<id>`), and its printed `baselines_test` goes in as it is. When `rsi score` refuses (a primary metric `rsi` does not compute, or one undefined on the set), see "③ Compare's baselines" in [evaluation.md](../../docs/interfaces/evaluation.md).
  - The `not_computed` of the output lists the metrics and baselines `rsi` does not know: this module computes them itself and states the method in the report; registration does not check them.
  - Slice and group results and the other numbers in the report are this module's choice and are not checked.
- The recommendation must agree with the promotion rule: by the task definition's `promotion_rule` and the decision method [task.md](../../docs/interfaces/task.md) describes, derive `promote` or `keep` from the deltas, intervals and guard metrics computed with `rsi score` and written into the evaluation (registration recomputes and checks them) (the gain must be greater than `min_delta`; a tie exactly equal to it, within a tolerance of 1e-9, is no gain: write `keep`), and `reason` cites the rule and the matching numbers. When the engine registers it and decides the promotion, it checks this in code and refuses a mismatch.
- Attributions may only take the values [evaluation.md](../../docs/interfaces/evaluation.md) allows; every failed item has evidence and exactly one attribution, and the module matches the attribution.
- The `plan`'s `entry_module` must be the module of some issue; every issue has evidence, a change and an expected gain.
- The evalset's uses are limited: each registered evaluation (except a milestone evaluation) takes one use; the sealed testset is used only after approval, with `uses_testset` true.
- The labels of the evalset and the sealed testset must never be written into the `plan`, the `record` or the per-sample rows of the report that other modules see.

## Running standalone

The owner runs it in their own sandbox directory, without the orchestrator:

1. Copy the `candidate`, the `delivery`, the `evalset` (with labels), the rows.jsonl of the `trainset` the candidate sent for evaluation used (only to compute the baselines), earlier `evaluation`s and the `record`s from registered artifacts into the sandbox, paths given by hand; the task definition is `<task.json>`. No workspace is needed, and the registry is not read.
2. Write the output to an output directory you create in the sandbox, one subdirectory per artifact, with null as the manifest's `id` and a `round` of your choice.
3. Validate every artifact with `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`; only an artifact that passes counts as handed in. The only `rsi` commands you may run are this one and `rsi score` (for the metrics, baselines and deltas in the report); do not run `rsi register` or any other command.
4. When the `plan` refers to the id of the `evaluation`, fill one in by hand in the sandbox; the engine checks it at registration.

## Steps and strategy sub-skills

Filled in by the module owner: slice analysis, failure attribution, plan writing, calibration of the expected gains. Scoring, baselines and confidence intervals are fixed code (`rsi score`, see "Gates") and are not defined again here. Write the repeatable steps as scripts; the skill covers only the parts that need judgement.

## Module metrics

Filled in by the module owner: the hit rate of the plans (how much of the gain a plan expected was really achieved) and how it is measured.

## Experiment records

Filled in by the module owner: hand in a `record` for every improvement experiment (fields in [record.md](../../docs/interfaces/record.md)), saying what was changed, the numbers before and after and the conclusion; `module` is `judge`.

## Long jobs

Filled in by the module owner: large resampling or per-sample analysis (run in the background, writing a completion marker). When it is expected to take longer than the brief allows, start it in the background, write a completion marker (first line ok or failed), and hand back the marker path, the log path and the expected duration at once; do not poll or wait.

Hand-back of a long job: when it is expected to take longer than the brief allows, start it in the background, write `handback.json` at once with `long_job` as `status` and the absolute path of the completion marker as `marker`, then end the session; do not poll or wait. If the completion marker is already there when you would hand back `long_job` (the long job ended before your hand-back, e.g. one a previous session started): do not hand back `long_job`. The engine only accepts a marker written after this launch; an earlier marker never counts as present and the run would wait forever. Read the results, finish the work, then hand back. When the job ends, succeeded or failed, it must write the completion marker, whose first line is `ok` or `failed`. When the marker is `ok`, the engine has the same session go on (`feedback-<k>.md` starts with `# Long job finished`): it reads the results, writes the artifacts and hands back with `ok`; when the marker is not `ok`, this dispatch is recorded as failed.

Before handing back `long_job`, write `<marker>.pid` next to the completion marker: the first line is the long job process's own pid, the second its start time (Unix seconds, `date +%s`). The pid must be written by the long job itself, e.g. in a shell `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <command>' > <log> 2>&1 &` (`exec` makes the command keep this pid), and in Python write `os.getpid()` and `int(time.time())` as the job starts; never use the background pid the shell gives instead, which may be a wrapper process and not the long job itself. Preferably run the long job in its own process group (e.g. Python's `subprocess.Popen(..., start_new_session=True)`). Write `handback.json` only after `<marker>.pid` is written. When the run is stopped, the engine checks by this file that the process is this long job, then terminates it (the whole process group); without this file it cannot terminate it and can only list it as possibly still running, for the user to end by hand.

Use a new completion marker path for every long job (e.g. `<name>-<k>.done`, `<k>` one more each time), and run `rm -f <marker> <marker>.pid` before starting it: the engine only accepts a marker whose modification time is not earlier than this launch; an earlier marker at the same path does not count as this long job's end.

## Owner and version

Filled in by the module owner: the owner, the current version, the change log, known limitations.

## Common mistakes

Filled in by the module owner: the mistakes this module has made, their consequences and the right way (a table). One is known: a recommendation that does not agree with the promotion rule, or attributions that do not take the allowed values.
