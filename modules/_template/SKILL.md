---
name: rsi-<module>
description: Use when the orchestrator dispatches the RSI <module> module: <one sentence on what this module does>; not for general <domain> requests.
---

# rsi-<module>: <module name>

This is the template of a module skill. Copy it to `modules/<module>/SKILL.md` (the module's own folder, which also holds its code, references and tests), then replace what is in angle brackets. The framework has written the interface in the first four sections; the module owner fills in the other six. How the module works inside is up to its owner; for the interface (inputs, outputs, gates, running standalone), the brief the code builds (`rsi/briefs.py`) and `docs/interfaces/modules.md` are authoritative.

## Inputs

The brief is built by code and placed in the output directory: `brief.md` (for people) and `brief.json` (the same content, structured: `module`, `purpose`, `round`, `task_json`, `ids`, `inputs`, `outdir`, `budget_left`, `goal` and more; the fields are described in `skills/rsi-orchestrator/templates/brief.md` under the framework repository root). The brief's `inputs` are authoritative: read only the paths listed there. When sent back, the same session gets `feedback-<k>.md` (`<k>` is the launch number), which says why it was sent back and what to change: fix that, then write `handback.json` again. The same session also gets two kinds of `feedback-<k>.md` that are not a return: one starting with `# Continue: the second part of the hand-back` (the second half of a two-part hand-back, e.g. train first hands back the trial candidates and judge the evaluation; once they are registered, it gets their registered ids here and then writes the search or the plan), and one starting with `# The user answered your questions` (see "Outputs").

List the artifact kinds (in backticks) and what each is for; read only the artifacts listed here, never the workspace or the registry. The brief also gives this module's `record`s of earlier rounds (only its own).

## Outputs

List the artifact kinds and the link to each one's field documentation; write each artifact as a subdirectory of the output directory, with null as the manifest's `id` and `round` as the brief says. Hand in a `record` for every improvement experiment.

Optional: at any time, rewrite the whole progress file `progress.json` in the output directory (the brief's `progress_file`); the progress page and the status line of `rsi run` show it as it is; the format is in [progress.md](../../docs/interfaces/progress.md). You may leave it out.

Hand-back: once every artifact is written and validated, write `handback.json` in the output directory: `status` (`ok`, `failed` or `long_job`), `artifacts` (subdirectory names in the output directory, one per artifact), `marker` (the absolute path of the long job's completion marker; required with `long_job`, else null), `notes` (gate numbers, cost, problems met; with `failed`, the reason) and `questions` (what the user must decide; `[]` when nothing). The hand-back counts only once the engine reads this file; a missing or malformed file counts as a failed hand-back, and writing it again correctly is enough. Reading is strict: no field other than those above, `artifacts` holds only subdirectory names (no path separator, no repeats), every item of `questions` is a non-empty string, `marker` is an absolute path or null, and with `failed` `notes` is not empty.

When the hand-back has `questions`, none of the artifacts of this hand-back is registered: the engine asks the user first; once the user answers, the answer goes back to the same session (`feedback-<k>.md` starts with `# The user answered your questions`, not a return). Change or confirm the artifacts by the answer, then write `handback.json` again, listing every subdirectory to register (those listed last time too). The `questions` of a long job (`long_job`) hand-back are asked only once the completion marker is there; when it lists no artifact, the answer comes in the feedback starting with `# Long job finished`, sent back together with the long job's results.

## Gates

The checks this module must pass before handing back, with values, thresholds and who checks them. Validate every artifact with `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`; compute the metrics, baselines and deltas you report with `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <label-rows-file> --predictions <predictions-file>`. The module may run only these two `rsi` commands.

## Running standalone

The owner runs the module in their own sandbox on copies of registered artifacts (paths given by hand) and checks the output with validate, without the orchestrator.

## Steps and strategy sub-skills

Filled in by the module owner: this module's steps and the list of its strategy sub-skills; which steps are scripts and which need judgement.

## Module metrics

Filled in by the module owner: this module's own metrics and how they are measured.

## Experiment records

Filled in by the module owner: hand in a `record` for every improvement experiment (fields in [record.md](../../docs/interfaces/record.md)).

## Long jobs

Filled in by the module owner: which steps are long jobs; start them in the background, write a completion marker (first line ok or failed), and do not poll once the marker path is handed back.

Hand-back of a long job: when it is expected to take longer than the brief allows, start it in the background, write `handback.json` at once with `long_job` as `status` and the absolute path of the completion marker as `marker`, then end the session; do not poll or wait. If the completion marker is already there when you would hand back `long_job` (the long job ended before your hand-back, e.g. one a previous session started): do not hand back `long_job`. The engine only accepts a marker written after this launch; an earlier marker never counts as present and the run would wait forever. Read the results, finish the work, then hand back. When the job ends, succeeded or failed, it must write the completion marker, whose first line is `ok` or `failed`. When the marker is `ok`, the engine has the same session go on (`feedback-<k>.md` starts with `# Long job finished`): it reads the results, writes the artifacts and hands back with `ok`; when the marker is not `ok`, this dispatch is recorded as failed.

Before handing back `long_job`, write `<marker>.pid` next to the completion marker: the first line is the long job process's own pid, the second its start time (Unix seconds, `date +%s`). The pid must be written by the long job itself, e.g. in a shell `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <command>' > <log> 2>&1 &` (`exec` makes the command keep this pid), and in Python write `os.getpid()` and `int(time.time())` as the job starts; never use the background pid the shell gives instead, which may be a wrapper process and not the long job itself. Preferably run the long job in its own process group (e.g. Python's `subprocess.Popen(..., start_new_session=True)`). Write `handback.json` only after `<marker>.pid` is written. When the run is stopped, the engine checks by this file that the process is this long job, then terminates it (the whole process group); without this file it cannot terminate it and can only list it as possibly still running, for the user to end by hand.

Use a new completion marker path for every long job (e.g. `<name>-<k>.done`, `<k>` one more each time), and run `rm -f <marker> <marker>.pid` before starting it: the engine only accepts a marker whose modification time is not earlier than this launch; an earlier marker at the same path does not count as this long job's end.

## Owner and version

Filled in by the module owner: the owner, the current version, the change log, known limitations.

## Common mistakes

Filled in by the module owner: the mistakes this module has made, their consequences and the right way (a table).
