# Delivery artifacts: delivery, feedback

Both kinds are produced by deliver, and their directories hold a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`: the manifest is checked against the common fields plus the kind's extra fields (no fields that are not listed). The formats of prediction rows and execution feedback rows are in "Decision rows" of README.md. Ids are only checked to be strings; whether they are registered is checked at registration.

## Prediction interface

The delivery package offers one prediction interface:

- Input: one sample's `state` (the input row's `state`, formatted by the task definition's `input.format`).
- Output: `{"probs", "decision", "uncertainty"}`, where `probs` maps labels to probabilities, `decision` is one label and `uncertainty` is a number or null.
- Every call writes one execution feedback row, appended to `trace`: the fields `id`, `state` (this call's input, i.e. the input row's `state`; an old workspace's trace calls it `input`, which registration still reads), `output` (`probs`, `decision`), `model`, `latency_ms`, `timestamp`, `outcome`. `outcome` is null while the real outcome is unknown, and a label once it is known. The `id` of an execution feedback row is this call's id, in a format deliver chooses (it need not equal the input row's id), as long as it does not repeat within one `trace`.

The form of the interface (command line, function, HTTP) is up to the deliver module and is written in `package.entry`. The delivery package usually wraps the candidate's prediction entry `model.predict` (see models.md), but to the outside `package.entry` is what counts.

## `delivery`

Packages a candidate as a callable service. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"delivery"` | Artifact kind |
| `candidate_id` | string | The candidate delivered |
| `evalset_id` | string | The evalset of `predictions_eval`; must be the current evalset |
| `valset_id` | string | The valset of `predictions_val`; must be the candidate's `valset_id` (see "Reproducing the candidate's valset predictions") |
| `testset_id` | string, optional | Required when the testset is used |
| `package` | object | The delivery package, see below |
| `runtime` | object | Runtime metrics, see below |

The sub-fields of `package`: `location` (where the delivery package is, see below); `entry` (the call entry, stating the interface's form and how to call it); `environment` (a description of the runtime environment).

The sub-fields of `runtime`:

- `latency_ms_p50`, `latency_ms_p95`: the median and 95th percentile, over the evalset inputs, of the wall-clock time (milliseconds) of each call of the prediction interface, one sample per call; the time to load the model is not counted.
- `cost_unit`: string, optional, the unit of the cost, e.g. `usd`, `gpu_seconds`.
- `cost_per_1k`: a number or null, the cost of 1000 predictions in `cost_unit`; null when unknown.
- `consistent_with_offline`: bool, whose meaning is fixed as the reproduction check below: the delivery package's predictions on the valset inputs agree with the candidate's `predictions_val`.

Required files: `predictions_eval` (the delivery package's predictions on the evalset's `inputs.jsonl`) and `predictions_val` (the delivery package's predictions on the valset's `inputs.jsonl`); optional `predictions_test` and `trace`, validated only when they appear in `files`. Checks inside the directory: valid prediction rows; an error when `runtime.consistent_with_offline` is false; a valid `package.location` (see below); `testset_id` required with `predictions_test`; valid execution feedback rows when there is a `trace`.

**The delivery package is self-contained.** `package.location` may take only one of two forms:

- A path relative to the delivery directory (file or directory) that, as written, does not leave the delivery directory (does not go out through `..`) and exists. At registration the whole delivery directory is copied into the workspace, and the delivery package with it into `<ws>/artifacts/<delivery id>/`, so after registration the delivery package no longer depends on the module's working directory.
- `external:<name>`: the entry of the same name in the common field `external`, which carries a location and a sha256, for a package too large for the directory. An entry on this machine (`location` `/absolute/path`) must be a single file, e.g. a tar archive, since validation and registration compute its sha256; pack a directory before registering it. A remote entry (`host:/path`) is not checked.

Absolute paths are always refused. The delivery directory may contain symlinks (e.g. a `python3 -m venv` environment in the delivery package linking to the system interpreter): registration copies what the link points to, and the registered delivery package holds the actual files and no longer depends on the link's target. A link to a file that does not exist is always refused. The model package `rsi report` lists is the registered path `<ws>/artifacts/<id>/<location>`, or the location and sha256 of the `external` entry.

**Reproducing the candidate's valset predictions.** deliver gets the `inputs.jsonl` of the candidate's valset (without labels), runs `predictions_val` through the delivery package's prediction interface, and writes this valset in `valset_id`. Registration checks:

- `valset_id` is registered and equals the candidate's `valset_id` (the candidate's `predictions_val` are on that valset).
- `predictions_val` covers exactly that valset's ids.
- Each row agrees with the row of the same id in the candidate's `predictions_val`: each label's probability differs by at most the task definition's `delivery.reproduction_tolerance` (1e-6 when not written, see task.md; `uncertainty` is not compared), and `decision` is the same.
- Rows whose probabilities are all within the tolerance and only `decision` differs (rows near the decision boundary) may number at most floor(`delivery.max_decision_mismatch_rate` × the valset's row count), 0 rows when not written; when there are such rows and they are let through, `rsi register` appends `(decision mismatches: <rows>)` to its output line. Rows whose decision differs and some probability is outside the tolerance are not covered by this loosening and are refused as usual.

`runtime.consistent_with_offline` refers to exactly this check. When the reproduction fails, registration refuses even if the flag says true, the error giving the number of rows that disagree and the differences of the first few. So the delivery package really is that candidate, and its results on the evalset can be credited to the candidate.

Registration also checks: `candidate_id` is registered and passed the train gate (`train_gate.passed` true; otherwise refused, saying the train gate was not passed); `evalset_id` is the current evalset, and `predictions_eval` covers exactly its ids; with `predictions_test`, `testset_id` is the current testset, the predictions cover exactly its ids, and this round has a `use_testset` approval not yet used.

## `feedback`

Execution feedback collected online or in a trial run. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"feedback"` | Artifact kind |
| `delivery_id` | string | The delivery the feedback comes from |
| `candidate_id` | string | The candidate the feedback comes from |

At least one of `delivery_id` and `candidate_id` must be given. Required file: `trace`. Checks inside the directory: every row is a valid execution feedback row (`outcome` must be a label or null).
