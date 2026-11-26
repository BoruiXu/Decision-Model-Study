# Experiment records: record

Each module produces a `record` after it completes a change; it is the material of Meta-RSI (section 9 of the design document). Its directory holds a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`, which checks only the fields' types and values; there are no extra checks inside the directory and no required files.

Each record answers: what problem was found, why the change was expected to work, what was changed, what the result was, what was learned, and whether this step is now done by hand, by script or automatically. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"record"` | Artifact kind |
| `module` | enum `data`, `train`, `compare`, `deliver`, `judge`, `orchestrator` (an old workspace also has `tune`, read only, under ② Train & tune) | The module the record belongs to |
| `owner` | string | Owner |
| `failure` | string | The problem found |
| `hypothesis` | string | Why the change was expected to work |
| `action` | string | What was changed |
| `result` | object | The result, see below |
| `knowledge` | string | What was learned |
| `automation` | enum `manual`, `scripted`, `automated` | How this step is done now |
| `artifacts` | list of strings | Ids of related artifacts |

The sub-fields of `result`:

- `metric`: the metric's name, saying on which set it was measured, e.g. `val accuracy`, `val auc`; `accuracy` alone does not say whether it is the valset or the evalset.
- `before`, `after`: the values before and after the change (number or null), which must be measured on the same set with the same metric to be comparable. `before` is this experiment's starting point: the best result before, the default config's result, or the result trained on the old data, whichever the experiment compares against; `after` is the result after the change. Write null when it cannot be measured.
- `conclusion`: `success`, `failure` or `inconclusive`.
