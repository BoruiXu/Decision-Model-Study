# The module progress file `progress.json`

The framework reads no training- or trsi-specific files. A module that wants the pages and the command line to show its progress writes `progress.json` in **this dispatch's output directory**, and the framework shows what it gives as it is. Whether to write it is up to the module; without it the progress is simply not shown, and hand-backs and registration are not affected.

```json
{
  "version": 1,
  "updated_at": "2026-11-24T12:46:23Z",
  "summary": "training 3 trials (GPUs 0, 4, 6)",
  "facts": [{"label": "configs", "value": "3/3"}, {"label": "GPU hours", "value": "1.3/8"},
            {"label": "incumbent", "value": "c4339e55d6 (lr 5e-6)"}],
  "table": {"title": "Trials of trsi run d-0001",
            "columns": ["trial", "changes from the start", "seed", "state", "GPU", "time", "valset auprc"],
            "rows": [["c4339e55d6-s1", "lr 5e-6", 1, "training step 22/32 · about 3 min left", "0", "6 min", null]]},
  "notes": ["next: the lr 5e-6 probe passed the margin; add seeds 1, 2"]
}
```

| Key | Required | Rule (the reader: `read_progress` in `rsi/progress.py`) |
|---|---|---|
| `version` | no | Left out or 1; any other value: the whole file is not used, and a problem is recorded |
| `updated_at` | yes | UTC ISO 8601 (`Z` or with a time zone); missing or not parseable: shown as usual, a problem recorded, staleness not judged |
| `summary` | no | A string, one sentence; over 200 characters it is clipped with "…" |
| `facts` | no | `[{label, value}]`, label a string ≤ 40 characters, value a string or number ≤ 80 characters; at most the first 12 are used; bad items are skipped and a problem recorded. **The order matters**: the command line's status line uses only the first 3 |
| `table` | no | `{title?, columns, rows}`: columns 1–12 strings; at most the first 100 rows are used, each a list (padded with `null` or clipped to the column count), cells strings, numbers, booleans or null, each cell ≤ 200 characters |
| `notes` | no | A list of strings, at most the first 20 used, each ≤ 500 characters |

- The file is a UTF-8 JSON object of at most 256 KiB (a larger one is not read, and a problem recorded); other keys are ignored; the writer replaces it atomically with a temporary file and `os.replace`. When it cannot be read, is not an object or its JSON is broken: no progress is shown, only one problem, and it **never makes the pages or the engine fail**. All text is escaped on the pages.
- **Stale**: while the dispatch is alive (`running`, or `long_job` with the completion marker not there yet) and `now − updated_at > PROGRESS_STALE_SECONDS = 300` seconds, the pages and the status line mark "no progress update for N minutes".
- Brief: the `progress_file` of `brief.json` is the absolute path of `<output directory>/progress.json`, and `brief.md` says in one line that it is optional and that the format is in this document.

## How the reader shows it

`rsi.progress.read_progress(path, now)` returns None when the file does not exist; otherwise it returns `path`, `updated_at` (an aware time or null), `age_s` (seconds since `updated_at`, or null), `summary`, `facts` (a list of (label, value)), `table` (`title`, `columns`, `rows`, or null), `notes` and `problems` (one sentence per rule broken).

- Text over a limit is clipped with "…" at the end, which is no problem; `facts`, `rows` and `notes` beyond their counts are not used, which is no problem either.
- Values of cells and facts: strings as they are, numbers written with `%.6g`, booleans as "yes" or "no", `null` shown empty.
- Over 12 `table.columns`: only the first 12 are used, and a problem recorded; when they are not a list, are empty or have a column name that is not a string, the whole table is not used, and a problem recorded. A row that is not a list is skipped, and a cell of the wrong type is shown empty, each recording a problem.
- `rsi.progress.dispatch_progress(ws, d)` reads the file in dispatch `d`'s output directory and adds `stale` (the "Stale" judgement above).

## Example: the train module

`trsi init --brief` records the brief's `progress_file` in run.yaml (`framework.progress_file`; `<output directory>/progress.json` when the brief has none); it is written after `trsi launch`, at every poll of `trsi watch` (in the background, writing all through the training), and before and after `trsi collect` packs each trial. The train command `modules/train/train` writes `<parent directory of --out>/progress.json` at its start, at every poll during training and at its end (when two train commands run in the same directory at once, the later write overwrites the earlier). A write failure only goes to the log and **never** makes watch, collect or the train command fail. Content:

- `summary`: how many trainings run and on which GPUs, how many wait to start, which trial is being packed;
- `facts` (in this order): `configs`, configs proposed / `search.max_configs` (counted per dispatch); `GPU hours`, spent / the brief's budget; `incumbent`, the incumbent's candidate id and its changes from the start;
- `table`, one row per trial: trial, config changes from the start ("start" when the same as the start), seed, state (`waiting to start` / `training step k/n · about m min left`, read from ojh's training log / `packing (GPU i)` / `done, not packed` / `done` / `failed: <first 80 characters of the reason>`), GPU, time, the primary metric on the valset;
- `notes`: the reason for STOP (the same line `trsi next` prints) and the packing errors of unpacked trials.

The training log is `<ojh run directory>/train-s<seed>.log`, and a step line looks like `train [###...] 22/32 (68.8%) loss … eta 0.1h`; the last complete such line in the last 64 KiB is read (ANSI colour codes removed; a part after the last newline that is still being written is not read), and "waiting to start" is written when none can be read. Time left: (n − k) × s/step when the line has `s/step`, otherwise `eta`. The time runs from the latest launch to when ojh writes `DONE` (to now while still running).
