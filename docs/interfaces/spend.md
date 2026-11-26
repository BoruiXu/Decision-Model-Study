# The spend ledger and the spend file `spend.jsonl`

The run keeps a ledger of four resources, each against its own optional budget in the task definition (task.md, `budget`). STATE's `budget_used` holds what has been used; a workspace made before the ledger has only `compute_hours` there and reads the other three as 0. The two dollar accounts are kept apart and **never added together**.

| Resource (`budget_used` key) | Budget key (`budget`) | Label | Counted from |
|---|---|---|---|
| `compute_hours` | `compute_hours` | GPU hours | The `cost.compute_hours` of registered and refused candidates (engine.md, "Compute of refused candidates"), and the `compute_hours` lines of spend files (e.g. GPU baselines, pool scoring). Hours a candidate manifest records in `cost.compute_hours` are counted from the manifest and are never written to a spend file, or they would count twice |
| `api_usd` | `api_cost` | data LLM $ | Only the LLM calls that generate or review data (the data generator and its reviewer, prompt-variant rewrites and their review): the `api_usd` lines of spend files |
| `agent_usd` | `agent_cost` | agent $ | Every agent session (`claude -p`), counted by the engine and never written to a spend file. Module agents: the `claude` runner reports each launch's `total_cost_usd`, which for a resumed session (`--resume`: returns, follow-ups, answers, check-ins) is the session's **cumulative** total, not the launch's own cost; so each launch adds the increase over its session's last counted total (the dispatch record's `spent_sessions`, session id → total counted; a new session, e.g. a fresh one taking over, starts from 0; a negative increase adds nothing), once per launch (`spent_launches`). Decision agents: each is its own `claude -p` call, never resumed; its cost is read from its log with the same parser and counted once per decision key (STATE's `decision_costs`, key → cost). A session that is killed or times out leaves no result object, so its cost is not known and not counted (it adds nothing and keeps the last total) |
| `annotation_items` | `annotation_items` | annotation | The `annotation_items` lines of spend files |

The resources, their keys and labels are `rsi.rules.RESOURCES`.

## The spend file

Every brief names `spend_file`, `<outdir>/spend.jsonl` in the dispatch's output directory. The module (its agent, or code it runs) appends one JSON object per line for each spend it causes outside its own agent session, except the hours a candidate manifest records in `cost.compute_hours` (counted from the manifest; never written here):

```json
{"at": "2026-11-26T08:15:00Z", "source": "llm:gen", "api_usd": 0.42, "note": "12 rows generated"}
{"at": "2026-11-26T08:40:00Z", "source": "compare:team_recipe", "compute_hours": 1.25}
```

| Key | Required | Rule |
|---|---|---|
| `at` | yes | A non-empty string: when (ISO 8601, UTC) |
| `source` | yes | A non-empty string: what spent it. An LLM call that generates or reviews data writes `llm:<tag>` (dgen's calls: `llm:gen` and `llm:analysis` for the generator, `llm:review` for the reviewer; `dgen paraphrase`: `llm:paraphrase` and `llm:paraphrase_review`); other work names itself, e.g. `compare:<baseline>`, `dgen:score-pool` (the data module's pool scoring with the best version's candidate) |
| `compute_hours` | no | A number >= 0: GPU hours. Never the hours a candidate manifest records in `cost.compute_hours`: those are counted from the manifest |
| `api_usd` | no | A number >= 0: dollars, **only** for LLM calls that generate or review data. Agent sessions are counted by the engine and never written here |
| `annotation_items` | no | A number >= 0: annotation items |
| `note` | no | A string, for people |

No other key is allowed. Write each line whole, ending with a newline (one `write` of the line and its `\n`, in append mode).

**Counting.** The engine counts the lines of a dispatch's spend file not counted yet (`Workspace.count_spend`) when it processes the dispatch's hand-back, when the dispatch fails, and when a stop or a close closes it (after any long job of it has been terminated, so its last lines are counted too); the dispatch record's `spend_lines` keeps how many lines have been read, so no line is counted twice. A budget guard must not fail open: every amount that is a finite number >= 0 is counted, also on a line whose other keys break the rules (`at` or `source` missing, a key not allowed); only a line that is not a JSON object, and an amount that is not a finite number >= 0, are not counted. Every line with a problem is logged as `spend_invalid` in `<ws>/work/run.jsonl` (`dispatch_id`, `file`, `line`, `error`, `counted`: the amounts counted from it, `text`) and counted in STATE's `spend_problems`, which the displays show. A blank line is read and skipped. A last line without its newline is read only once it parses (its writer may be halfway through it). The validator is `rsi.workspace.spend_line_problems`.

## Enforcement

- **The stop check.** The run stops when the use of any resource with a budget reaches it (`rsi.rules.reached`): the stop reason says `<label> used <u> reached budget <b>` for each (e.g. `data LLM $ used 20 reached budget 20`); `rsi next` stops the run once the round is decided. A budget of 0 for data LLM dollars, agent dollars or annotation items (e.g. `annotation_items: 0`, no annotation) is reached only once something is used; a GPU-hours budget of 0 is reached at once, as before.
- **`over_budget`.** When the task lists the `over_budget` approval and the round has none, an exhausted resource asks it before ① a trainset (a round after the first), ② and a new ③ dispatch (not before ③'s second part or a decision about its dispatch: ③ is report only); the question names every exhausted resource. Approved, the round goes on to its decision, then the stop check stops the run. A hand-back whose candidates would take the GPU hours over the budget is still held before registration as before (engine.md, "Holding a hand-back over the budget").
- **The brief.** `budget_left` gives what remains of each resource: `{"compute_hours", "api_usd", "agent_usd", "annotation_items"}`, each a number (never below 0) or null without a budget. A module stops its own spending at it (e.g. the data generator's LLM calls stop at `budget_left.api_usd`).
- **Data-LLM calls** (`modules/common/llm`, `ClaudeCall`): each call is one fresh `claude -p` process (a new session, so the result's `total_cost_usd` is its own cost) and appends one line `{"at", "source": "llm:<tag>", "api_usd"}` whatever its outcome, held from a stop until written. A call whose cost is not reported is billed with a `note` saying why: a killed one (it ran past its time limit, its caller was stopped, or a signal ended it) at the highest `api_usd` already recorded for its source in the spend file, else 1.00 USD, since it may have spent anything; one that could not start, or ended by itself without a result object (the CLI prints one whenever the model ran), at 0. Before each call it refuses to start (`LlmBudgetExceeded`) once the `api_usd` lines already in the dispatch's spend file reach `budget_left.api_usd`: every LLM call of the dispatch (all recipes, generators, reviewers, paraphrases) appends to that one file, so they share one account. A dgen recipe's `max_usd` caps what its generator and reviewer spend together. A recipe that reaches either cap stops short (`stopped: "api budget"`); its calls so far are billed.

## Display

`rsi status` prints the ledger on one line (`rsi.live.ledger_text`), each resource `used/budget` (`–` without a budget, dollars used with two decimals), then the count of spend-file lines with problems when there are any:

```
budget: GPU hours 1.3/8 · data LLM $ 4.20/20 · agent $ 31.50/– · annotation 0/– · spend-file problems 2
```

The status line of `rsi run` (and the `now:` line of `rsi status`) ends with the short form after ` | budget: `: only the resources with a budget or a use, and the problem count (`rsi.live.ledger_short`; nothing when there is none). The run report's "Attempts and budget" section has one line per resource and, when there are any, a line with the problem count; the overview page's "How far from the stop" card has one meter per resource (GPU hours always; the others once they have a budget or a use) and the problem count, and the events page one line per `spend_invalid`. The two dollar accounts are always separate fields.
