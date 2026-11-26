# Decision Model RSI framework design (top level)

> Training and tuning have since been merged into ② Train & tune, and ③ Compare (a code module, report only, on with the task's `compare` section) now comes between it and delivery: the main flow is ① Data → ② Train & tune → ③ Compare → ④ Deliver → ⑤ Judge & decide. The five-module numbering in this document is that of the time.

> This document is the original layered design (the modules scheduled by the main flow table in the top-level skill). The main flow is now written in code and run by the flow engine of `rsi run`, and the top-level skill is only a thin operator skill: see [plan.md](plan.md) and [interfaces/engine.md](interfaces/engine.md); where this document describes the top-level agent's scheduling order and dispatching subagents, those documents take precedence.

- Date: 2026-11-21
- Status: confirmed (2026-11-21, the user asked to start building from this document)
- Scope: the top-level design, and the inputs and outputs of the top level and of each module. The exact formats of the hand-over artifacts and the validation scripts are written separately once this document is confirmed.

## 1. What this is

This is a general, layered skill framework that builds a decision model from a one-sentence request and keeps improving it. The user describes a decision task in a prompt, for example "given a transaction, decide whether to allow it, block it or send it to a person". The top level turns the request into a task definition and then schedules five modules: data, train, tune, deliver, and judge. Every turn of the loop produces a candidate version; the judge module decides whether it is better than the current best version, analyses the failures and proposes what to change next round. The top level follows the rules until the target is reached or the budget is used up, and finally gives the best version to the user. This loop of continuous improvement is RSI.

The framework is not tied to any particular task, data source, machine or model backend; the task definition of each run gives those. Each of the five modules is a skill, written and maintained by a different owner. This document defines what the top level does, what each module does, and what they hand over to each other.

## 2. Goals, non-goals, anti-goals

**Goals**
- Go all the way from the user's prompt to delivering a callable decision model, and enter the next round of improvement automatically.
- Clear responsibilities, inputs and outputs for the top level and the five modules, so that a module owner who keeps the inputs and outputs can change anything inside without affecting anyone else.
- One framework for three decision types: binary, one of several, and scoring.
- A place for each of the three levels of RSI that Prof. He proposed (System, Component, Meta).

**Non-goals**
- No prescribed method inside the modules, e.g. how data is generated, which backend the model uses, how parameters are searched, how the interface is deployed.
- No prescribed rhythm of team collaboration. Each run of the framework is one complete flow.
- The first version does not aim at full automation. Key points such as the task definition, the next round's plan and using the sealed testset are confirmed by a person.

**Anti-goals** (failures even when everything runs)
- The framework hard-codes the assumptions of a particular task.
- The top level does a module's work itself.
- A module that produces models can see or change the criteria that judge it.
- A module cannot run without the top level, and its owner cannot improve it alone.

## 3. Choice of approach

Three **ways of organising** were considered. A is central scheduling with modules as mere workers being called: the simplest, but owners cannot evaluate and improve their modules alone. B is central scheduling plus a common module template, each module stating its inputs, outputs, gates, module metrics and how to run it standalone. C is decentralised, each module watching a shared artifact store and running when a new input arrives: the most decoupled, but it needs a separate trigger mechanism, and errors are hard to trace. B was chosen: it can go all the way from a one-sentence request to delivery, and lets each owner improve their own module alone.

Two options were considered for **where judgement and decisions live**. One is all at the top level: fewer hand-overs, but this is exactly the part that most needs continuing research and improvement, and at the top level no owner would do its Component RSI, while the top level itself should be stable and change little. The other keeps the rules at the top level and gives judgement and decisions to a dedicated module. The latter was chosen, and the evalset's labels are kept only by this module; the deliver module only runs predictions on the inputs and never sees the labels. So referee and player are kept apart, and the modules that produce models cannot tune against the evalset over and over.

## 4. Overall architecture

```
Layer 0  top-level orchestrator: sets the rules, schedules, executes
Layer 1  five modules: ① Data  ② Train  ③ Tune  ④ Deliver  ⑤ Judge & decide
Layer 2  strategy sub-skills each module owner splits out
```

From the user's prompt to the final result:

```
User prompt
   │
   ▼
Top level: task intake → task definition (with promotion rule, stop conditions, budget), confirmed by the user
   │
   ▼
① Data ──► trainset; round 1 also produces the valset and the evalset, the evalset handed to ⑤ to keep
   │
   ▼
② Train ◄──► ③ Tune ──► candidate model
   │
   ▼
④ Deliver ──► model package and prediction interface; predictions run on the evalset inputs only; latency and cost; execution feedback
   │
   ▼
⑤ Judge & decide ──► scores, compares with the current best version → recommendation to promote or keep
                     failure analysis and attribution → next round's plan
   │
   ▼
Top level: checks and executes by the promotion rule, registers versions, checks the stop conditions
   ├─ stop → to the user: the run report, the current best version's model package, prediction interface and evaluation report, all experiment records
   └─ go on → back to one of ①, ②, ③, ④ by the plan
```

## 5. What the top level does

The top level does only three kinds of things; it does none of a module's work and no analysis that needs judgement.

**Coordinate.** Turn the user's request into a task definition, by which the four producing modules and the judge module all work. Set what each hand-over holds, validating each hand-over before passing it on. Decide which module runs next, give it its inputs, take back its outputs, and wait in the background for long jobs. Allocate budgets such as compute, annotation and call costs to the modules, and stop at the points that need a person's confirmation.

**Keep the rules.** The task definition, the promotion rule, the stop conditions, the frozen valset and evalset, and the sealed testset are all fixed by the top level, and no module may change them. Changing the task definition needs the user's confirmation and makes a new version. The top level also keeps the version registry: which data, which config and which model each version used can all be traced.

**Execute.** Given the judge module's recommendation, check the numbers against the task definition's promotion rule, and only when they agree promote or keep. How metrics, baselines, deltas and confidence intervals are computed is fixed by code (`rsi/metrics.py`), not decided by each module: the train and judge modules compute the numbers they report with `rsi score`, and at registration the code recomputes them by the same definitions and refuses a mismatch. So the train gate and promotion decisions both use numbers the code has checked, and numbers from different modules can be compared directly. In round 1 there is no current best version yet, so a candidate model that passes the train module's gate becomes the first version directly, and the evaluation report is produced as usual. The next round's plan goes to the user for confirmation at the approval point and is then dispatched to the module the plan names. When a stop condition holds the run stops, and the run report and the results go to the user. The top level also collects and registers the modules' experiment records.

**Stop conditions.** The run stops when any one holds: the current best version's primary metric reaches the target (`stop.target`); the compute budget is used up (`budget.compute_hours`); the completed rounds reach `stop.max_rounds`; `stop.patience` rounds in a row without promotion (3 when not written); the user asks to end. The code checks the first four at the end of each round (`rsi next`); the user may ask to end at any time, and the top level records the user's own words with `rsi stop`. After a stop, only experiment records may still be registered and the cleanup after a delete approval done.

**Attempts per round and closing a round.** When a round's candidates fail the train module's gate, the top level dispatches train and tune again in this round, or has the data module produce a new trainset, until a candidate passes. Each round tries at most `stop.max_attempts_per_round` times (3 when not written; a candidate produced by train and a search by tune each count once, and the trials within a search do not count separately), and registrations beyond that are refused. When they are used up and still no candidate passed the train gate, the round is closed (`rsi close-round`): it is recorded as keep, counts as a round without promotion, and counts in `max_rounds` and `patience`; the judge module writes the next round's plan from why this round's candidates failed and from the experiment records. So even if every round fails, the run stops after at most `patience` rounds and never gets stuck in a round.

## 6. What the five modules do

| Module | In Prof. He's terms | Responsibilities |
|---|---|---|
| ① Data | Data generation, data quality, data checks | Get or generate data, check and improve its quality; produce the valset and the evalset in round 1 |
| ② Train | Model, model architecture | Train the decision model; provide a train command that needs no agent, for tune to call |
| ③ Tune | Hyperparameter tuning | Search better configs and training strategies using past experiments |
| ④ Deliver | Runtime system | Package the model and the prediction interface; run predictions on the evalset inputs only; measure latency and cost; collect execution feedback |
| ⑤ Judge & decide | Evaluation, RSI Controller | Keep the evalset's labels; score and compare; recommend promotion; analyse and attribute failures; propose the next round's plan |

## 7. Data split

| Set | Produced by | Who sees the labels | Use | Changes |
|---|---|---|---|---|
| Trainset | ① | ②, ③; ⑤ uses it only for the baselines | Training | May have a new version every round |
| Valset | ①, round 1 | ②, ③, ①; ④ gets only the inputs (without labels), to reproduce the candidate's valset predictions | Choosing checkpoints, setting thresholds, comparing configs, dedup checks | Fixed after round 1; a refresh needs the user's approval |
| Evalset | ①, round 1, handed to ⑤ once produced | ⑤ only | Promotion decisions | Fixed; the number of promotion decisions it takes part in is capped, and a refresh after the cap needs the user's approval |
| Sealed testset | ①, round 1, handed to ⑤ once produced | ⑤ only, and every use needs the user's approval | Final check at a milestone release | Never changes |

The valset stays fixed so that the modules' comparisons of old and new use the same ruler. The evalset goes only to ⑤, because it decides promotion and must be kept apart from the modules that produce models. Although the data module produces the evalset in round 1, it keeps no access once it hands it over: at registration the evalset's and testset's directories are moved into the workspace whole, and the data module's output directory keeps no labels (reads on the same machine are held back by convention, not prevented by code); after that, each round the top level checks that the trainset does not repeat the valset, the evalset or the testset. After an evalset refresh, the current best version must be scored again on the new evalset, so the current best version's model is kept until a new version replaces it.

## 8. Inputs and outputs of each part

Every module meets three conditions: it reads only its inputs; what it hands over passes the top level's validation; it can run standalone on existing upstream artifacts without the whole flow. For every improvement experiment, each module also hands in an experiment record (section 9).

### 8.1 Top level · task intake

| Item | Content |
|---|---|
| Input | The user's request: what decision to make, where the data is, what resources may be used, what level to reach |
| Output | Task definition: decision type; actions or labels and their meanings; the input the model can see; data sources; primary and other metrics; baselines; target; promotion rule; stop conditions; budget; resources; approval points |

When information is missing, ask all at once. Once the user confirms, the task definition is fixed.

### 8.2 ① Data

| Item | Content |
|---|---|
| Input | Task definition. Optional: the previous trainset; the issues of the next round's plan attributed to data; execution feedback with real outcomes; error slices from ⑤ |
| Output | A new version of the trainset and a data quality report. In round 1 or an approved refresh, also the valset, the evalset and the sealed testset |
| Gate | The data quality checks the task definition sets, e.g. sample size, label distribution, label reliability, near duplicates |
| Module metric | This round's evaluation against the current best version: the primary metric delta and the promotion decision of the candidate trained on the new trainset against the best version (trained on the old trainset) on the evalset, recorded by ⑤. The data module does not train and does not check the new trainset itself (since 2026-11-25; before that it trained once on the new and once on the old trainset and compared the valset primary metric) |

### 8.3 ② Train

| Item | Content |
|---|---|
| Input | Task definition; trainset; valset; training config (the default config, or one given by ③) |
| Output | Candidate model (model files, a directly runnable prediction entry, the full training config, the training code's version, the version of the trainset used); predictions and metrics on the valset; training log |
| Gate | The primary metric on the valset strictly better than every baseline the task definition lists (direction by the task definition; equal is not better) |
| Module metric | With the trainset fixed, the gain over the previous training method |

The train module also provides a train command that needs no agent: trainset, valset and config in, candidate model out. Every trial of tune calls it; the data module does not call it (since 2026-11-25 the data module does not train, and train and judge judge whether a new trainset is good).

### 8.4 ③ Tune

| Item | Content |
|---|---|
| Input | Task definition; trainset; valset; past experiments (existing candidate models and search records); budget |
| Output | The best config and its candidate model; the search record (each trial's config, result and cost) |
| Gate | The best config not worse than the default config on the valset; when that cannot be met, hand back the default config's result and record it truthfully. All trials of a search come from the same version of the training code; changes to the training code and the default config go between rounds |
| Module metric | The gain over the default config for the same budget |

### 8.5 ④ Deliver

| Item | Content |
|---|---|
| Input | Candidate model; the evalset's inputs (without labels); the inputs of the candidate's valset (without labels); at a milestone release, with the user's approval, the sealed testset's inputs |
| Output | Model package (model, threshold, runtime environment, how to start it; in the delivery directory or an external entry with sha256); prediction interface; predictions on the evalset and the valset; latency and cost; execution feedback |
| Gate | The model package loads; the interface's output on the valset inputs reproduces the candidate's valset predictions (checked row by row by code at registration) |
| Module metric | Latency, cost, whether packaging succeeds at the first try |

The prediction interface is the minimal Runtime in the delivery. It takes one sample's input, returns each action's probability, the chosen action and the uncertainty, and writes one execution feedback row for every call: input, output, model version, time taken, and the real outcome learned later (if any). Execution feedback with real outcomes can go back to the data module and become new training samples.

### 8.6 ⑤ Judge & decide

| Item | Content |
|---|---|
| Input | Task definition (metrics, baselines, promotion rule); evalset (with labels); the predictions, latency and cost ④ handed over; the current best version's results and predictions; the trainset of the candidate sent to evaluation (only for the baselines); execution feedback; all experiment records |
| Output | Evaluation report: each metric, the comparison with the current best version and the baselines with intervals, results by group and slice. A recommendation to promote or keep, with evidence. Failure analysis: where it goes wrong and which module it is attributed to. The next round's plan: which module to go back to, what to change, how much gain to expect |
| Gate | The recommendation must cite the promotion rule and the matching numbers; every issue in the plan has evidence and exactly one attribution |
| Module metric | The plan's hit rate: how much of the gain the plan expected really came true |

Attributions map to the module re-entered as follows: insufficient data coverage, noisy labels or a problematic label definition go back to data; too little model capacity or poor calibration go back to train; a poor config goes back to tune; latency, cost or interface problems go back to deliver.

### 8.7 Top level · closing the loop

| Item | Content |
|---|---|
| Input | ⑤'s recommendation and plan; budget used |
| Output | Version registry (the current best version updated on promotion); dispatching the next round; or stopping, handing over the run report and the final result |

## 9. Hand-over artifacts

| Artifact | Kind name | Produced by | Used by |
|---|---|---|---|
| Task definition | `task` | top level | all |
| Trainset (versioned) | `trainset` | ① | ②, ③; ⑤ only for the baselines |
| Valset | `valset` | ① | ①, ②, ③; ④ only the inputs |
| Evalset | `evalset` | produced by ①, kept by ⑤ | ⑤; ④ only the inputs |
| Sealed testset | `testset` | produced by ①, kept by ⑤ | ⑤; ④ only the inputs, and with the user's approval |
| Candidate model | `candidate` | ② (③ produces them through the train command) | ③, ④ |
| Search record | `search` | ③ | ⑤, top level |
| Model package, prediction interface, predictions on the evalset, latency and cost | `delivery` | ④ | ⑤, the user |
| Execution feedback | `feedback` | ④, or the interface once connected to a real system | ①, ⑤ |
| Evaluation report and promotion recommendation | `evaluation` | ⑤ | top level |
| Next round's plan | `plan` | ⑤ | top level, then dispatched to the module concerned |
| Experiment record | `record` | every module | ⑤, top level |

**Experiment records** are the material of Meta-RSI. Each holds: the problem found, why the change was expected to work, what was changed, the result (the values before and after the change, and whether the conclusion is success, failure or inconclusive), what was learned, and whether this step is now done by hand, by script or automatically.

The exact format and validation script of each artifact kind are written separately once this document is confirmed.

## 10. Three levels of RSI

**System RSI** is the top level's loop. Every turn produces a candidate version, and ⑤ decides whether to promote it, so the current best version only gets better or stays.

**Component RSI** is each module improving itself by its own module metric (section 8). An owner can run their module standalone on existing upstream artifacts, and hand it back to the framework once improved.

**Meta-RSI** is learning from the improvement process itself. From the experiment records ⑤ works out which improvement strategies work in which situations, to propose more accurate plans; recurring lessons go to the module owners to change their skills, or become automatic rules.

## 11. Fixed rules and approval points

These are fixed by the top level and no module may change them: the task definition (with the promotion rule and stop conditions), the valset, the evalset, the sealed testset, the version registry.

Default approval points: confirming or changing the task definition; expected use over the budget; using the sealed testset; refreshing the valset or the evalset; carrying out the next round's plan (confirmed every round in the first version); deleting artifacts.

**Delivery at the stop.** At every stop, whether a stop condition held or the user asked, the top level produces the run report with `rsi report`. The report is generated by code with fixed content: the task, the stop reason, each round's decision and reason, the versions and the current best version (with the locations of the model package and the evaluation report), milestone evaluations, the experiment records grouped by module, each round's attempts and the budget used. After the report the top level adds the interpretation code cannot give: what was learned and what to try next, only relaying existing evaluations, plans and experiment records. Then it gives the report to the user together with the current best version's model package, prediction interface and evaluation report. When not a single version was promoted, the report opens by saying that no releasable version was produced and lists each round's conclusion, and goes to the user as a failure report.

## 12. The structure of the layered skills

| Layer | Skill | Maintained by |
|---|---|---|
| Layer 0 | `rsi-orchestrator` | the framework |
| Layer 1 | `rsi-data`, `rsi-train` (train module; the former `rsi-tune` has been merged into it), `rsi-deliver`, `rsi-judge`, with fixed names; each in `modules/<module>/SKILL.md` | each module owner |
| Layer 2 | Strategy sub-skills, e.g. under the data module by label source (rules, model judging, human annotation), under the train module by model backend | each module owner |

The top level loads only layer 1, and layer 1 picks layer 2 by the task definition. Layer 1 skills follow one template: inputs, outputs, gates, how to run standalone, steps and strategy sub-skills, module metrics, when to write experiment records, how long jobs run in the background, owner and version. The top level depends only on the first four; owners may change the rest at any time.

## 13. How the top level is implemented

**Who runs it.** The top level is a workflow skill, `rsi-orchestrator`. The user opens a session in Claude Code and states the request directly, or calls `/rsi-orchestrator`; once the main session loads this skill it becomes the top-level agent. After a stop midway, calling the same skill again reads the workspace's progress files and goes on from where it stopped.

**How modules are dispatched.** Only the main session can dispatch subagents, and a subagent cannot dispatch further subagents. So the top level must be the main session; each module is run by a subagent the main session dispatches, which loads the matching layer 1 skill. Modules cannot dispatch agents to each other; when one needs another module's capability it calls a command that module provides, e.g. tune calls the train module's train command.

**Scheduling order.** The main session dispatches modules one after another by the main flow table in the workflow skill, never all at once. Each round's order is ① → ② and ③ → ④ → ⑤ → top-level check. ② and ③ depend only on the trainset and the valset and can be dispatched together; with several candidate models, ④ can package them together. The next round starts at the module the plan names, without dispatching all five modules. Long jobs are started in the background by the subagent, which then hands back; the main session waits for the completion marker and does not poll.

**Scripts as gatekeepers.** A skill is an instruction to an agent, and agents may stray, so every mechanical judgement is made by the framework's command-line tool `rsi`, which agents can only call: validating hand-over artifacts (`rsi validate`), creating the workspace (`rsi init`), registering artifacts and checking each round's attempts (`rsi register`), viewing the state (`rsi status`), computing metrics, baselines and deltas by the definitions fixed in code (`rsi score`, which modules may run too), checking evaluation results against the promotion rule and registering versions (`rsi promote`), closing a round without a candidate to evaluate (`rsi close-round`), recording approvals (`rsi approve`), checking the stop conditions (`rsi check-stop`), checking the stop and moving the round on (`rsi next`), recording a stop the user asks for (`rsi stop`), producing the run report (`rsi report`). The scripts guard each other: an artifact that failed validation cannot be registered; registration recomputes the metrics, baselines, train gate, deltas and intervals of candidates and evaluation reports by the same definitions and refuses a mismatch; an evaluation result not registered cannot promote a version; a plan not approved cannot dispatch the next round. So even if an agent skips a step, the flow cannot go on.

**Formats.** Artifacts machines read and write are always JSON and JSONL, and descriptions for people are Markdown. `rsi` depends only on the Python standard library and runs on any machine with Python 3.9 or later.

## 14. Next steps

1. Write the format and validation script of each kind of hand-over artifact.
2. Write `rsi-orchestrator`, the template of the layer 1 skills and a first version of the five module entries (with only the four items inputs, outputs, gates and running standalone filled in).
3. Pick two example tasks with different decision types, one binary and one of several, and go from prompt to delivery to verify the framework.

## 15. Open questions

1. The default promotion rule. With a small evalset, requiring the lower bound of the "candidate − current best" interval to be above 0 is hard to pass; whether the framework provides a default rule, and which, is open. Each task may still set its own in its task definition.
2. The two example tasks for verification and their data sources.
