"""compare: the compare module (③ Compare), a code module the engine runs as a fixed command with no agent.

It trains the task's baselines once per trainset, has each predict the valset and the evalset inputs, hands them
back as `baseline` artifacts and then the `comparison` of the round's candidate with them on the valset
(modules/compare/SKILL.md is the contract). Run it as `python -m compare run --brief
<brief.json> --outdir <dir> --mode <mode> [--feedback <file>]` with `<repo>/modules/compare`,
`<repo>/modules/common`, `<repo>/modules/train` (trsi) and `<repo>` on PYTHONPATH; `modules/compare/run` does that in
the task's interpreter.

- inputs: the brief's inputs (Inputs.load);
- reuse: each baseline's spec, spec_sha256 and plan (reuse a registered baseline, re-predict it, or train);
- cpu: the CPU baselines lr, gbdt and mlp (scikit-learn, xgboost or lightgbm, imported when used);
- gpu: the GPU baselines zero_shot, few_shot and team_recipe, run as one long job (`python -m compare gpujob`);
- milestone: the baselines of the best version's comparison scored on the testset inputs (milestone_compare);
- report: the comparison (the candidate against each registered baseline on the valset, report.md);
- preds: prediction rows;
- write: baseline directories, skipped.json, handback.json and spend lines;
- cli: the command line.

The code keeps to Python 3.9 syntax; the pinned libraries need 3.11 (the ojh venv's). Every number it writes is
computed with rsi.metrics, as registration recomputes it.
"""

COMPARE_VERSION = "1.0.0"
