"""dpolicy: the decision policy of a candidate or a baseline, a calibration of its probabilities followed by a decision
rule, fitted on its valset predictions (standard library only, Python 3.9+).

- calibrate: fit and apply the calibrations none, platt, temperature and isotonic;
- rules: the threshold of a binary decision rule (max_f1, cost, precision_at, recall_at, top_frac, fixed, argmax);
- fit: the valset's fit and select parts, every calibration and rule fitted on the fit part and scored on the select
  part against the identity policy (rsi.metrics.paired_delta), and the policy chosen (policy.json);
- apply: a policy applied to one prediction row. It is self-contained (math and json only): the train command copies
  it into every candidate as policy_apply.py, which the candidate's predict.py loads by path.

Version 1 is a calibration and a threshold only: no abstain band, second stage or rolling refit. Bump
DPOLICY_VERSION with every change to how a policy is fitted or applied: it is part of the compare baselines'
spec_sha256 (compare.reuse.policy_key), so the baselines of policy tasks are fitted again.

Import it with `<repo root>/modules/common` on sys.path; fit uses the framework's rsi.metrics, so the repository root
is put on the path too.
"""
import os
import sys

DPOLICY_VERSION = "1.0.0"

# The repository root (three levels above this package's directory): fit.py uses rsi.metrics and rsi.rows.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
