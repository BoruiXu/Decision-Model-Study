"""The comparison, the second part of a compare dispatch: the candidate against each registered baseline on the valset,
every number as `rsi score` prints it."""
import json
import os

import pytest

pytest.importorskip("sklearn")

from conftest import (handback, manifest, rsi_score_compare, rsi_validate, write_registered)  # noqa: E402
from compare.cli import main  # noqa: E402


def test_comparison_after_baselines_registered(tmp_path, brief, task):
    main(["run", "--brief", brief, "--outdir", str(tmp_path), "--mode", "first"])
    write_registered(tmp_path, 2, {"bl-lr": "baseline-0001", "bl-gbdt": "baseline-0002", "bl-mlp": "baseline-0003"})
    main(["run", "--brief", brief, "--outdir", str(tmp_path), "--mode", "followup",
          "--feedback", str(tmp_path / "feedback-2.md")])
    assert (handback(tmp_path)["status"], handback(tmp_path)["artifacts"]) == ("ok", ["comparison"])
    m = manifest(tmp_path / "comparison")
    assert rsi_validate("comparison", tmp_path / "comparison", task) == []
    assert (m["candidate_id"], m["baselines"]) == ("candidate-0001", {"lr": "baseline-0001", "gbdt": "baseline-0002",
                                                                      "mlp": "baseline-0003"})
    cand_preds = os.path.join(json.loads(open(brief).read())["inputs"]["candidate"], "predictions_val.jsonl")
    want = rsi_score_compare(brief, cand_preds, tmp_path / "bl-lr" / "predictions_val.jsonl")  # no "metric"
    assert m["deltas_val"]["lr"] == dict(want, metric=task["metrics"]["primary"])
    auc = dict((n, m["metrics_val"][n]["auc"]) for n in m["baselines"])
    assert m["strongest"] == max(sorted(auc), key=auc.get)          # a tie goes to the alphabetically first name
    report = (tmp_path / "comparison" / "report.md").read_text()
    assert "| %s (%s) |" % (m["strongest"], m["baselines"][m["strongest"]]) in report
