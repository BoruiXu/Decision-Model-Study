import pytest

from harness.evaluator.metrics import (at_threshold, auc, aggregate, budget_threshold, ece, fit_threshold,
                                       max_share_at_budget, within_group_auc)
from harness.evaluator.text import text_of


def test_auc_soft_rates():
    # weights: (a,b) = rate_a * (1 - rate_b); scores 0.9, 0.5, 0.1 with rates 1, 0.5, 0
    value = auc([0.9, 0.5, 0.1], [1.0, 0.5, 0.0])
    # pairs: (0,1) w=.5 win; (0,2) w=1 win; (1,2) w=.5 win; (1,0) w=0; (2,*) w=0; (1,1) w=.25 tie
    wins = .5 + 1 + .5 + .25 * .5
    total = .5 + 1 + .5 + .25
    assert value == pytest.approx(wins / total)
    assert auc([0.1, 0.2], [1.0, 1.0]) is None


def test_within_group_prior_is_half():
    groups = ["a", "a", "b", "b"]
    assert within_group_auc(groups, [.3, .3, .8, .8], [1., 0., 1., 0.]) == pytest.approx(0.5)
    assert within_group_auc(["a", "a"], [.3, .3], [1., 1.]) is None


def test_fit_threshold_tie_rule():
    # always-small and threshold .5 give the same Jaccard: .5 is nearest to .5
    best = fit_threshold([.2, .9], [1., 1.])
    assert best["jaccard"] == 1.0
    # candidates .2 (all routed), .5 (one routed: j=.5), .9, 1.1 -> .2 wins on jaccard alone
    assert best["threshold"] == .2
    best = fit_threshold([.4, .6], [0., 0.])
    # all-zero agreements: routing nothing gives union 0 -> jaccard 1; only t=1.1 does
    assert best["threshold"] == 1.1
    # equal jaccard: .5 vs .7 both route nothing relevant -> pick nearest .5
    best = fit_threshold([.1, .3], [0., 0.])
    assert best["jaccard"] == 1.0 and best["threshold"] == .5


def test_aggregate_keys():
    out = aggregate([.9, .1], [1., 0.], .5)
    assert out["jaccard"] == 1.0 and out["small_selected"] == 1 and out["tp_pairs"] == 16
    assert set(out) >= {"preservation", "small_usage", "questions", "fp_pairs", "fn_pairs", "threshold"}


def test_max_share_ties_interpolated():
    # one tied group of 2 (incons 0 and 1), budget .25: take 1 item fully-free? mean .5 > .25
    # prefix empty, group gives (0+1)/2 = .5 > .25 and n_r == 0 -> best stays 0
    assert max_share_at_budget([.5, .5], [0., 1.], .25) == 0.0
    # high-score clean item then a tied pair straddling the budget
    share = max_share_at_budget([.9, .5, .5], [0., 1., 0.], .25)
    # n_r=1,s_r=0; group sg=1, ng=2: (1)/(3) > .25; t=(.25*1-0)/(1-.25*2)=.5 -> best=1+.5*2=2
    assert share == pytest.approx(2 / 3)


def test_budget_threshold_none_feasible():
    out = budget_threshold([.2, .8], [1., 1.], .1)
    assert out == dict(threshold=1.1, small_share=0.0, inconsistency=None)


def test_budget_threshold_and_at_threshold():
    out = budget_threshold([.9, .6, .2], [0., 0., 1.], .1)
    assert out["threshold"] == .6 and out["small_share"] == pytest.approx(2 / 3)
    res = at_threshold([.9, .6, .2], [1., 0., 1.], [0., 0., 1.], .6)
    assert res["jaccard"] == pytest.approx(1 / 3) and res["small_share"] == pytest.approx(2 / 3)
    assert at_threshold([.1], [1.], [0.], .5)["inconsistency"] is None


def test_ece():
    assert ece([.5, .5], [.5, .5]) == pytest.approx(0.0)
    assert ece([.95, .95], [0., 0.]) == pytest.approx(.95)
    assert ece([.1, .9], [1., 0.]) == pytest.approx(.9)


def test_text_of_messages():
    row = {"state": {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}}
    assert text_of(row, "messages") == "user: hi\nassistant: yo"
    assert text_of({"messages": [{"role": "user", "content": "x"}]}, "messages") == "user: x"
    assert text_of({"state": {"b": 1, "a": "é"}}, "state_json") == '{"a": "é", "b": 1}'
    with pytest.raises(ValueError):
        text_of(row, "other")
