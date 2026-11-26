"""The quality checks of an arm's rows and the filter (dgen 0.4's, without its own test or A/B: whether the rows help
is judged by train, compare and judge). Every check runs; `filter` chooses which of them remove rows:

  rules             adapter.check of the row's state                                                   -> removable
  dedup             on the model-view vector rounded to 8 decimals: dup_internal (an earlier row of the arm),
                    leak_val (a valset input row: the data module has nothing of the evalset or the testset, which
                    the framework checks at trainset registration), copy_of_real (a trainset or pool row, the
                    trainset's generated rows included; `hard` rows are real by design and exempt from its
                    removal)                                                                          -> removable
  distance          the distance to the nearest real row of the same label over the typical real nearest-neighbour
                    distance (nn_ratio); `far` above the real p99                                     -> removable
  discriminability  5-fold HistGradientBoosting AUC telling the arm's rows from real rows of each label (0.5:
                    indistinguishable)                                                                -> report only
  cllm              the CLLM groups (dgen.cllm) of each row, scored by the fold that held out its seed  -> removable

The real rows (dgen.inputs: neither prompt variants nor rows generated in an earlier round) of the trainset and the
pool are the distance, discriminability and CLLM references; nothing is fitted on valset rows.
"""
import hashlib

import numpy as np

from . import DGEN_VERSION
from . import cllm
from . import rows as RW

CHECKS = ("rules", "dedup", "distance", "discriminability", "cllm")
FILTERABLE = ("rules", "dedup", "distance", "cllm")
DEFAULT_FILTER = ("rules", "dedup", "cllm")
SAMPLE = 50


def vector_hash(v):
    return hashlib.md5(np.round(np.asarray(v, dtype=float), 8).tobytes()).hexdigest()


def discriminability_auc(Zr, Zs, seed=0):
    """{"auc", "n"}: the 5-fold AUC on n rows of each side (n = min(real, generated, 5000)); auc None under 20. With
    a few dozen rows a side the trees can hardly split (min_samples_leaf 20), so the AUC stays near 0.5: read it with
    its n."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    n = min(len(Zr), len(Zs), 5000)
    if n < 20:
        return {"auc": None, "n": int(n)}
    rng = np.random.default_rng(seed)
    X = np.vstack([Zr[rng.choice(len(Zr), n, replace=False)], Zs[rng.choice(len(Zs), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    return {"auc": float(cross_val_score(HistGradientBoostingClassifier(random_state=seed), X, y, cv=5,
                                         scoring="roc_auc").mean()), "n": int(n)}


def run_checks(inputs, space, adapter, rows, prov, filt=DEFAULT_FILTER, th=cllm.TH):
    """(kept rows, flags, summary) of the arm's `rows` (prov: id -> provenance line). `space` is fitted on
    inputs.real(), so its X holds those rows in that order."""
    labels = inputs.labels
    n = len(rows)
    ys = np.array([inputs.label(r) for r in rows], dtype=object)
    flags = []
    for r in rows:
        p = prov.get(r["id"], {})
        seed = p.get("seed_id") or ((r.get("meta") or {}).get("extra") or {}).get("seed_id")
        flags.append({"id": r["id"], "label": inputs.label(r), "seed_id": seed, "type": p.get("type"),
                      "real": bool(p.get("real"))})
    removed_by = [[] for _ in rows]
    summ = {"dgen_version": DGEN_VERSION, "n": n, "by_label": dict((l, int((ys == l).sum())) for l in labels),
            "checks": list(CHECKS), "filter": list(filt)}
    Xs = space.raw(rows)
    Zs = space.z(Xs)

    def mark(check, mask):
        if check in filt:
            for i in np.where(mask)[0]:
                removed_by[i].append(check)

    # rules
    v = [adapter.check(r["state"]) for r in rows]
    for f, x in zip(flags, v):
        f["rules"] = x
    m = np.array([bool(x) for x in v], dtype=bool)
    summ["rules_violating"] = int(m.sum())
    mark("rules", m)
    # dedup
    real = inputs.real()  # the rows the space was fitted on, in the same order: its X
    hs = [vector_hash(x) for x in Xs]
    real_h = set(vector_hash(x) for x in space.raw(inputs.known()))
    val_h = set(vector_hash(x) for x in space.raw(inputs.val_inputs))
    seen, dup = set(), np.zeros(n, dtype=bool)
    for i, h in enumerate(hs):
        dup[i] = h in seen
        seen.add(h)
    leak = np.array([h in val_h for h in hs], dtype=bool)
    same_real = np.array([h in real_h for h in hs], dtype=bool)
    is_real = np.array([f["real"] for f in flags], dtype=bool)
    for i, f in enumerate(flags):
        f.update(dup_internal=bool(dup[i]), leak_val=bool(leak[i]), copy_of_real=bool(same_real[i]))
    summ.update(dup_internal=int(dup.sum()), leak_val=int(leak.sum()), copy_of_real=int(same_real.sum()),
                synthetic_copy_of_real=int((same_real & ~is_real).sum()))
    mark("dedup", dup | leak | (same_real & ~is_real))
    # distance
    Zr = space.z(space.X)
    yr = np.array([inputs.label(r) for r in real], dtype=object)
    ratio, far = np.zeros(n), np.zeros(n, dtype=bool)
    summ["distance"] = {}
    if n:
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(0)
        for lab in labels:
            Rl, S = Zr[yr == lab], np.where(ys == lab)[0]
            if len(Rl) < 3 or not len(S):
                continue
            nn = NearestNeighbors(n_neighbors=2).fit(Rl)
            rr = nn.kneighbors(Rl[rng.choice(len(Rl), min(3000, len(Rl)), replace=False)])[0][:, 1]
            typ, p99 = float(np.median(rr)) + 1e-9, float(np.quantile(rr, 0.99))
            d = nn.kneighbors(Zs[S], n_neighbors=1)[0][:, 0]
            ratio[S], far[S] = d / typ, d > p99
            summ["distance"][lab] = {"median_nn_ratio": float(np.median(d / typ)), "far": int((d > p99).sum()),
                                     "real_nn_p99": p99}
    for i, f in enumerate(flags):
        f.update(nn_ratio=round(float(ratio[i]), 4), far=bool(far[i]))
    mark("distance", far)
    # discriminability (report only)
    summ["discriminability"] = dict((lab, discriminability_auc(Zr[yr == lab], Zs[ys == lab]))
                                    for lab in labels if (ys == lab).any())
    # cllm
    conf, al = np.zeros(n), np.zeros(n)
    g = np.full(n, "ambiguous", dtype=object)
    if n:
        ref = cllm.Reference(Zr, list(yr), [r["id"] for r in real], labels)
        conf, al = ref.score(Zs, list(ys), [cllm.fold_of(f["seed_id"] or f["id"]) for f in flags])
        g = cllm.groups(conf, al, th)
    for i, f in enumerate(flags):
        f.update(cllm_conf=round(float(conf[i]), 4), cllm_al=round(float(al[i]), 4), cllm_group=str(g[i]))
    summ["cllm"] = {"reference": cllm.library(), "threshold": th,
                    "by_label": dict((lab, cllm.shares(g, ys == lab)) for lab in labels)}
    mark("cllm", g == "hard")
    for f, rb in zip(flags, removed_by):
        f["removed"], f["removed_by"] = bool(rb), rb
    kept = [r for r, f in zip(rows, flags) if not f["removed"]]
    summ["removed"] = n - len(kept)
    summ["kept"] = len(kept)
    summ["kept_by_label"] = dict((l, sum(1 for f in flags if f["label"] == l and not f["removed"])) for l in labels)
    summ["removed_by"] = dict((c, sum(1 for rb in removed_by if c in rb)) for c in filt)
    by_type = {}
    for f in flags:
        d = by_type.setdefault(str(f["type"]), {"n": 0, "kept": 0})
        d["n"] += 1
        d["kept"] += int(not f["removed"])
    summ["by_type"] = by_type
    return kept, flags, summ


def sample(kept, flags, seeds, label_of, seed):
    """sample.jsonl: at most 50 kept rows, stratified by label, each with what changed from its seed."""
    by_id = dict((f["id"], f) for f in flags)
    out = []
    for r in RW.stratified_sample(kept, label_of, SAMPLE, seed):
        f = by_id[r["id"]]
        s = seeds.get(f["seed_id"])
        changed = {}
        if s is not None and s is not r:
            from features.flatten import edit_view
            a, b = edit_view(s["state"]), edit_view(r["state"])
            changed = dict((k, {"seed": a.get(k), "row": b.get(k)}) for k in sorted(set(a) | set(b))
                           if a.get(k) != b.get(k))
        out.append({"id": r["id"], "label": f["label"], "seed_id": f["seed_id"], "type": f["type"],
                    "changed": changed, "nn_ratio": f["nn_ratio"], "cllm_group": f["cllm_group"]})
    return out


def _fmt(x):
    return "-" if x is None else ("%.3g" % x if isinstance(x, float) else str(x))


def report(recipe, stats, summ):
    """quality_report.md."""
    L = ["# Quality report: %s (%s, arm %s)" % (recipe["name"], recipe["method"], recipe["sha12"]), "",
         "dgen %s. Generated %d of %d requested (short %s; dropped by rule %d; skipped for structure %s). Kept %d, "
         "removed %d (filter: %s)." % (DGEN_VERSION, stats.get("n", 0), stats.get("requested", 0),
                                       _fmt(stats.get("short")), sum(stats.get("dropped_by_rule", {}).values()),
                                       _fmt(stats.get("skipped_structure")), summ["kept"], summ["removed"],
                                       ", ".join(summ["filter"])), "",
         "| check | value |", "|---|---|",
         "| rules violating | %d |" % summ["rules_violating"],
         "| dup_internal | %d |" % summ["dup_internal"],
         "| leak_val (equal to a valset input) | %d |" % summ["leak_val"],
         "| copy_of_real (of them synthetic) | %d (%d) |" % (summ["copy_of_real"], summ["synthetic_copy_of_real"])]
    for lab, d in sorted(summ["distance"].items()):
        L.append("| distance %s: median nn ratio, far | %.3g, %d |" % (lab, d["median_nn_ratio"], d["far"]))
    for lab, d in sorted(summ["discriminability"].items()):
        L.append("| discriminability AUC %s (report only; n rows a side) | %s (n %d) |"
                 % (lab, _fmt(d["auc"]), d["n"]))
    for lab, d in sorted(summ["cllm"]["by_label"].items()):
        if d["n"]:
            L.append("| CLLM %s: easy, ambiguous, hard | %s, %s, %s |" % (lab, _fmt(d["easy"]), _fmt(d["ambiguous"]),
                                                                       _fmt(d["hard"])))
    L += ["", "Removed by: %s. Kept by label: %s." % (
        ", ".join("%s %d" % kv for kv in summ["removed_by"].items()) or "nothing",
        ", ".join("%s %d" % kv for kv in summ["kept_by_label"].items())),
        "", "CLLM reference: %s, threshold %s." % (summ["cllm"]["reference"], summ["cllm"]["threshold"])]
    return "\n".join(L) + "\n"
