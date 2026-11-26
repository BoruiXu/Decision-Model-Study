"""Weakness types: the errors of each kind grouped by KMeans in the z-space, each described against the correctly
decided rows of the same true label (no LLM needed).

Per kind: KMeans (n_init 10, random_state seed) for k in 2..min(6, n // 10), k chosen by the silhouette on at most
3000 rows; fewer than 20 errors, or no k with a silhouette of at least 0.1, gives k = 1. Each type has its size, its
share of the kind, its centroid and p90 radius (z units), the mean probability of the candidate's decision, its member
ids and its signature: the top 5 features by |median z(type) - median z(correct rows of the same true label)|, with
both medians in the model view's own units.
"""
import numpy as np

KMAX, MIN_SIL, MIN_ERRORS, SIL_ROWS, TOP = 6, 0.1, 20, 3000, 5


def choose_k(Z, seed):
    """(k, silhouette or None, labels)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    one = (1, None, np.zeros(len(Z), int))
    if len(Z) < MIN_ERRORS:
        return one
    rng = np.random.default_rng(seed)
    best = one
    for k in range(2, min(KMAX, len(Z) // 10) + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        ix = rng.choice(len(Z), min(SIL_ROWS, len(Z)), replace=False)
        if len(set(lab[ix])) < 2:
            continue
        s = float(silhouette_score(Z[ix], lab[ix]))
        if s >= MIN_SIL and (best[1] is None or s > best[1]):
            best = (k, s, lab)
    return best


def type_key(kind, g):
    return "%s#%d" % (kind, g)


def build(space, errors, ok_rows, label_of, seed=0):
    """weakness.json: {"columns", "scaler", "types"}. `errors`: [(row, diag)] with diag {"kind", "true", "decision",
    "p_decision"}; `ok_rows`: correctly decided rows (the ok sample); label_of(row) is its hard label."""
    names = list(space.names)
    out = {"columns": names, "scaler": space.to_json(), "types": []}
    kinds = []
    for _, d in errors:
        if d["kind"] not in kinds:
            kinds.append(d["kind"])
    for kind in sorted(kinds):
        F = [(r, d) for r, d in errors if d["kind"] == kind]
        true = F[0][1]["true"]
        R = [r for r in ok_rows if label_of(r) == true]
        Xf = space.raw([r for r, _ in F])
        Z = space.z(Xf)
        k, sil, labels = choose_k(Z, seed)
        Xr = space.raw(R) if R else None
        Zr = space.z(Xr) if R else None
        groups = np.unique(labels)  # the clusters KMeans filled (an empty one gives no type)
        for g, cluster in enumerate(groups):
            m = labels == cluster
            cent = Z[m].mean(0)
            sig = []
            if Zr is not None:
                diff = np.median(Z[m], 0) - np.median(Zr, 0)
                for j in np.argsort(-np.abs(diff), kind="stable")[:TOP]:
                    sig.append({"feature": names[j], "diff_sd": round(float(diff[j]), 3),
                                "median_type": float(np.median(Xf[m, j])),
                                "median_correct": float(np.median(Xr[:, j]))})
            members = [F[i] for i in np.where(m)[0]]
            out["types"].append({
                "key": type_key(kind, g), "kind": kind, "true": true, "decision": F[0][1]["decision"],
                "size": int(m.sum()), "share_of_kind": round(float(m.mean()), 4), "k": len(groups), "silhouette": sil,
                "centroid": [float(x) for x in cent],
                "radius_p90": float(np.quantile(np.linalg.norm(Z[m] - cent, axis=1), 0.9)),
                "mean_p_decision": float(np.mean([d["p_decision"] for _, d in members])),
                "signature": sig, "member_ids": [r["id"] for r, _ in members]})
    return out
