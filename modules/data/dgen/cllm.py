"""CLLM label check (Seedat et al., ICML 2024), as dgen 0.4 runs it.

A boosted-tree reference is cross-fitted on real rows only (the trainset and the pool, without prompt variants or rows
generated in an earlier round: dgen.inputs), in K = 5 folds by int(md5(id), 16) % 5: XGBClassifier(n_estimators=100,
random_state=fold) when xgboost imports, else GradientBoostingClassifier(n_estimators=100, max_depth=3). A row is
scored by the model of fold(seed_id or id), the one that did not see its seed. Over the staged predictions of 99
iterations, confidence is the mean P(given label) and aleatoric the mean g(1 - g). Groups at threshold th (0.2): easy
(conf >= 1 - th and aleatoric <= th), hard (conf <= th and aleatoric <= th: the label conflicts with similar real
rows), ambiguous (the rest).

The candidate being repaired is never the reference: it would call exactly its own failures hard.
"""
import hashlib

import numpy as np

K, TH, STAGES = 5, 0.2, 99


def fold_of(row_id, k=K):
    return int(hashlib.md5(str(row_id).encode("utf-8")).hexdigest(), 16) % k


def library():
    """The reference's library, as recorded in the outputs."""
    try:
        import xgboost
        return "xgboost %s XGBClassifier(n_estimators=100, random_state=fold)" % xgboost.__version__
    except Exception:  # noqa: BLE001  (xgboost absent or not loading: scikit-learn's)
        import sklearn
        return "scikit-learn %s GradientBoostingClassifier(n_estimators=100, max_depth=3)" % sklearn.__version__


def _booster(fold):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, random_state=fold)
    except Exception:  # noqa: BLE001
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=fold)


def _staged(model, X):
    """(stages, rows, classes) probabilities of the first STAGES iterations."""
    if hasattr(model, "get_booster"):
        return np.stack([model.predict_proba(X, iteration_range=(0, i)) for i in range(1, STAGES + 1)])
    return np.stack(list(model.staged_predict_proba(X))[:STAGES])


class Reference(object):
    """The five fold models on (Z, y): y are label names; `classes` the labels the rows hold, in `labels` order. A
    fold whose training part holds one label only predicts it with probability 1."""

    def __init__(self, Z, y, ids, labels, k=K):
        y = np.asarray(y, dtype=object)
        self.k = k
        self.classes = [label for label in labels if (y == label).any()]
        folds = np.array([fold_of(i, k) for i in ids])
        self.models = []
        for f in range(k):
            part = folds != f
            present = [c for c in self.classes if (y[part] == c).any()]
            if len(present) < 2:
                self.models.append((None, present))
                continue
            m = _booster(f)
            m.fit(Z[part], np.array([present.index(v) for v in y[part]]))
            self.models.append((m, present))

    def score(self, Z, y, folds):
        """(conf, aleatoric) of each row: label y[i], scored by the model of folds[i]."""
        conf, al = np.zeros(len(y)), np.zeros(len(y))
        for f in range(self.k):
            idx = np.where(np.asarray(folds) == f)[0]
            if not len(idx):
                continue
            m, present = self.models[f]
            if m is None:
                P = np.ones((STAGES, len(idx), max(1, len(present))))
            else:
                P = _staged(m, Z[idx])
            for j, i in enumerate(idx):
                g = P[:, j, present.index(y[i])] if y[i] in present else np.zeros(STAGES)
                conf[i], al[i] = g.mean(), (g * (1 - g)).mean()
        return conf, al


def groups(conf, al, th=TH):
    g = np.full(len(conf), "ambiguous", dtype=object)
    g[(conf >= 1 - th) & (al <= th)] = "easy"
    g[(conf <= th) & (al <= th)] = "hard"
    return g


def shares(g, mask):
    """{"n", "easy", "ambiguous", "hard"}: the rows of `mask` and the share of each group (None without rows)."""
    n = int(mask.sum())
    return dict([("n", n)] + [(k, float(np.mean(g[mask] == k)) if n else None)
                              for k in ("easy", "ambiguous", "hard")])
