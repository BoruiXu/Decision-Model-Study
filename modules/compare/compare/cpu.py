"""The CPU baselines: lr, gbdt and mlp, fitted on a feature table (features.table.Featurizer) inline.

Each searches a small fixed grid (COMPARE_GRIDS; the task's compare.grids.<name> replaces one) and keeps the grid
point with the best primary metric on the valset, in the metric's direction; a tie goes to the first point in grid
order (the grid's keys in their written order, each key's values in order, the last key varying fastest). Everything
is seeded, so a refit gives the same predictions. scikit-learn, xgboost and lightgbm are imported when a baseline is
fitted, never at import.
"""
import collections
import importlib
import itertools
import math
import random
import warnings

from rsi import metrics as M
from rsi.rules import adapter_path, verbatim_paths

from features.table import Featurizer, real_rows

from . import preds

CPU_BASELINES = ("lr", "gbdt", "mlp")
COMPARE_GRIDS = {
    "lr": {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
    "gbdt": {"n_estimators": [200, 500], "max_depth": [4, 6]},
    "mlp": {"hidden_layer_sizes": [[64], [128, 64]], "alpha": [1e-4, 1e-3]},
}
# The parameters every grid point shares ("<seed>" is replaced by compare.seed); a grid key of the same name wins.
FIXED = {
    "lr": {"max_iter": 2000},
    "gbdt": {"learning_rate": 0.05, "subsample": 0.8, "random_state": "<seed>"},
    "mlp": {"early_stopping": True, "max_iter": 300, "random_state": "<seed>"},
}
MODELS = {
    "lr": "sklearn LogisticRegression after sign(x)*log1p(|x|) and StandardScaler",
    "gbdt": "xgboost XGBClassifier when xgboost imports, else lightgbm LGBMClassifier, on the raw features",
    "mlp": "sklearn MLPClassifier after sign(x)*log1p(|x|) and StandardScaler",
}


Table = collections.namedtuple("Table", "X_tr y X_val X_other signature")


def feature_table(task, train_rows, val_rows, other_rows, max_train_rows, seed):
    """The feature table of the CPU baselines: the real rows of `train_rows` (a stratified seeded subsample of
    max_train_rows when set) and their hard labels; every row of `val_rows` and of `other_rows` (the evalset inputs,
    or at the milestone the testset inputs). The featurizer (the task's adapter, else the generic model view) fixes its
    columns on those trainset rows. ValueError (features.FeatureError) when the features cannot be made, or a column
    determines a label."""
    labels = task["decision"]["labels"]
    train = real_rows(train_rows)
    y = [M.hard_label(r["target"], labels) for r in train]
    keep = subsample(y, max_train_rows, seed)
    train, y = [train[i] for i in keep], [y[i] for i in keep]
    fz = Featurizer(adapter_path(task), verbatim_paths(task))
    fz.fit(train, y)
    return Table(fz.transform(train)[1], y, fz.transform(val_rows)[1], fz.transform(other_rows)[1], fz.signature())


def grid_of(name, settings):
    """The grid of baseline `name`: the task's compare.grids.<name>, else COMPARE_GRIDS[name]."""
    return (settings.get("grids") or {}).get(name, COMPARE_GRIDS[name])


def spec(name, settings):
    """The resolved specification of CPU baseline `name` (what its spec_sha256 covers), with the library that fits it
    here and its version (library_of), so that a baseline is fitted again, never reused, after a library change."""
    seed = settings["seed"]
    fixed = dict((k, seed if v == "<seed>" else v) for k, v in FIXED[name].items())
    return {"model": MODELS[name], "library": library_of(name), "fixed": fixed, "grid": grid_of(name, settings),
            "seed": seed, "max_train_rows": settings["max_train_rows"],
            "rows": "real trainset rows (no prompt variants)", "select": "the primary metric on the valset"}


def library_of(name):
    """{"name", "version"} of the library that fits baseline `name` here: scikit-learn for lr and mlp; for gbdt
    xgboost when it imports, else lightgbm (both None when neither does, or the version None when it cannot be
    read)."""
    if name == "gbdt":
        try:
            lib = gbdt_library()
        except ValueError:
            return {"name": None, "version": None}
    else:
        lib = "scikit-learn"
    try:
        version = importlib.import_module({"scikit-learn": "sklearn"}.get(lib, lib)).__version__
    except Exception:  # noqa: BLE001  (a library that does not load has no version to record)
        version = None
    return {"name": lib, "version": version}


def grid_points(grid):
    """The grid's points in grid order: dicts key -> value."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[list(grid[k]) for k in keys])]


def subsample(y, n, seed):
    """Indices of a stratified seeded subsample of size `n` of the labels `y` (all of them when n is None or not
    smaller), in their original order. Each label gets its share, rounded by largest remainder."""
    if n is None or n >= len(y):
        return list(range(len(y)))
    by = {}
    for i, label in enumerate(y):
        by.setdefault(label, []).append(i)
    exact = dict((label, n * len(idx) / float(len(y))) for label, idx in by.items())
    quota = dict((label, int(math.floor(x))) for label, x in exact.items())
    rest = sorted(by, key=lambda label: (-(exact[label] - quota[label]), label))
    for label in rest[:n - sum(quota.values())]:
        quota[label] += 1
    rng = random.Random(seed)
    chosen = []
    for label in sorted(by):
        chosen += rng.sample(by[label], quota[label])
    return sorted(chosen)


def signed_log(x):
    import numpy as np
    return np.sign(x) * np.log1p(np.abs(x))


def gbdt_library():
    """"xgboost" when it imports, else "lightgbm"; ValueError when neither does. Any error of the import counts as
    not importing: xgboost raises its own XGBoostError when its native library does not load (e.g. no libomp)."""
    errors = []
    for lib in ("xgboost", "lightgbm"):
        try:
            importlib.import_module(lib)
            return lib
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s: %s" % (lib, type(e).__name__, (str(e).splitlines() or [""])[0]))
    raise ValueError("gbdt needs xgboost or lightgbm, and neither imports (%s; modules/common/install_deps.sh "
                     "installs xgboost)" % "; ".join(errors))


def _params(name, point, seed):
    p = dict((k, seed if v == "<seed>" else v) for k, v in FIXED[name].items())
    p.update(point)
    if name == "mlp" and isinstance(p.get("hidden_layer_sizes"), list):
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return p


def make_model(name, point, seed, library=None):
    """An unfitted model of baseline `name` at grid point `point`."""
    if name == "gbdt":
        params = _params(name, point, seed)
        if library == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(tree_method="hist", n_jobs=1, verbosity=0, **params)
        from lightgbm import LGBMClassifier
        params.setdefault("subsample_freq", 1)   # without it lightgbm ignores subsample
        return LGBMClassifier(deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1, **params)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**_params(name, point, seed))
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(**_params(name, point, seed))
    else:
        raise ValueError("unknown CPU baseline %r (one of %s)" % (name, ", ".join(CPU_BASELINES)))
    return Pipeline([("log", FunctionTransformer(signed_log)), ("scale", StandardScaler()), ("model", model)])


def _probs(model, X, seen, labels):
    """label -> probability rows: the model's columns (encoded seen labels) mapped back, unseen labels 0."""
    import numpy as np
    out = []
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    for row in raw:
        p = dict((label, 0.0) for label in labels)
        total = float(row.sum())
        for j, code in enumerate(model.classes_):
            p[seen[int(code)]] = float(row[j]) / total
        out.append(p)
    return out


def fit_cpu(name, X_tr, y_tr, X_val, X_eval, grid, seed, task, val_rows):
    """Fit CPU baseline `name` at every point of `grid` on (X_tr, y_tr: hard labels), score each point's valset
    predictions (X_val: one row per row of val_rows) by the primary metric on the real valset rows, and keep the best
    (a tie: the first). Returns (P_val, P_eval, config): label -> probability rows for X_val and X_eval, and
    {"grid_point", "val_scores" (one per point, in grid order), "train_rows"}, plus "library" for gbdt. ValueError
    when the trainset rows have fewer than two labels."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    labels = task["decision"]["labels"]
    seen = [label for label in labels if label in set(y_tr)]
    if len(seen) < 2:
        raise ValueError("the trainset rows have one label only (%s): nothing to separate" % ", ".join(seen))
    code = dict((label, i) for i, label in enumerate(seen))
    X_tr, y = np.asarray(X_tr, dtype=np.float64), np.asarray([code[label] for label in y_tr])
    X_val, X_eval = np.asarray(X_val, dtype=np.float64), np.asarray(X_eval, dtype=np.float64)
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    ids = [r["id"] for r in val_rows]
    real = set(r["id"] for r in real_rows(val_rows))
    sel_rows = [r for r in val_rows if r["id"] in real]
    library = gbdt_library() if name == "gbdt" else None
    best, scores = None, []
    for point in grid_points(grid):
        model = make_model(name, point, seed, library)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_tr, y)
        P_val = _probs(model, X_val, seen, labels)
        scored = [r for r in preds.rows(ids, P_val, labels) if r["id"] in real]
        s = M.score(primary, scored, sel_rows, task)
        scores.append(s)
        if best is None or (s > best[0] if higher else s < best[0]):
            best = (s, point, model, P_val)
    _, point, model, P_val = best
    config = {"grid_point": point, "val_scores": scores, "train_rows": int(len(y))}
    if library is not None:
        config["library"] = library
    return P_val, _probs(model, X_eval, seen, labels), config
