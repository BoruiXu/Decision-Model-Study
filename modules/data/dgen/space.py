"""The z-space of dgen: each row's model view (features.table.Featurizer with the task's adapter), sign(x)·log1p(|x|),
z-scored on the real rows of the trainset and the pool. Weakness types, signatures, SMOTE neighbours, the distance
check and CLLM all work in it; dedup compares the model view itself (raw).
"""
import numpy as np

from features.table import Featurizer
from rsi.rules import adapter_path, verbatim_paths

from . import DgenError


def featurizer(task):
    return Featurizer(adapter_path(task), verbatim_paths(task))


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


class Space(object):
    """fit(rows, labels) fixes the columns (the Featurizer's label-leak check included) and the scaler, and keeps the
    fit rows' model-view matrix as X (in their order); raw(rows) is the model-view matrix, z(rows or raw matrix) the
    z-scored one."""

    def __init__(self, task):
        self.fz = featurizer(task)
        self.names, self.mu, self.sd, self.X = None, None, None, None

    def fit(self, rows, labels=None):
        if not rows:
            raise DgenError("no rows to fit the feature space on (the trainset and the pool are empty)")
        try:
            self.names = self.fz.fit(rows, labels)
        except ValueError as e:  # features.FeatureError
            raise DgenError("features: %s" % e)
        if not self.names:
            raise DgenError("features: the rows' model view has no numeric column, so dgen has no space to work in "
                            "(give the task a features adapter whose features() returns numbers)")
        self.X = self.raw(rows)
        T = signed_log(self.X)
        self.mu = T.mean(0)
        sd = T.std(0)
        sd[sd == 0] = 1.0
        self.sd = sd
        return self

    def raw(self, rows):
        if not rows:
            return np.zeros((0, len(self.names)))
        try:
            return np.array(self.fz.transform(rows)[1], dtype=float)
        except ValueError as e:
            raise DgenError("features: %s" % e)

    def z(self, rows_or_raw):
        X = rows_or_raw if isinstance(rows_or_raw, np.ndarray) else self.raw(rows_or_raw)
        return (signed_log(X) - self.mu) / self.sd

    def to_json(self):
        return {"columns": list(self.names), "transform": "sign(x)*log1p(|x|), then (t - mu) / sd",
                "mu": [float(x) for x in self.mu], "sd": [float(x) for x in self.sd],
                "features": self.fz.signature()}
