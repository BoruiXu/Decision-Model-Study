"""Length-only baseline: logistic regression on log(1 + character length) of the baseline text.

Soft targets as in the TF-IDF baseline: each question counts as a positive with weight y and a negative
with weight 1 - y (each plus 1e-9).
"""
import math

import numpy as np
from sklearn.linear_model import LogisticRegression


def _x(examples):
    return np.array([[math.log1p(len(e["text"]))] for e in examples])


class LengthBaseline:
    def __init__(self, examples):
        x = _x(examples)
        rates = np.array([e["rate"] for e in examples], dtype=float)
        self.model = LogisticRegression(max_iter=2000).fit(
            np.vstack([x, x]), np.repeat([1, 0], len(examples)),
            sample_weight=np.concatenate([rates, 1 - rates]) + 1e-9)

    def predict(self, examples):
        return self.model.predict_proba(_x(examples))[:, 1].tolist()
