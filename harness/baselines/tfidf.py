"""TF-IDF + logistic baseline ported from v1 `cheap_baseline.py` (soft targets, CV-chosen strength).

Examples are dicts `{id, text, rate, group}`; v1's `category` is re-keyed to `group`.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from harness.evaluator.metrics import auc

STRENGTHS = (1., 10., 100.)
SEED = 0


# Provenance: data_pipeline_v2/sanity_check/step1_data/cheap_baseline.py `Features`.
class Features:
    """Fitted on training text only, then applied to held-out text."""

    def __init__(self, examples):
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                     max_features=200_000)
        texts = [e["text"] for e in examples]
        self.train = self._stack([self.words.fit_transform(texts), self.chars.fit_transform(texts)], texts)

    @staticmethod
    def _stack(parts, texts):
        return hstack([*parts, csr_matrix([[math.log1p(len(text)) / 10] for text in texts])]).tocsr()

    def of(self, examples):
        texts = [e["text"] for e in examples]
        return self._stack([self.words.transform(texts), self.chars.transform(texts)], texts)


# Provenance: cheap_baseline.py `logistic`.
def logistic(x, examples, strength):
    rates = np.array([e["rate"] for e in examples])
    # Soft targets: each question counts as a positive with weight y and a negative with 1 - y.
    return LogisticRegression(C=strength, max_iter=2000).fit(
        vstack([x, x]).tocsr(), np.repeat([1, 0], len(examples)), sample_weight=np.concatenate([rates, 1 - rates]) + 1e-9)


# Provenance: cheap_baseline.py `choose_strength`.
def choose_strength(examples):
    """Regularisation picked by cross-validation inside the given questions, which are never scored."""
    if len(examples) < 60:
        return STRENGTHS[1]
    held_out_scores = {strength: [None] * len(examples) for strength in STRENGTHS}
    for train, held_out in KFold(3, shuffle=True, random_state=SEED).split(examples):
        inner = [examples[i] for i in train]
        features = Features(inner)
        x = features.of([examples[i] for i in held_out])
        for strength in STRENGTHS:
            for i, p in zip(held_out, logistic(features.train, inner, strength).predict_proba(x)[:, 1]):
                held_out_scores[strength][i] = p
    rates = [e["rate"] for e in examples]
    return max(STRENGTHS, key=lambda strength: auc(held_out_scores[strength], rates) or 0.)


# Provenance: cheap_baseline.py `TextBaseline`.
class TextBaseline:
    def __init__(self, examples):
        self.strength = choose_strength(examples)
        self.features = Features(examples)
        self.model = logistic(self.features.train, examples, self.strength)

    def predict(self, examples):
        return self.model.predict_proba(self.features.of(examples))[:, 1].tolist()


# Provenance: cheap_baseline.py `category_prior` (train/examples take `group` instead of `category`).
def group_prior(train, examples):
    """Predict each question by the mean agreement rate of its group in the training set."""
    groups = {}
    for example in train:
        groups.setdefault(example["group"], []).append(example["rate"])
    overall = sum(e["rate"] for e in train) / len(train)
    return [sum(groups[e["group"]]) / len(groups[e["group"]]) if e["group"] in groups else overall
            for e in examples]
