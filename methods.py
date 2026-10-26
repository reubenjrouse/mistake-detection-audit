"""Mistake detectors. Each has fit(rows) on correct training steps and score(rows) -> array.

Higher score = more likely a mistake. `rows` is a dict of arrays from the steps table.
"""
import numpy as np


def cosine_distance(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - (a * b).sum(1)


class Identity:
    """Expected outcome = the state before the step ("nothing should change")."""

    def fit(self, rows):
        return self

    def score(self, rows):
        return cosine_distance(rows["s_after"], rows["s_before"])


class Prototype:
    """Expected outcome = average correct s_after for this step type (state-agnostic)."""

    def fit(self, rows):
        self.protos = {sid: rows["s_after"][rows["step_id"] == sid].mean(0)
                       for sid in np.unique(rows["step_id"])}
        return self

    def score(self, rows):
        unseen = set(rows["step_id"]) - set(self.protos)
        if unseen:
            raise ValueError(f"no training prototype for step ids {sorted(unseen)}")
        expected = np.stack([self.protos[sid] for sid in rows["step_id"]])
        return cosine_distance(rows["s_after"], expected)


METHODS = {"identity": Identity, "prototype": Prototype}
