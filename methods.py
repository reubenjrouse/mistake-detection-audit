"""Mistake detectors. Each predicts the expected outcome of a step; score = distance to what happened.

fit(rows) sees correct training steps only. expected(rows) -> (n, dim) predicted s_after.
Higher score = more likely a mistake. `rows` is a dict of arrays from the steps table.
"""
import numpy as np
import torch
from torch import nn


def cosine_distance(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - (a * b).sum(1)


class Method:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, rows):
        return self

    def score(self, rows):
        return cosine_distance(rows["s_after"], self.expected(rows))

    def _check_seen(self, step_ids, seen):
        unseen = set(step_ids) - set(seen)
        if unseen:
            raise ValueError(f"no training data for step ids {sorted(unseen)}")


class Identity(Method):
    """Expected outcome = the state before the step ("nothing should change")."""

    def expected(self, rows):
        return rows["s_before"]


class Prototype(Method):
    """Expected outcome = average correct s_after for this step type (state-agnostic)."""

    def fit(self, rows):
        self.protos = {sid: rows["s_after"][rows["step_id"] == sid].mean(0)
                       for sid in np.unique(rows["step_id"])}
        return self

    def expected(self, rows):
        self._check_seen(rows["step_id"], self.protos)
        return np.stack([self.protos[sid] for sid in rows["step_id"]])


class StateNN(Method):
    """Expected outcome = s_after of the same-step training example with the most similar s_before.

    Conditions on the current state without learning anything.
    """

    def fit(self, rows):
        unit = rows["s_before"] / np.linalg.norm(rows["s_before"], axis=1, keepdims=True)
        self.bank = {sid: (unit[rows["step_id"] == sid], rows["s_after"][rows["step_id"] == sid])
                     for sid in np.unique(rows["step_id"])}
        return self

    def expected(self, rows):
        self._check_seen(rows["step_id"], self.bank)
        unit = rows["s_before"] / np.linalg.norm(rows["s_before"], axis=1, keepdims=True)
        out = np.empty_like(rows["s_after"])
        for i, sid in enumerate(rows["step_id"]):
            befores, afters = self.bank[sid]
            out[i] = afters[np.argmax(befores @ unit[i])]
        return out


class WorldModel(Method):
    """Expected outcome = MLP(s_before, step text), optionally residual. One network shared by all steps.

    Defaults chosen by prediction error on correct val steps only (no error labels):
    direct prediction and short training overfit least.
    """

    def __init__(self, seed=0, hidden=512, epochs=10, lr=1e-3, weight_decay=1e-2, residual=False):
        super().__init__(seed)
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.weight_decay, self.residual = weight_decay, residual

    def _tensors(self, rows):
        before = torch.tensor((rows["s_before"] - self.mu) / self.sd, dtype=torch.float32)
        text = torch.tensor(rows["text_emb"], dtype=torch.float32)
        return before, torch.cat([before, text], 1)

    def _predict(self, before, inputs):
        delta = self.net(inputs)
        return before + delta if self.residual else delta

    def fit(self, rows):
        torch.manual_seed(self.seed)
        both = np.concatenate([rows["s_before"], rows["s_after"]])
        self.mu, self.sd = both.mean(0), both.std(0) + 1e-6
        before, inputs = self._tensors(rows)
        target = torch.tensor((rows["s_after"] - self.mu) / self.sd, dtype=torch.float32)

        dim = before.shape[1]
        self.net = nn.Sequential(nn.Linear(inputs.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, dim))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(target)).split(128):
                loss = ((self._predict(before[idx], inputs[idx]) - target[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def expected(self, rows):
        return self._predict(*self._tensors(rows)).numpy() * self.sd + self.mu


METHODS = {"identity": Identity, "prototype": Prototype, "state_nn": StateNN, "world_model": WorldModel}
