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


class Contrastive(Method):
    """Scores whether s_after is a believable outcome of this step from s_before.

    Trained on correct steps only, against fake mistakes made from them:
      "other_step": the after-state of a different step in the same recording
                    (wrong outcome, but same kitchen, person and lighting);
      "half_done":  the state halfway through the same step.
    Score = -logit(real), so higher = more likely a mistake.
    """

    def __init__(self, seed=0, hidden=512, epochs=10, lr=1e-3, weight_decay=1e-2,
                 negatives=("other_step", "half_done")):
        super().__init__(seed)
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.weight_decay, self.negatives = weight_decay, negatives

    def _inputs(self, before, text, after):
        b, a = (before - self.mu) / self.sd, (after - self.mu) / self.sd
        return torch.tensor(np.hstack([b, text, a, a - b]), dtype=torch.float32)

    def _other_step(self, rows, rng):
        """For each row, a random different step from the same recording."""
        other = np.arange(len(rows["step_id"]))  # rows alone in their recording keep themselves
        for idx in self.same_recording:
            other[idx] = idx[(np.arange(len(idx)) + rng.integers(1, len(idx), len(idx))) % len(idx)]
        return other

    def fit(self, rows):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        both = np.concatenate([rows["s_before"], rows["s_after"]])
        self.mu, self.sd = both.mean(0), both.std(0) + 1e-6
        # Recordings with a single correct step cannot provide an "other_step" fake.
        groups = [np.flatnonzero(rows["recording_id"] == r) for r in np.unique(rows["recording_id"])]
        self.same_recording = [g for g in groups if len(g) > 1]
        multi = np.concatenate(self.same_recording)

        before, text, after = rows["s_before"], rows["text_emb"], rows["s_after"]
        dim = self._inputs(before[:1], text[:1], after[:1]).shape[1]
        self.net = nn.Sequential(nn.Linear(dim, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            # Fresh fakes each epoch; one real example per fake keeps the classes balanced.
            parts = []
            if "other_step" in self.negatives:
                other = self._other_step(rows, rng)
                parts.append((self._inputs(before[multi], text[multi], after[other[multi]]), 0.0))
                parts.append((self._inputs(before[multi], text[multi], after[multi]), 1.0))
            if "half_done" in self.negatives:
                parts.append((self._inputs(before, text, rows["s_mid"]), 0.0))
                parts.append((self._inputs(before, text, after), 1.0))
            x = torch.cat([p for p, _ in parts])
            y = torch.cat([torch.full((len(p),), label) for p, label in parts])
            for idx in torch.randperm(len(y)).split(128):
                loss = nn.functional.binary_cross_entropy_with_logits(self.net(x[idx]).squeeze(1), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, rows):
        return -self.net(self._inputs(rows["s_before"], rows["text_emb"], rows["s_after"])).squeeze(1).numpy()


METHODS = {"identity": Identity, "prototype": Prototype, "state_nn": StateNN,
           "world_model": WorldModel, "contrastive": Contrastive}
