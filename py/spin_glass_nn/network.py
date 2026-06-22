"""Small MLP from scratch for Hessian spectrum analysis."""

from __future__ import annotations
import numpy as np


class MLP:
    """2-layer MLP with ReLU activations."""

    def __init__(self, n_in: int = 2, n_hidden: int = 20, n_out: int = 2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_out, n_hidden))
        self.b2 = np.zeros(n_out)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._h_pre = X @ self.W1.T + self.b1
        self._h = np.maximum(0, self._h_pre)
        return self._h @ self.W2.T + self.b2

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        logits = self.forward(X)
        mx = logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(logits - mx).sum(axis=1)) + mx[:, 0]
        return float(np.mean(lse - logits[np.arange(len(y)), y]))

    def gradient_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        N = len(y)
        logits = self.forward(X)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        dO = probs / N
        dO[np.arange(N), y] -= 1.0 / N

        self.W2 -= lr * (dO.T @ self._h)
        self.b2 -= lr * dO.sum(axis=0)
        dH = dO @ self.W2 * (self._h_pre > 0)
        self.W1 -= lr * (dH.T @ X)
        self.b1 -= lr * dH.sum(axis=0)

        return self.loss(X, y)

    def hessian_diagonal(self, X: np.ndarray, y: np.ndarray, eps: float = 5e-4) -> np.ndarray:
        """Estimate diagonal of Hessian via finite differences."""
        params = self._get_params()
        n = len(params)
        base = self.loss(X, y)
        diag = np.zeros(n)
        for k in range(n):
            p1, p2 = params.copy(), params.copy()
            p1[k] += eps; p2[k] -= eps
            self._set_params(p1); l1 = self.loss(X, y)
            self._set_params(p2); l2 = self.loss(X, y)
            diag[k] = (l1 - 2 * base + l2) / eps**2
        self._set_params(params)
        return diag

    def _get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def _set_params(self, p: np.ndarray) -> None:
        nh, ni = self.W1.shape; no = self.W2.shape[0]
        i = 0
        self.W1 = p[i: i + nh*ni].reshape(nh, ni); i += nh * ni
        self.b1 = p[i: i + nh]; i += nh
        self.W2 = p[i: i + no*nh].reshape(no, nh); i += no * nh
        self.b2 = p[i:]

    def copy(self) -> "MLP":
        import copy
        return copy.deepcopy(self)
