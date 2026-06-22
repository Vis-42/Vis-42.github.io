"""Echo State Network implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Array = np.ndarray


@dataclass
class ESNConfig:
    N_reservoir: int = 100
    sparsity: float = 0.1
    spectral_radius: float = 0.9
    input_scaling: float = 0.5
    leak_rate: float = 1.0
    ridge_alpha: float = 1e-6
    seed: int = 42


class EchoStateNetwork:
    def __init__(self, cfg: ESNConfig, n_input: int = 3):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        N = cfg.N_reservoir
        self.W_in = (2*rng.random((N, n_input)) - 1) * cfg.input_scaling
        # Sample distinct entries without replacement so the realised connection
        # density equals `sparsity` exactly — the sweep axis of the phase diagram.
        # (Sampling with replacement loses ~5% of edges to collisions.)
        W_res = np.zeros(N * N)
        n_edges = round(N * N * cfg.sparsity)
        idx = rng.choice(N * N, size=n_edges, replace=False)
        W_res[idx] = rng.standard_normal(n_edges)
        W_res = W_res.reshape(N, N)
        eigvals = np.linalg.eigvals(W_res)
        rho = np.max(np.abs(eigvals))
        if rho > 1e-8:
            W_res *= cfg.spectral_radius / rho
        self.W_res = W_res
        self.W_out: Optional[Array] = None
        self.state = np.zeros(N)
        self.last_input = np.zeros(n_input)

    def _step(self, u: Array) -> None:
        lr = self.cfg.leak_rate
        new = np.tanh(self.W_res @ self.state + self.W_in @ u)
        self.state = (1 - lr) * self.state + lr * new

    def train(self, data: Array, washout: int) -> None:
        n, d = data.shape
        N = self.cfg.N_reservoir
        self.state[:] = 0
        coll = np.zeros((n - washout - 1, N + d))
        for t in range(n - 1):
            self._step(data[t])
            if t >= washout:
                coll[t - washout, :N] = self.state
                coll[t - washout, N:] = data[t]
        targets = data[washout + 1:]
        A = coll.T @ coll + self.cfg.ridge_alpha * np.eye(N + d)
        self.W_out = np.linalg.solve(A, coll.T @ targets).T
        # Advance one step with final training point so state is correct for prediction
        self._step(data[-1])
        self.last_input = data[-1].copy()

    def predict(self, n_steps: int) -> Array:
        if self.W_out is None:
            raise RuntimeError("Train the ESN first.")
        d = self.W_out.shape[0]
        preds = np.zeros((n_steps, d))
        inp = self.last_input.copy()
        for t in range(n_steps):
            self._step(inp)
            out = self.W_out @ np.concatenate([self.state, inp])
            preds[t] = out
            inp = out
        return preds
