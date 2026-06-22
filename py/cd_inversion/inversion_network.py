"""Physics-constrained MLP for CD spectrum inversion."""
from __future__ import annotations
import numpy as np


def _relu(x): return np.maximum(0, x)
def _softmax(x): e = np.exp(x - x.max()); return e / e.sum()


class CDInvNet:
    def __init__(self, input_dim: int = 61, hidden_dim: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        k1 = np.sqrt(2 / input_dim); k2 = np.sqrt(2 / hidden_dim)
        self.W1 = rng.standard_normal((hidden_dim, input_dim)) * k1; self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * k2; self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((3, hidden_dim)) * k2;          self.b3 = np.zeros(3)

    def forward_raw(self, x):
        h1 = _relu(self.W1 @ x + self.b1)
        h2 = _relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def predict(self, x):
        return _softmax(self.forward_raw(x))

    def _backprop(self, x, y_true, lambda_phys):
        h1 = _relu(self.W1 @ x + self.b1)
        h2 = _relu(self.W2 @ h1 + self.b2)
        out = self.W3 @ h2 + self.b3
        pred = _softmax(out)
        loss = float(np.mean((pred - y_true) ** 2))

        d = 2 * (pred - y_true) / 3
        S = pred; J = np.diag(S) - np.outer(S, S)
        dout = J @ d
        gW3 = np.outer(dout, h2); gb3 = dout
        dh2 = self.W3.T @ dout * (h2 > 0)
        gW2 = np.outer(dh2, h1); gb2 = dh2
        dh1 = self.W2.T @ dh2 * (h1 > 0)
        gW1 = np.outer(dh1, x); gb1 = dh1
        return gW1, gb1, gW2, gb2, gW3, gb3, loss

    def train(self, spectra, compositions, epochs=25, lr=1e-3,
              lambda_phys=0.1, batch_size=32, seed=0):
        rng = np.random.default_rng(seed)
        N = len(spectra); losses = []
        for ep in range(epochs):
            order = rng.permutation(N); el = 0.0; nb = 0
            for bs in range(0, N, batch_size):
                bidx = order[bs:bs+batch_size]
                gW1=np.zeros_like(self.W1); gb1=np.zeros_like(self.b1)
                gW2=np.zeros_like(self.W2); gb2=np.zeros_like(self.b2)
                gW3=np.zeros_like(self.W3); gb3=np.zeros_like(self.b3); bl=0.0
                for i in bidx:
                    g1,g2,g3,g4,g5,g6,l = self._backprop(spectra[i], compositions[i], lambda_phys)
                    gW1+=g1; gb1+=g2; gW2+=g3; gb2+=g4; gW3+=g5; gb3+=g6; bl+=l
                bsz = len(bidx)
                for attr, grad in [("W1",gW1),("b1",gb1),("W2",gW2),("b2",gb2),("W3",gW3),("b3",gb3)]:
                    setattr(self, attr, getattr(self, attr) - lr * grad / bsz)
                el += bl/bsz; nb += 1
            losses.append(el/nb)
        return losses
