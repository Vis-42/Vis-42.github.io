# Causal Inference: Transfer Entropy, CCM, and Causal Emergence
### Transfer Entropy · Convergent Cross-Mapping · Elementary Cellular Automata

Interactive Pluto notebook.

---

## Project Overview

Causality is not correlation. This project implements and compares three fundamentally different approaches to inferring causal relationships from time-series data:

- **Transfer entropy (TE)**: information-theoretic; measures how much X's past reduces uncertainty about Y's future, estimated via KSG k-nearest-neighbor entropy
- **Convergent Cross-Mapping (CCM)**: attractor-based; tests whether X's shadow manifold (delay embedding of Y) can reconstruct X — the signature of dynamical coupling
- **Causal emergence**: coarse-graining-based; measures whether a macro description has strictly higher effective information (EI) than its micro substrate, applied across all 256 ECA rules

Each method operationalizes causality differently and gives different answers on the same data. Understanding when each approach wins is the central lesson.

---

## Files

```
jl/causal_inference/
├── app.pluto.jl    # Interactive Pluto notebook
└── README.md       # This file
```

---
## Running

```bash
cd jl/causal_inference
julia --project=../.. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl in the Pluto UI
```

---

## Physics / Theory

**Transfer Entropy** (Schreiber 2000):

$$\text{TE}_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log \frac{p(y_{t+1} | y_t^{(k)}, x_t^{(l)})}{p(y_{t+1} | y_t^{(k)})}$$

Estimated via KSG (k-nearest-neighbor) approach: TE = H(Y_future | Y_past) − H(Y_future | Y_past, X_past).

**Convergent Cross-Mapping** (Sugihara et al. 2012): if X causes Y, then X leaves a footprint in Y's attractor — Y's shadow manifold Mʸ can reconstruct X using nearest-neighbor prediction. Cross-map skill ρ increases with library length L (convergence) and is asymmetric between causes and effects.

**Causal Emergence** (Hoel et al. 2013): ΔEI = EI(Φ∘T∘Φ⁻¹) − EI(T) where Φ is a coarse-graining. Positive ΔEI means the macro-level causal structure is richer than the micro-level one.
