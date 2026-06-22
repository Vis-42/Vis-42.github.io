# Computational Projects for Your Portfolio

> **Status (June 2026):** All 13 explorations listed here have been built and are live at `parthbhargava.net`. This document is the original project planning reference — kept for context on what each project was motivated by and where it sits in the research landscape.

---

## Project Options (Ranked by Breadth of Signal to PIs)

### 1. Vicsek Model — Phase Diagram + Order Parameter Analysis ⭐ Recommended
**Relevant to:** Active matter, nonlinear dynamics, stat mech, collective behavior

**What to build:**
- Simulate N self-propelled particles with Vicsek alignment + noise
- Sweep over (noise η, density ρ) → generate full phase diagram
- Measure order parameter φ = |⟨v̂⟩| as function of η at multiple densities
- Show the phase transition (ordered → disordered) clearly
- Optional: finite-size scaling analysis at the transition point
- Optional: animate the flock transition as a gif

**Tech stack:** Python (numpy + matplotlib) or Julia. No GPU needed.

**Why it signals well:** Directly implements the founding model of active matter. Shows you understand phase transitions, order parameters, and numerical sampling. Every active matter PI will immediately know you're serious.

**Reference:** Vicsek et al., Phys. Rev. Lett. **75**, 1226 (1995) — [arXiv:cond-mat/0611743](https://arxiv.org/abs/cond-mat/0611743)

---

### 2. SINDy on Lorenz + a Biological Oscillator ⭐ Recommended
**Relevant to:** Nonlinear dynamics, data-driven physics, ML-physics, biological rhythms

**What to build:**
- Implement SINDy (Sparse Identification of Nonlinear Dynamics) from scratch
- Apply to Lorenz system: recover equations from noisy time-series
- Apply to a biological oscillator (FitzHugh-Nagumo or van der Pol)
- Show how noise level and library size affect recovery accuracy
- Bonus: apply to a real experimental time series (ECG, EEG, or any chaotic time series from a database)

**Tech stack:** Python. Reference implementation: github.com/dynamicslab/pysindy

**Why it signals well:** SINDy is Brunton's flagship contribution. Shows you can bridge dynamical systems theory with data-driven methods. Directly relevant to Gilpin (UT Austin), Brunton (UW), and any biophysics PI using time-series analysis.

**Reference:** Brunton, Proctor, Kutz, PNAS **113**, 3932 (2016) — [arXiv:1509.03580](https://arxiv.org/abs/1509.03580)

---

### 3. Kuramoto Model — Synchronization + Bifurcation Diagram
**Relevant to:** Nonlinear dynamics, network science, coupled oscillators

**What to build:**
- Simulate N coupled oscillators with Kuramoto coupling K
- Find the critical coupling Kc for synchronization transition
- Plot order parameter r vs K — show the bifurcation clearly
- Vary the network topology (all-to-all, random Erdős-Rényi, scale-free) and compare Kc
- Phase portrait animation of oscillators going in/out of sync

**Tech stack:** Python.

**Why it signals well:** Kuramoto is the standard model for synchronization. Strogatz wrote the textbook on it. Putting it on graph topologies bridges nonlinear dynamics + network science — relevant to both communities.

**Reference:** Strogatz, Physica D **143**, 1 (2000). Also Strogatz textbook Chapter 4.

---

### 4. Reaction-Diffusion Turing Patterns + Bifurcation Analysis
**Relevant to:** Pattern formation, biophysics, nonlinear dynamics

**What to build:**
- Implement Gray-Scott or Turing (activator-inhibitor) reaction-diffusion PDE
- Finite-difference solver on a 2D grid
- Sweep over (feed rate f, kill rate k) → generate the Pearson diagram of pattern types
- Optional: track bifurcations as parameters cross boundaries

**Tech stack:** Python (numpy). Can use scipy for ODEs.

**Why it signals well:** Pattern formation is central to both biophysics (morphogenesis) and nonlinear dynamics. Shows PDE numerics + visual parameter sweeps.

---

### 5. Run-and-Tumble Particle Dynamics
**Relevant to:** Active matter, non-equilibrium stat mech, biophysics (bacterial swimming)

**What to build:**
- Simulate run-and-tumble particles in 2D (constant speed + Poisson tumbling)
- Measure mean-squared displacement vs time — show crossover from ballistic to diffusive
- Add confinement (box or harmonic trap) and measure steady-state density profiles
- Optional: derive an effective diffusion coefficient and compare to simulation

**Tech stack:** Python.

**Why it signals well:** Directly relevant to ICTS work (Kundu, Sabhapandit) and NCBS (Thutupalli). Shows you know the biological physics of bacteria.

---

## GitHub Repo Structure (Template)

```
your-project/
├── README.md          ← 500-word writeup: motivation, methods, results, what's next
├── notebooks/
│   ├── 01_simulation.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_figures.ipynb
├── src/
│   └── model.py       ← clean, documented code
├── figures/
│   └── phase_diagram.png
└── requirements.txt
```

**README must include:**
1. One-paragraph motivation (why is this interesting?)
2. What you implemented
3. Key result (one figure, described clearly)
4. "Next steps" — this shows intellectual engagement beyond the assignment

---

## After the Project

- Make the repo public. Put the link on your CV and in every cold email.
- Write a short summary tweet or LinkedIn post — some PIs actually look.
- Consider submitting to the [Undergraduate Research Journal of Physics](https://urjp.org/) if the writeup is strong.
- Use it as the basis for your UROP proposal.

---

*Links: [[game-plan]] · [[cold-email-guide]] · [[fields/active-matter]] · [[fields/nonlinear-dynamics]]*
