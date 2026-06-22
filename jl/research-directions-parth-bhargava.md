# Solo Research Projects for Parth Bhargava
## A Literature-Informed Map of Frontier Computational Physics Problems

*Prepared April 2026 | Based on deep literature surveys across nonlinear dynamics, biophysics, network science, physics-informed ML, and computational materials science.*

---

## Profile Summary

Year-2 Physics BSc at NUS (GPA 4.36), upcoming exchange at UT Austin. 15+ experiments across condensed matter, spectroscopy, plasma, biophysics. Computational modeling in Python, Julia, MATLAB. Focus on nonlinear dynamics and statistical methods. Research interests: complex systems, biological complexity, network science, emergence, computational modeling.

**Key differentiator:** Most computational physics students have *either* strong coding *or* deep experimental skills. You have both, plus domain-specific expertise in spectroscopy (XRD, Raman, fluorescence, CD). Projects that bridge computation with spectroscopic data are your competitive edge.

---

## How to Read This Document

I've surveyed the frontier of research across **five major directions** aligned with your interests. For each, I've identified:

- The current state of the field and key recent papers (2024-2026)
- Specific open problems where solo computational work is possible
- Concrete project specifications (scope, effort, tools, target journals)
- Why the project is novel and publishable

The projects are ranked in a **final synthesis** at the end. Start there if you want the punchline first.

---

## Direction 1: Nonlinear Dynamics & Chaos — Computational Frontiers

### 1A. Reservoir Computing with Chaotic Systems

**What it is:** Echo State Networks (ESNs) and reservoir computing use a fixed, randomly wired recurrent network as a "reservoir" of rich dynamics. Only the output layer is trained. Recent work uses chaotic dynamical systems themselves as computational substrates.

**Key recent work:**
- "Emerging opportunities and challenges for the future of reservoir computing" — *Nature Communications* (2024)
- "Learning spatiotemporal chaos using next-generation reservoir computing" — *Chaos* (2023)
- "Deterministic Reservoir Computing for Chaotic Time Series Prediction" — *Scientific Reports* (2025)
- "Storage and selection of multiple chaotic attractors in minimal reservoir computers" — arXiv (2025)

**Open problems:**
1. **Minimal reservoir design** — What is the smallest network (fewest nodes, sparsest connectivity) that can still predict N-dimensional chaos? Current approaches are ad hoc.
2. **Prediction horizon limits** — Why do RC predictions fail beyond ~1-2 Lyapunov times? Is this fundamental or algorithmic?
3. **Multi-attractor learning** — Can one reservoir simultaneously learn multiple chaotic systems?

**PROJECT: Systematic Phase Diagram of Minimal Reservoir Computing**

Build ESNs with deliberately minimal hidden layers (10-50 nodes) with tunable sparsity. Train on trajectory data from Lorenz-63, Rössler, and Hénon systems. Systematically vary network sparsity (5%-50%), spectral radius (0.7-1.0), and input scaling. Map the (sparsity, spectral radius) parameter space to identify the *minimal* configuration that maintains prediction fidelity across 1-5 Lyapunov time horizons.

- **Effort:** ~2000 lines Python (NumPy, SciPy, custom ESN). 2-3 months.
- **Compute:** Laptop. Data generation: seconds. Training: minutes.
- **Tools:** PyESN or custom implementation. RK4 for data generation.
- **Target journals:** *Chaos*, *Physical Review E*, *Neural Networks*
- **Why novel:** Directly addresses the open question of minimal reservoir design with systematic parameter sweeps nobody has done.

---

### 1B. SINDy — Sparse Identification of Nonlinear Dynamics

**What it is:** Brunton & Kutz's SINDy algorithm (PNAS, 2016, 2000+ citations) discovers governing equations from data by sparse regression over a library of candidate functions. The bottleneck: you must *choose the library* manually.

**Key recent work:**
- Brunton et al. "Machine learning for sparse nonlinear modeling and control" — *Annual Review of Control* (2025)
- "LES-SINDy: Laplace-Enhanced SINDy" — *J. Computational Physics* (2025)
- "SINDyG: sparse identification from graph-structured data" — *J. Complex Networks* (2024)
- "Structured Kolmogorov-Arnold Neural ODEs" — arXiv (2025)

**Open problems:**
1. **Automated library selection** — How to choose basis functions when underlying physics is unknown?
2. **Noise robustness** — SINDy struggles above ~5% noise.
3. **Parametric discovery** — Discovering how equations *change* as system parameters vary.

**PROJECT: SINDy with Sequential Library Expansion**

Implement SINDy on synthetic data from Van der Pol, Duffing, and driven pendulum systems. Instead of a fixed library, start with polynomials up to degree 2, evaluate residuals, then iteratively add terms (sin, cos, rational, exponential) until the sparse solution converges. Compare convergence speed, accuracy, and robustness to noise against fixed full-library SINDy.

- **Effort:** ~1500 lines Python (PySINDy + custom). 2-3 months.
- **Compute:** Laptop. Under 5 minutes per run.
- **Tools:** PySINDy, scikit-learn, RK4 for data generation.
- **Target journals:** *SIAM J. Applied Dynamical Systems*, *Physical Review E*, *Neural Networks*
- **Why novel:** Automates SINDy's biggest bottleneck. No one has published a systematic sequential expansion approach.

---

### 1C. Chimera States in Coupled Oscillator Networks

**What it is:** Chimera states are the striking coexistence of synchrony and incoherence in networks of identical oscillators. First discovered theoretically by Kuramoto (2002), now observed experimentally, but poorly understood in realistic network topologies.

**Key recent work:**
- "Chimera states in an adaptive higher-order network of Kuramoto oscillators" — *Nonlinear Dynamics* (2025)
- "Chimera-like states in networks with mixed repulsive coupling" — *Commun. Nonlinear Sci.* (2025)
- "Persistence of chimera states in real-world networks" — *Eur. Phys. J. B* (2023)

**Open problems:**
1. **Topology dependence** — Which network structures *guarantee* chimeras? Rings work; small-world, modular, adaptive — results are incomplete.
2. **Chimera death and breathing chimeras** — Newer variants lack theoretical understanding.
3. **Control and stabilization** — Chimeras have tiny basins of attraction.

**PROJECT: Chimera Phase Diagram in Adaptive Kuramoto Networks**

Simulate N=100-200 Kuramoto oscillators in a ring topology with tunable coupling range. Systematically map the (coupling strength K, coupling range R, frequency disorder Δω) parameter space. Detect chimera states using the local order parameter metric. Then introduce adaptive rewiring (oscillators strengthen coupling to phase-aligned neighbors) and measure how the chimera region expands or contracts.

- **Effort:** ~800 lines Python. 2 months.
- **Compute:** RK4 on coupled ODEs. ~10-20 minutes per parameter sweep.
- **Target journals:** *Chaos*, *Physical Review E*, *Frontiers in Applied Mathematics*
- **Why novel:** Topology × parameter phase diagrams are incomplete. Adaptive networks are a 2024-2025 frontier.

---

### 1D. Data-Driven Discovery of Conservation Laws

**What it is:** Using ML to automatically discover conserved quantities (energy, momentum, Casimir invariants) from trajectory data, without knowing the equations of motion.

**Key recent work:**
- "Machine Learning Conservation Laws of Dynamical Systems" — *Physical Review E* (Feb 2025) — kernel-based approach
- "Discovering conservation laws using optimal transport and manifold learning" — *Nature Communications* (2023)
- "Machine Learning Conservation Laws from Trajectories" — *Physical Review Letters* (2021)

**Open problems:**
1. **Symbolic extraction** — ML discovers conserved quantities numerically; extracting symbolic forms (e.g., E = p²/2m + mgx) remains unsolved.
2. **Broken symmetries** — Real systems have approximate conservation laws. Detecting and quantifying breaking is open.
3. **High-dimensional systems** — Most papers test on 2D-3D toy systems.

**PROJECT: Kernel-Based Conservation Law Discovery + Symbolic Extraction**

Generate high-quality trajectory data from simple pendulum, 2-body Kepler problem, and Duffing oscillator. Apply kernel ridge regression (RBF kernel) to discover conserved quantities. Then pipe the numerical conserved function through PySR (symbolic regression) to extract closed-form expressions. Benchmark: does the pipeline recover E, L exactly?

- **Effort:** ~800 lines Python. 2 months.
- **Compute:** Laptop. Under 5 minutes.
- **Tools:** scikit-learn (KRR), PySR (symbolic regression), RK4.
- **Target journals:** *Physical Review E*, *J. Computational Physics*
- **Why novel:** Combines the Feb 2025 kernel approach with symbolic regression — nobody has closed this loop.

---

### 1E. KAM Theory — Computational Hamiltonian Chaos

**What it is:** KAM theorem guarantees invariant tori in near-integrable Hamiltonian systems survive small perturbations. But computing *when* tori break is hard, and standard integrators (RK4) violate symplecticity.

**PROJECT: KAM Torus Breakdown in Hénon-Heiles via Frequency Analysis**

Simulate the Hénon-Heiles system (2 DOF near-integrable Hamiltonian) for varying perturbation strength. Use frequency analysis and Fast Lyapunov Indicators to detect torus breakup. Compare RK4 against symplectic integrators (Störmer-Verlet) over 10⁶ steps to quantify when symplectic methods become necessary.

- **Effort:** ~1200 lines Python. 2-3 months.
- **Target journals:** *Physical Review E*, *SIAM JADS*, *Celestial Mechanics and Dynamical Astronomy*

---

## Direction 2: Biophysics & Biological Complexity

### 2A. Intrinsically Disordered Proteins & Phase Separation

**What it is:** IDPs don't fold into stable structures but drive liquid-liquid phase separation (LLPS) — forming membraneless organelles. Predicting which sequences undergo phase separation from first principles remains open.

**Key recent work:**
- Hybrid Resolution (HyRes) protein models (2024) — bridge backbone-level and coarse-grained
- Active learning + ML for saturation concentration prediction — *PNAS* (2025)
- LASSI lattice model for multivalent protein phase separation — open-source, Python-accessible

**PROJECT: Sequence-to-Phase-Separation Prediction via Lattice Models + ML**

Use the LASSI lattice model to simulate phase separation for a library of IDP sequences with varying charge patterning, hydrophobic clustering, and disorder propensity. Train a neural network on sequence features to predict saturation concentration. Validate against published CD/fluorescence measurements from IDP databases (DisProt, PED).

- **Effort:** 3-4 months. LASSI is open-source.
- **Compute:** Laptop CPU.
- **Target journals:** *Biophysical Journal*, *J. Physical Chemistry B*, *Proteins*
- **Why novel:** Combines lattice simulation + ML + spectroscopic validation. Your CD/fluorescence expertise is rare in this space.

---

### 2B. Protein Conformational Landscapes Beyond AlphaFold

**What it is:** AlphaFold predicts structure *if stable* but cannot estimate ΔG or generate conformational ensembles. This is a massive open problem.

**Key recent work:**
- IFUM (2025, *Nature Communications*): deep NN jointly estimating ΔG + ensemble of folded/unfolded states
- Monte Carlo landscape sampling for small proteins (~40 residues) — *Nature Scientific Reports* (2024)
- Gō models outperform all-atom MD for capturing native-state fluctuations

**PROJECT: Gō Model Free Energy Landscapes with Spectroscopic Validation**

Implement an off-lattice Gō model for a small protein (Trp-cage, WW-domain). Use replica-exchange Monte Carlo to sample the folding free energy landscape. Identify transition states and folding intermediates. Compare predicted secondary structure content at each stage against published fluorescence and CD unfolding data.

- **Effort:** ~500 lines Python + Monte Carlo. 3-4 months.
- **Compute:** Laptop CPU.
- **Target journals:** *Scientific Reports*, *Computational Biology and Chemistry*, *Biophysical Journal*
- **Why novel:** Bridging Gō model predictions with spectroscopic observables is rarely done. Your experimental background makes this credible.

---

### 2C. Physics-Informed Spectroscopic Inversion (YOUR SUPERPOWER)

**What it is:** Inverse problems — going from spectrum to structure — are hard but transformative. Recent work uses physics-constrained ML to invert spectroscopic data.

**Key recent work:**
- Physics-guided ML for optical spectrum inversion — *Scientific Reports* (2024)
- Deep learning on CD spectra predicting secondary structure with 0.96 correlation — *Analytical Chemistry* (2025)
- Fluorescence lifetime imaging + ML (2024)

**PROJECT: Physics-Constrained Neural Network for CD Inversion of Disordered Proteins**

Build a forward model that predicts CD spectra from structural ensembles. Then train a physics-informed neural network (loss = spectral fit + physical constraints like radius of gyration, charge interactions) to *invert*: given observed CD, predict the conformational ensemble. Validate against small-angle X-ray scattering (SAXS) data from literature.

- **Effort:** 4-6 months. May need GPU for training (free via Google Colab).
- **Target journals:** *Biophysical Journal*, *J. Physical Chemistry B*, *Proteins*
- **Why novel:** CD inversion for IDPs is nearly untouched. Combining it with physical constraints is very new.

---

## Direction 3: Network Science, Information Theory & Emergence

### 3A. Information Flow on Networks

**Key concepts:** Transfer entropy (TE) quantifies directed information flow between coupled dynamical systems. Lizier's local TE framework decomposes this spatiotemporally.

**PROJECT: Transfer Entropy Estimator Benchmarking on Coupled Chaotic Systems**

Compare TE estimators (plug-in, Kraskov k-NN, symbolic binning) on synthetic coupled oscillators (Rössler, Hénon maps) with known coupling strengths. Systematically vary coupling strength, noise level, and time-series length. Which estimator recovers the true coupling structure most accurately with limited data?

- **Effort:** 2-3 months. Julia (TransferEntropy.jl) or Python (JIDT, IDTxl).
- **Target journals:** *Chaos*, *Physical Review E*, *PNAS Nexus*

---

### 3B. Phase Transitions on Higher-Order Networks

**What it is:** Standard percolation on pairwise networks produces second-order phase transitions. But higher-order interactions (simplicial complexes, hypergraphs) produce novel explosive and discontinuous transitions.

**Key recent work:**
- "Hyperedge overlap drives explosive transitions" — *Nature Communications* (2024)
- "The dynamic nature of percolation on networks with triadic interactions" — *Nature Communications* (2023)

**PROJECT: Percolation on Simplicial Complexes — Critical Exponents and Universality**

Implement percolation on simplicial complexes of varying dimensions. Track the giant connected component, susceptibility, and compute critical exponents via finite-size scaling. Does k-dimensional percolation have different universality classes than (k+1)-dimensional? Compare against standard network percolation.

- **Effort:** 2-3 months. Python (NetworkX + custom topology code).
- **Target journals:** *Physical Review E*, *Chaos*, *SIAM JADS*
- **Why novel:** Higher-order percolation critical exponents are largely unmapped computationally.

---

### 3C. Causal Inference via Convergent Cross-Mapping

**What it is:** Granger causality fails on deterministic dynamical systems. Sugihara's CCM (Nature, 2012) uses Takens embedding to detect causal influence in coupled chaotic systems.

**PROJECT: Network CCM for Inferring Directed Coupling in Oscillator Networks**

Implement multi-variable CCM to infer directed coupling networks from time-series data of coupled Kuramoto oscillators. Benchmark against known coupling structure. Test how embedding dimension, noise, and time-series length affect network inference accuracy. Compare against transfer entropy and Granger causality.

- **Effort:** 3 months. Python (skccm) or Julia (DynamicalSystems.jl).
- **Target journals:** *Chaos*, *Physical Review E*, *Methods in Ecology and Evolution*

---

### 3D. Quantifying Emergence — Causal Emergence in Cellular Automata

**What it is:** Erik Hoel's causal emergence framework (2013+) shows that coarse-grained descriptions can have *more* causal power than fine-grained ones. Effective Information (EI) measures this.

**Key recent work:**
- "Causal Emergence 2.0" — arXiv (Feb 2025)
- "Finding emergence in data by maximizing effective information" — *NSR* (2024)

**PROJECT: Causal Emergence Across Wolfram's Elementary Cellular Automata**

Compute EI for all 256 elementary cellular automata rules. Implement Monte Carlo sampling of input distributions to estimate EI at micro and macro scales. Which Wolfram classes exhibit genuine causal emergence? Does emergence correlate with computational class (Class IV)?

- **Effort:** 2-3 months. ~600 lines Python/Julia.
- **Compute:** Laptop.
- **Target journals:** *Entropy*, *Chaos*, *Phil. Trans. Royal Society*
- **Why novel:** Systematic EI computation across all 256 rules hasn't been published. Connecting emergence to Wolfram's computational classification is an open question.

---

### 3E. Statistical Mechanics of Neural Network Loss Landscapes

**What it is:** Neural networks map to spin glass models. Training is a phase transition: disorder → hidden order. The 2024 Physics Nobel context has made this connection mainstream.

**Key recent work:**
- "Neural Networks as Spin Models: From Glass to Hidden Order" — arXiv (2024)
- "Phase transitions reveal hierarchical structure in deep neural networks" — arXiv (Dec 2024)
- "Exploring Loss Landscapes through the Lens of Spin Glass Theory" — arXiv (2024)

**PROJECT: Spin Glass Analysis of Small Neural Network Training Dynamics**

Train a small NN (50-100 hidden units) on a toy task (XOR, spiral classification). Compute the Hessian eigenvalue spectrum at different stages of training. Track whether replica symmetry breaking signatures (following TAP analysis) emerge and dissolve during training. Does the glass-to-order transition correlate with generalization?

- **Effort:** 3-4 months. JAX or PyTorch.
- **Target journals:** *Physical Review Letters*, *Nature Machine Intelligence*, *ICLR/NeurIPS*
- **Why novel:** Finite-size (realistic N) analysis of the glass-to-order transition is largely unexplored. Most theory assumes infinite width.

---

## Direction 4: Physics-Informed Machine Learning

### 4A. Symbolic Regression for Physical Laws

**What it is:** PySR and AI Feynman discover symbolic equations from data. The Feynman dataset (120 equations, Easy/Medium/Hard) benchmarks the field.

**Key recent work:**
- PySR 0.19+ with LLM integration — *Genetic Programming & Evolvable Machines* (2024)
- "Rethinking Symbolic Regression Benchmarks" — arXiv (2024)

**PROJECT: Symbolic Regression with Dimensional Analysis Constraints on Hard Feynman Problems**

Apply PySR to the Hard subset of the Feynman dataset with a custom loss function that penalizes dimensionally inconsistent expressions. Add physics-informed priors: conservation law constraints, symmetry requirements. Benchmark against vanilla PySR and other SR methods.

- **Effort:** 3-4 weeks for prototype, 2 months for full paper.
- **Tools:** PySR, custom loss functions.
- **Target journals:** *PNAS*, *Science Advances*, *Nature Machine Intelligence*
- **Why novel:** Dimensional analysis as a hard constraint in symbolic regression is surprisingly underexplored.

---

### 4B. Weak-Form PINNs for Novel PDE Systems

**What it is:** Standard PINNs fail on systems with shocks and discontinuities. Weak-form PINNs replace strong differential equations with integral conservation laws, keeping fluxes bounded.

**Key recent work:**
- "Weak-form PINNs for Burgers' equation" — *Scientific Reports* (2025)
- "Challenges in shock fronts" — arXiv (2025)

**PROJECT: Weak-Form PINNs for Shallow Water Equations**

Apply weak-form PINNs to the shallow water equations (relevant to tsunami modeling, dam breaks). Most PINN work focuses on Burgers and Euler equations — shallow water is underexplored. Compare against standard finite-volume solvers for accuracy and speed.

- **Effort:** 4-6 weeks. DeepXDE or custom JAX.
- **Target journals:** *J. Computational Physics*, *Scientific Reports*, *Computers & Fluids*

---

### 4C. XRD Inverse Problem with Neural Networks

**What it is:** Predicting crystal structure (space group, lattice parameters, atomic positions) directly from X-ray diffraction patterns.

**Key recent work:**
- DiffractGPT — *J. Phys. Chem. Letters* (2024): generative transformer for XRD inversion

**PROJECT: Neural Network for Crystal Structure Identification from Powder XRD**

Train a transformer or CNN to predict crystal structure (space group + lattice parameters) from simulated powder XRD patterns. Start with a curated database of known structures. Test generalization to noisy and partially observed patterns. Validate on your own experimental XRD data from coursework.

- **Effort:** 6-8 weeks. PyTorch, crystallography databases.
- **Target journals:** *npj Computational Materials*, *Chemistry of Materials*
- **Why novel:** Your hands-on XRD experience gives you physical intuition most ML researchers lack. End-to-end structure determination from powder XRD is actively developing.

---

### 4D. Differentiable Physics Simulations

**What it is:** JAX-based differentiable simulators enable backpropagation through physics simulations, solving inverse problems by gradient descent.

**Key frameworks (2024-2025):**
- JAX-MD (molecular dynamics), JAX-MPM (continuum mechanics), TORAX (tokamak transport)

**PROJECT: Differentiable Pendulum — Learning Physical Parameters from Trajectory Data**

Implement a differentiable multi-pendulum simulator in JAX. Given noisy observed trajectories, use gradient descent to infer physical parameters (masses, lengths, damping coefficients, driving frequencies). Benchmark against Bayesian inference and SINDy. Scale from double to triple pendulum.

- **Effort:** 2-3 months. JAX.
- **Target journals:** *NeurIPS workshop*, *J. Computational Physics*
- **Why novel:** Connects directly to your triple pendulum chaos project. Adds a data-driven dimension.

---

## Final Synthesis: Ranked Recommendations

### Tier 1: Highest Impact × Best Profile Match × Solo Feasibility

These are the projects I'd bet on for you specifically.

**1. Conservation Law Discovery + Symbolic Extraction (Direction 1D)**
Why: Combines your numerical methods skills with frontier ML. The kernel approach (Feb 2025) + PySR pipeline is a clean, novel contribution. You understand the physics deeply (Hamiltonian mechanics, Lagrangian formalism). Publishable in 3-4 months. Physical Review E loves this kind of work.

**2. Causal Emergence in Cellular Automata (Direction 3D)**
Why: Perfectly suited to your interest in emergence and statistical reasoning. Computationally clean — you're computing information-theoretic quantities over discrete systems. The connection to Wolfram's computational classes is an open and intellectually exciting question. You can go deep on the theory. Publishable in Entropy or Chaos.

**3. Physics-Constrained CD Inversion for IDPs (Direction 2C)**
Why: This is where you have a genuine competitive advantage. Your spectroscopy experience is rare among computational students. Building the forward model (structure → CD) and then inverting it with physical constraints is serious physics + ML. Higher effort but potentially higher impact. Biophysical Journal would be interested.

**4. SINDy with Sequential Library Expansion (Direction 1B)**
Why: Clean algorithmic contribution to a well-known framework. You already understand the physics of the test systems (driven pendula, Duffing, Van der Pol). PySINDy exists as a starting point. Fast time-to-publication.

### Tier 2: Excellent Projects, Slightly Higher Bar

**5. Chimera Phase Diagram in Adaptive Networks (Direction 1C)** — Clean computational sweep, directly publishable, but requires patience with parameter space exploration.

**6. Symbolic Regression with Dimensional Constraints (Direction 4A)** — Fastest path to a high-profile publication if results are strong. PySR + dimensional analysis is surprisingly underexplored.

**7. Percolation on Simplicial Complexes (Direction 3B)** — Timely (higher-order networks are hot), computationally tractable, novel critical exponents.

**8. Spin Glass Analysis of NN Training (Direction 3E)** — Most ambitious. If you pull it off, it's Physical Review Letters material. But requires deeper ML infrastructure knowledge.

### Tier 3: Strong but More Conventional

**9. Minimal Reservoir Computing (Direction 1A)** — Solid contribution, clear benchmarks, but the field is crowded.

**10. Gō Model Landscapes + Spectroscopic Validation (Direction 2B)** — Clean biophysics project, publishable, but incrementally novel.

**11. Weak-Form PINNs for Shallow Water (Direction 4B)** — Applied ML contribution; publishable but more engineering than science.

**12. XRD Inverse Problem (Direction 4C)** — Leverages your XRD background beautifully, but requires substantial data curation.

---

## Practical Next Steps

### Immediate (This Week)
1. Pick ONE project from Tier 1. Read the 3-5 key papers cited.
2. Install the core tools (PySINDy, PySR, JIDT, LASSI — whichever is relevant).
3. Reproduce a baseline result from one of the key papers.

### Short-Term (Weeks 2-4)
4. Identify the specific gap or extension you'll pursue.
5. Generate synthetic data / set up the computational pipeline.
6. Produce your first novel result (even if preliminary).

### Medium-Term (Months 2-4)
7. Systematic experiments — parameter sweeps, comparisons, ablations.
8. Write up results. Target a specific journal.
9. Share preprint on arXiv for visibility.

### Skills to Develop in Parallel
- **JAX** — the differentiable computing framework. Essential for modern physics-ML.
- **Information theory** — read Cover & Thomas, or Bialek's online lectures.
- **Graph theory / computational topology** — NetworkX, and understand simplicial complexes.

---

## Key References by Direction

### Nonlinear Dynamics & Chaos
- Brunton & Kutz, "Discovering governing equations from data" — PNAS (2016)
- "Emerging opportunities for reservoir computing" — Nature Communications (2024)
- "Machine Learning Conservation Laws" — Physical Review E (Feb 2025)
- Lukoševičius & Jaeger, "Practical guide to ESNs" — Neural Networks (2012)

### Biophysics
- LASSI lattice model — PLOS Computational Biology (2019)
- IFUM protein stability — Nature Communications (2025)
- Physics-guided ML inversion — Scientific Reports (2024)
- DisProt database — Nucleic Acids Research (2024)

### Network Science & Emergence
- Lizier et al., "Differentiating information transfer and causal effect" — EPJ B (2010)
- Sugihara et al., "Detecting causality in complex ecosystems" — Science (2012)
- Hoel, "Causal Emergence 2.0" — arXiv (Feb 2025)
- "Hyperedge overlap drives explosive transitions" — Nature Communications (2024)
- "Neural Networks as Spin Models" — arXiv (2024)

### Physics-Informed ML
- PySR — Genetic Programming & Evolvable Machines (2024)
- "Weak-form PINNs" — Scientific Reports (2025)
- DiffractGPT — J. Phys. Chem. Letters (2024)
- JAX-MD — github.com/jax-md/jax-md

### Computational Tools
- PySINDy (SINDy): pysindy.readthedocs.io
- PySR (Symbolic Regression): github.com/MilesCranmer/PySR
- JIDT (Information Dynamics): github.com/jlizier/jidt
- TransferEntropy.jl: juliadynamics.github.io
- LASSI (IDP phase separation): lassi.seas.upenn.edu
- ReservoirPy (ESN): github.com/reservoirpy/reservoirpy
- DeepXDE (PINNs): github.com/lululxvi/deepxde
- JAX-MD: github.com/jax-md/jax-md
- PyPhi (IIT): github.com/wmayner/pyphi

---

*This document represents a synthesis of literature surveys covering ~200 recent papers across five research directions. All projects are designed to be executable solo on a laptop with no special compute infrastructure.*
