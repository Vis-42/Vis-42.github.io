# Active Matter: Collective Motion and Single-Cell Swimming
### Vicsek Model · Run-and-Tumble Dynamics · MSD Theory

Interactive Pluto notebook + Keynote presentation with Julia-generated animations.

---

## Project Overview

Self-propelled particles (SPPs) continuously inject energy at the individual level, breaking detailed balance and enabling collective behaviors forbidden in equilibrium systems. This project simulates two canonical models of active matter:

- **Vicsek model**: N particles align with neighbors' average heading plus noise; the competition between alignment and noise produces a sharp flocking transition with order parameter φ = ⟨cos θᵢ⟩
- **Run-and-tumble (RT) dynamics**: a single bacterium (E. coli model) propels at constant speed v₀ and reorients at Poisson rate λ; exact analytic MSD interpolates between ballistic (t≪1/λ) and diffusive (t≫1/λ) regimes

The interactive notebook (`app.pluto.jl`) runs both simulations live with sliders. The animations visualize the key physics at each slide.

---

## Files

```
jl/active_matter/
├── app.pluto.jl              # Interactive Pluto notebook
├── presentation.md           # Full slide content
├── README.md                 # This file
│
└── animations/
    ├── run_all.jl            # Render all 7 animations
    ├── Project.toml
    ├── Manifest.toml
    │
    ├── shared/
    │   ├── style.jl          # Dark theme, RES=(1280,720), FPS=30, color palette
    │   └── vicsek.jl         # Vicsek step, φ computation, RT simulation, MSD theory
    │
    ├── s01_motivation.jl     → s01_motivation.mp4
    ├── s02_vicsek_phases.jl  → s02_vicsek_phases.mp4
    ├── s03_order_parameter.jl → s03_order_parameter.mp4
    ├── s04_bifurcation.jl    → s04_bifurcation.mp4
    ├── s05_runtumble_path.jl → s05_runtumble_path.mp4
    ├── s06_msd.jl            → s06_msd.mp4
    └── s07_deff_comparison.jl → s07_deff_comparison.mp4
```

---

## Slide ↔ Animation Correspondence

| Animation | Slide topic | What it shows |
|-----------|-------------|---------------|
| `s01_motivation.mp4` | Active vs passive matter | Ordered flock vs disordered gas: heading-colored arrows with real Vicsek dynamics |
| `s02_vicsek_phases.mp4` | Vicsek phases | N=150 particles in two panels (η=0.5 vs η=3.0): live alignment physics |
| `s03_order_parameter.mp4` | φ(t) traces | Order parameter time series at three noise levels building up progressively |
| `s04_bifurcation.mp4` | Bifurcation diagram | φ vs η, mean ± std band, two densities revealed progressively |
| `s05_runtumble_path.mp4` | Run-and-tumble path | Single RT trajectory with tumble events marked; path grows over time |
| `s06_msd.mp4` | MSD crossover | Log-log MSD at three tumble rates; theory lines (ballistic t², diffusive t) |
| `s07_deff_comparison.mp4` | D_eff vs λ | Scatter points + theory curve for effective diffusivity vs tumble rate |

---

## Running

```bash
cd jl/active_matter/animations
julia --project=. run_all.jl
```

All outputs land in `animations/output/`. Estimated render time: ~3–5 minutes.

### Single animation

```bash
cd jl/active_matter/animations
julia --project=. s02_vicsek_phases.jl
```

### Interactive notebook

```bash
cd jl/active_matter
julia --project=../.. -e 'using Pluto; Pluto.run()'
# Open app.pluto.jl in Pluto UI
```

---

## Physics

**Vicsek model** (discrete time, periodic box L×L):

$$\theta_i(t+1) = \langle \theta_j \rangle_{|r_i - r_j| < R} + \eta\,\xi_i$$

where ξᵢ ∈ [−π, π] is uniform noise and η is the noise amplitude. Order parameter φ = |N⁻¹Σeⁱθᵢ|. The transition at ηc depends on density ρ = N/L².

**Run-and-tumble MSD** (exact, OU-based):

$$\text{MSD}(t) = 2v_0^2 \left[ \frac{t}{\lambda} + \frac{e^{-\lambda t} - 1}{\lambda^2} \right]$$

giving ballistic (∝ t²) at short times and diffusive (∝ t) at long times with D_eff = v₀²/λ.

---

## Shared Infrastructure

**`shared/style.jl`**: dark theme, 1280×720 resolution, FPS=30, color assignments (ordered phase = sky blue, disordered = amber, theory = white, trajectory = teal).

**`shared/vicsek.jl`**: `vicsek_step!()` for one discrete Vicsek step, `run_vicsek()` for full simulation, `run_tumble_sim()` for RT trajectory, `msd_theory()` for the analytic MSD formula.
