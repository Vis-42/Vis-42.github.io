# Protein Folding Landscape — Gō Model + Replica Exchange MC

Coarse-grained protein folding simulation using an off-lattice Gō model for a 25-bead Trp-cage analog, with replica exchange Monte Carlo and free energy profile F(Q).

## What it does

- Builds a 25-bead helix-like native structure with ~10 native contacts
- Runs Metropolis MC with a Lennard-Jones 12-10 Gō contact potential
- Performs replica exchange MC across T = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
- Computes F(Q) free energy profile from the sampled Q trajectories

## Files

```
jl/protein_landscape/
├── app.pluto.jl              # Interactive Pluto notebook
├── presentation.md           # Full slide content
├── README.md                 # This file
│
└── animations/
    ├── run_all.jl            # Render all 6 animations
    ├── Project.toml
    ├── Manifest.toml
    │
    ├── shared/
    │   ├── style.jl          # Dark theme, RES=(1280,720), FPS=30, color palette
    │   └── gomodel.jl        # Gō model energy, MC step, replica exchange, F(Q), Q order param
    │
    ├── s01_funnel.jl         → s01_funnel.mp4
    ├── s02_go_potential.jl   → s02_go_potential.mp4
    ├── s03_mc_trajectory.jl  → s03_mc_trajectory.mp4
    ├── s04_freeenergy.jl     → s04_freeenergy.mp4
    ├── s05_replica_exchange.jl → s05_replica_exchange.mp4
    └── s06_barrier.jl        → s06_barrier.mp4
```

## Slide ↔ Animation Correspondence

| Animation | Slide topic | What it shows |
|-----------|-------------|---------------|
| `s01_funnel.mp4` | Folding funnel schematic | Energy vs Q funnel and rugged landscape curves building progressively |
| `s02_go_potential.mp4` | 12-10 Gō contact potential | V_gō(r) and V_LJ(r) curves building up; marks native contact distance |
| `s03_mc_trajectory.mp4` | Q(t) trajectories | Q(t) traces at T = 0.5, 1.0, 2.0 growing over time; folded/unfolded labels |
| `s04_freeenergy.mp4` | F(Q) at four temperatures | F(Q) curves for T below/at/above T_fold building progressively |
| `s05_replica_exchange.mp4` | Replica exchange temperature walk | Temperature ladder visualization with exchange events |
| `s06_barrier.mp4` | Free energy barrier | F(Q) two-basin structure with barrier height ΔF annotated |

## Running

```bash
cd jl/protein_landscape/animations
julia --project=. run_all.jl
```

All outputs land in `animations/output/`. Estimated render time: ~4–6 minutes.

### Interactive notebook

```bash
cd jl/protein_landscape
julia --project=../.. -e 'using Pluto; Pluto.run()'
```
