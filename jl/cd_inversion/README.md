# CD Spectroscopy Inversion — Neural Network

Physics-constrained MLP to invert circular dichroism spectra: from measured wavelength-vs-ellipticity curves to protein secondary structure fractions.

## What it does

- Builds a physics-based forward model: CD = f_helix × S_helix + f_sheet × S_sheet + f_coil × S_coil
- Generates 1000 synthetic CD spectra with random secondary structure compositions
- Trains a 3-layer MLP with a physics loss (spectral reconstruction consistency) + MSE
- Compares with/without physics constraint

## Files

```
jl/cd_inversion/
├── app.pluto.jl    # Interactive Pluto notebook
├── main.jl         # Standalone training script
├── src/            # MLP, CD forward model, ternary helpers
├── outputs/        # Precomputed training results
├── Project.toml
├── Manifest.toml
└── README.md       # This file
```

## Running

```bash
# Interactive notebook
cd jl/cd_inversion
julia --project=../.. -e 'using Pluto; Pluto.run()'

# Standalone script (trains MLP, writes outputs/)
julia --project=. main.jl
```
