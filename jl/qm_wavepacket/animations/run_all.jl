#!/usr/bin/env julia
# run_all.jl — Render every LIGHTWEIGHT slide animation for the QM Wavepacket presentation.
#
# Usage (from this directory):
#   julia --project=. run_all.jl
#
# All outputs go to  animations/output/  as .mp4 (or .png for static slides).
#
# SCREEN RECORDINGS (not rendered here — use the Pluto notebook directly):
#   Slide 1   – Motivation            → record the live notebook helix + barrier demo
#   Slides 5.1–5.6 – Six potentials   → record from the notebook for each potential setting
#   Slide 9   – Notebook layout        → screenshot or recording of the dual-panel UI
#   Slides 11.1–11.6 – Results         → same notebook recordings as 5.x, metrics card visible
#   Slide 12  – Custom potential        → record typing a custom expression into the field
#
# TEXT-ONLY SLIDES (no animation needed):
#   Slide 8   – Grid parameters table
#   Slide 13  – Conclusions / summary table

using Pkg
Pkg.activate(@__DIR__)

mkpath(joinpath(@__DIR__, "output"))

const SCRIPTS = [
    # Section 2: The Quantum State
    "s02_1_classical_quantum.jl",
    "s02_2_interference.jl",
    "s02_3_plane_wave.jl",
    "s02_4_curvature_norm.jl",
    "s02_5_stationary_states.jl",
    # Section 3: Gaussian Wavepacket
    "s03_1_min_uncertainty.jl",
    "s03_3_carrier_wave.jl",
    "s03_4_phase_group.jl",
    "s03_5_spreading.jl",
    # Section 4: Argand Helix
    "s04_1_complex_arrow.jl",
    "s04_2_helix_buildup.jl",
    "s04_3_tapered_helix.jl",
    "s04_4_helix_dict.jl",
    # Section 6: Crank-Nicolson numerics
    "s06_1_euler_fails.jl",
    "s06_2_unit_circle.jl",
    "s06_3_tridiagonal.jl",     # → PNG (static)
    "s06_4_phase_accuracy.jl",
    # Section 7: Absorbing layer profile
    "s07_absorbing.jl",
    # Section 10: Observable time traces
    "s10_observables.jl",
]

total = length(SCRIPTS)
for (i, script) in enumerate(SCRIPTS)
    println("\n[$i/$total]  $script ...")
    t0 = time()
    include(joinpath(@__DIR__, script))
    println("    done in $(round(time()-t0, digits=1)) s")
end

println("\n✓  All animations complete → $(joinpath(@__DIR__, "output"))\n")

println("Slide → animation file mapping:")
manifest = [
    "2.1  Classical vs Quantum"       => "s02_1_classical_quantum.mp4",
    "2.2  Why ψ is complex"           => "s02_2_interference.mp4",
    "2.3  de Broglie → TDSE"          => "s02_3_plane_wave.mp4",
    "2.4  Meaning of terms"           => "s02_4_curvature_norm.mp4",
    "2.5  Stationary states"          => "s02_5_stationary_states.mp4",
    "3.1  Minimum uncertainty"        => "s03_1_min_uncertainty.mp4",
    "3.3  Carrier wave"               => "s03_3_carrier_wave.mp4",
    "3.4  Phase vs group velocity"    => "s03_4_phase_group.mp4",
    "3.5  Spreading"                  => "s03_5_spreading.mp4",
    "4.1  Complex arrow"              => "s04_1_complex_arrow.mp4",
    "4.2  Helix buildup"              => "s04_2_helix_buildup.mp4",
    "4.3  Tapered helix"              => "s04_3_tapered_helix.mp4",
    "4.4  Helix dictionary"           => "s04_4_helix_dict.mp4",
    "6.1  Euler fails"                => "s06_1_euler_fails.mp4",
    "6.2  CN unit circle"             => "s06_2_unit_circle.mp4",
    "6.3  Tridiagonal (static)"       => "s06_3_tridiagonal.png",
    "6.4  Phase accuracy"             => "s06_4_phase_accuracy.mp4",
    "7    Absorbing layer profile"    => "s07_absorbing.mp4",
    "10   Observable time traces"     => "s10_observables.mp4",
]
for (slide, file) in manifest
    status = isfile(joinpath(@__DIR__, "output", file)) ? "✓" : "✗ MISSING"
    println("  $status  $slide  →  $file")
end

println("""
\nSCREEN RECORDINGS (record from the Pluto notebook — jl/qm_wavepacket/app.pluto.jl):
  Slide 1      → V=barrier A=2.5, helix + tunneling (the flagship demo)
  Slide 5.1    → V=free,    watch helix widen and packet spread
  Slide 5.2    → V=barrier, A=2.5 (tunneling) and A=0.8 (impedance mismatch)
  Slide 5.3    → V=well,    helix winds tighter inside well
  Slide 5.4    → V=parabola, watch ⟨x⟩ oscillate classically (multiple periods)
  Slide 5.5    → V=double,  k₀=3 (T≈1) vs k₀=4 (T≈0) — two separate recordings
  Slide 5.6    → V=step,    A=0.75, partial reflection above barrier
  Slide 9      → Screenshot of the full dual-panel notebook UI
  Slides 11.*  → Same as 5.*, but scroll to show the live metrics card
  Slide 12     → Type  3*E * exp(-2e19*(x - L/2)^2)  into the custom field

TEXT-ONLY SLIDES (no animation):
  Slide 8   → Grid parameters table — format in Keynote
  Slide 13  → Conclusions / summary table — format in Keynote
""")
