#!/usr/bin/env julia
# run_all.jl: Render all 11 slide animations for the Reservoir Computing presentation.
#
# Usage (from this directory):
#   julia --project=. run_all.jl
#
# All outputs go to  animations/output/  as .mp4

using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

mkpath(joinpath(@__DIR__, "output"))

const SCRIPTS = [
    "s01_motivation.jl",
    "s02_lyapunov.jl",
    "s03_lorenz_attractor.jl",
    "s04_echo_state.jl",
    "s05_spectral_radius.jl",
    "s06_reservoir_init.jl",
    "s07_reservoir_dynamics.jl",
    "s08_ridge_regression.jl",
    "s09_autonomous_pred.jl",
    "s10_phase_diagram.jl",
    "s11_trajectory_recon.jl",
]

total = length(SCRIPTS)
for (i, script) in enumerate(SCRIPTS)
    println("\n[$i/$total]  $script ...")
    t0 = time()
    include(joinpath(@__DIR__, script))
    println("    done in $(round(time()-t0, digits=1)) s")
end

println("\n✓  All animations complete → $(joinpath(@__DIR__, "output"))\n")

# ── Complete slide ↔ animation mapping ───────────────────────────────────────
println("=" ^ 72)
println("COMPLETE SLIDE → ANIMATION MAPPING  (11 animated + 4 text-only)")
println("=" ^ 72)

anim_manifest = [
    "Motivation"                      => "s01_motivation.mp4",
    "Theory: Lyapunov Exponents"      => "s02_lyapunov.mp4",
    "Theory: Lorenz-63 System"        => "s03_lorenz_attractor.mp4",
    "Theory: Echo State Property"     => "s04_echo_state.mp4",
    "Theory: Spectral Radius"         => "s05_spectral_radius.mp4",
    "Architecture: Reservoir Init"    => "s06_reservoir_init.mp4",
    "Architecture: Reservoir Dynamics"=> "s07_reservoir_dynamics.mp4",
    "Methods: Ridge Regression"       => "s08_ridge_regression.mp4",
    "Methods: Autonomous Prediction"  => "s09_autonomous_pred.mp4",
    "Results: Phase Diagram"          => "s10_phase_diagram.mp4",
    "Results: Trajectory Recon"       => "s11_trajectory_recon.mp4",
]

println("\n[ANIMATION] Pre-rendered Julia outputs:")
for (slide, file) in anim_manifest
    status = isfile(joinpath(@__DIR__, "output", file)) ? "✓" : "✗ MISSING"
    println("  $status  $slide  →  $file")
end

println("""

[TEXT ONLY] No animation:
  Outline
  Methods: Hyperparameter Sweep
  Conclusions
  Limitations
  Natural Extensions
""")
