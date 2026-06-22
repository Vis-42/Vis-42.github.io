#!/usr/bin/env julia
# run_all.jl: Render all 7 animations for the Active Matter presentation.
#
# Usage (from this directory):
#   julia --project=. run_all.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

mkpath(joinpath(@__DIR__, "output"))

const SCRIPTS = [
    "s01_motivation.jl",
    "s02_vicsek_phases.jl",
    "s03_order_parameter.jl",
    "s04_bifurcation.jl",
    "s05_runtumble_path.jl",
    "s06_msd.jl",
    "s07_deff_comparison.jl",
]

total = length(SCRIPTS)
for (i, script) in enumerate(SCRIPTS)
    println("\n[$i/$total]  $script ...")
    t0 = time()
    include(joinpath(@__DIR__, script))
    println("    done in $(round(time()-t0, digits=1)) s")
end

println("\n✓  All animations complete → $(joinpath(@__DIR__, "output"))\n")

println("=" ^ 72)
println("SLIDE → ANIMATION MAPPING  (7 animated + 4 text-only)")
println("=" ^ 72)

anim_manifest = [
    "§1 Hook: flock vs gas, same rule"        => "s02_vicsek_phases.mp4",
    "§2 Active vs passive matter"             => "s01_motivation.mp4",
    "§4 Order parameter φ(t) + bifurcation"   => "s03_order_parameter.mp4",
    "§5 Flocking transition + susceptibility" => "s04_bifurcation.mp4",
    "§6 Run-and-tumble trajectory"            => "s05_runtumble_path.mp4",
    "§7 MSD ballistic→diffusive crossover"    => "s06_msd.mp4",
    "§8 Effective diffusion: theory vs sim"   => "s07_deff_comparison.mp4",
]

println("\n[ANIMATION] Pre-rendered outputs:")
for (slide, file) in anim_manifest
    status = isfile(joinpath(@__DIR__, "output", file)) ? "✓" : "✗ MISSING"
    println("  $status  $slide  →  $file")
end

println("""

[TEXT ONLY] No animation:
  Vicsek Model equations (Theory I)
  Theory IV: Exact MSD derivation
  The Interactive Notebook
  Conclusions and Extensions
""")
