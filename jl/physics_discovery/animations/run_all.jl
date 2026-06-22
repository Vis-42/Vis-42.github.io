# run_all.jl: generate all physics_discovery slides
# Run: julia --project=. run_all.jl

import Pkg
Pkg.instantiate()

output_dir = joinpath(@__DIR__, "output")
isdir(output_dir) || mkpath(output_dir)

scripts = [
    "s01_motivation.jl",
    "s02_sindy_library.jl",
    "s03_sindy_reconstruction.jl",
    "s04_conservation_phase.jl",
    "s05_kernel_invariant.jl",
    "s06_symbolic_gp.jl",
    "s07_gp_convergence.jl",
    "s08_comparison.jl",
]

for s in scripts
    path = joinpath(@__DIR__, s)
    println("\n─── $s ───")
    include(path)
end

println("\n✓ All physics_discovery animations complete → $(output_dir)")
