using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Statistics, Random

include(joinpath(@__DIR__, "src", "SimplicialComplex.jl"))
include(joinpath(@__DIR__, "src", "Percolation.jl"))

using .SimplicialComplex, .Percolation

mkpath(joinpath(@__DIR__, "outputs"))
outdir = joinpath(@__DIR__, "outputs")

println("Percolation on Simplicial Complexes")
println("=" ^ 60)

N = 200; p_edge = 0.05; n_samples = 8
p_range = 0.0:0.05:1.0

println("Generating clique complex (N=$N, p_edge=$p_edge) ...")
cx = generate_clique_complex(N, p_edge; dim_max=3)
println("  Edges: $(length(cx[:edges]))")
println("  Triangles: $(length(get(cx, :triangles, [])))")
println("  Tetrahedra: $(length(get(cx, :tetrahedra, [])))")

results = Dict()
for dim in 1:3
    name = ["edges", "triangles", "tetrahedra"][dim]
    simp = simplices_of_dim(cx, dim)
    if isempty(simp)
        println("No $(name) in complex, skipping dim=$dim")
        continue
    end
    println("Computing percolation (dim=$dim, $n_samples samples per p) ...")
    S   = [giant_component_size(cx, p; dimension=dim, n_samples=n_samples) for p in p_range]
    chi = susceptibility(cx, p_range; dimension=dim, n_samples=n_samples)
    pc_idx = argmax(chi)
    pc  = collect(p_range)[pc_idx]
    println("  p_c ≈ $(round(pc; digits=2)),  S_max=$(round(maximum(S); digits=3))")
    results[dim] = (S=S, chi=chi, pc=pc)
end

# CSV
open(joinpath(outdir, "rule_metrics.csv"), "w") do f
    println(f, "p,S_dim1,S_dim2,S_dim3,chi_dim1,chi_dim2,chi_dim3")
    for (i, p) in enumerate(p_range)
        s1 = get(get(results, 1, (S=[0.0],)), :S, [0.0])[i]
        s2 = get(get(results, 2, (S=[0.0],)), :S, [0.0])[i]
        s3 = get(get(results, 3, (S=[0.0],)), :S, [0.0])[i]
        c1 = get(get(results, 1, (chi=[0.0],)), :chi, [0.0])[i]
        c2 = get(get(results, 2, (chi=[0.0],)), :chi, [0.0])[i]
        c3 = get(get(results, 3, (chi=[0.0],)), :chi, [0.0])[i]
        println(f, "$p,$s1,$s2,$s3,$c1,$c2,$c3")
    end
end
println("Saved: outputs/rule_metrics.csv")

# Summary
open(joinpath(outdir, "summary.txt"), "w") do f
    println(f, "Percolation on Clique Complex (N=$N, p_edge=$p_edge)")
    for (dim, r) in sort(collect(results))
        println(f, "  dim=$dim: p_c=$(round(r.pc; digits=3)), S_max=$(round(maximum(r.S); digits=3))")
    end
end
println("Saved: outputs/summary.txt")

try
    using Plots
    p1 = Plots.plot(collect(p_range), get(results, 1, (S=zeros(length(p_range)),)).S;
        label="k=1 (pairwise)", lw=2, color=:green,
        xlabel="Bond probability p", ylabel="Giant component S(p)",
        title="Percolation on Simplicial Complex (N=$N)")
    haskey(results, 2) && Plots.plot!(p1, collect(p_range), results[2].S;
        label="k=2 (triangles)", lw=2, color=:violet)
    haskey(results, 3) && Plots.plot!(p1, collect(p_range), results[3].S;
        label="k=3 (tetrahedra)", lw=2, color=:orange)
    Plots.savefig(p1, joinpath(outdir, "ei_vs_rule.png"))
    println("Saved: outputs/ei_vs_rule.png")

    p2 = Plots.plot(collect(p_range), get(results, 1, (chi=zeros(length(p_range)),)).chi;
        label="k=1 χ", lw=2, color=:green,
        xlabel="Bond probability p", ylabel="Susceptibility χ(p)",
        title="Susceptibility — Peak at p_c")
    haskey(results, 2) && Plots.plot!(p2, collect(p_range), results[2].chi;
        label="k=2 χ", lw=2, color=:violet)
    haskey(results, 3) && Plots.plot!(p2, collect(p_range), results[3].chi;
        label="k=3 χ", lw=2, color=:orange)
    Plots.savefig(p2, joinpath(outdir, "class_delta_ei.png"))
    println("Saved: outputs/class_delta_ei.png")
catch e
    println("Plots not available: $e")
end

println("\nDone.")
