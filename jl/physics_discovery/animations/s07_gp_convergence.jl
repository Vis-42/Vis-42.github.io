# s07_gp_convergence.jl: GP R² on all 5 Feynman equations
# Slide: "8. Symbolic Regression II: Feynman benchmark"
#
# Static: bar chart of R² per equation + convergence curves for each

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 4 * FPS

println("Running GP on all 5 Feynman equations ...")
results = []
for idx in 1:5
    X, y, _ = feynman_data(idx; N=300, noise_σ=0.0, seed=42)
    nv = size(X, 2)
    b, r2, nn, bh, _ = gp_run(X, y, nv; n_gen=60, pop_size=100, seed=42)
    eq = FEYNMAN_EQNS[idx]
    push!(results, (name=eq[1], formula=eq[2], r2=r2, n_nodes=nn,
                    best_str=tree_string(b, eq[3]), hist=bh))
    println("  $(eq[1]): R²=$(round(r2;digits=4)), nodes=$(nn), found=$(tree_string(b,eq[3]))")
end

fig = Figure(size = RES)
Label(fig[0,:], L"\text{GP on Feynman Benchmark: 5 Physical Equations}";
    color=C_TXT, fontsize=22)

# R² bar chart
ax1 = Axis(fig[1,1:2];
    title  = L"R^2 \text{ at generation 60 (pop=100)}",
    ylabel = L"R^2",
    xticks = (1:5, [r.name for r in results]),
    xticklabelrotation = 0.4,
    xticklabelsize = 14,
    limits = (nothing, (0.0, 1.05)))
bar_colors = [r.r2 > 0.99 ? C_RECON : (r.r2 > 0.95 ? C_LIB : C_ERR) for r in results]
barplot!(ax1, 1:5, [r.r2 for r in results]; color=bar_colors)
hlines!(ax1, [0.99]; color=C_THEORY, linestyle=:dash, linewidth=1.5,
    label=L"R^2 = 0.99")
axislegend(ax1; position=:rb, labelsize=14)

# Convergence curves
ax2 = Axis(fig[1,3];
    title  = L"\text{Best fitness vs generation}",
    xlabel = L"\text{generation}",
    ylabel = L"\mathcal{F}",
    yscale = log10)
colors_eq = [C_TRAJ, C_RECON, C_LIB, C_CONS, C_THEORY]
max_gen = maximum(length(r.hist) for r in results)
n_gen = Observable(1)
for (i, r) in enumerate(results)
    ng = length(r.hist)
    lines!(ax2, @lift(1:min($n_gen, ng)), @lift(r.hist[1:min($n_gen, ng)]);
        color=colors_eq[i], linewidth=2.0, label=r.name[1:min(12,end)])
end
axislegend(ax2; position=:rt, labelsize=12, nbanks=1)

# Discovered expression display
ax3 = Axis(fig[2,1:3]; title=L"\text{Discovered formulas}",
    aspect = DataAspect())
hidedecorations!(ax3); hidespines!(ax3)
for (i, r) in enumerate(results)
    y_pos = 1.0 - (i-1)/6.0
    text!(ax3, 0.02, y_pos;
        text="$(r.name):  found = $(r.best_str)",
        color=colors_eq[i], fontsize=14, align=(:left,:center))
    text!(ax3, 0.52, y_pos;
        text="true = $(r.formula)   R²=$(round(r.r2;digits=4))",
        color=C_DIM, fontsize=14, align=(:left,:center))
end

Label(fig[3,:],
    L"\text{Simple physics formulas (depth}\leq 3\text{) are recovered in 20–60 generations.}\;\text{Parsimony keeps trees compact.}";
    color=C_DIM, fontsize=14)

record(fig, out("s07_gp_convergence.mp4"), 1:N_FRAMES; framerate=FPS) do f
    n_gen[] = max(1, round(Int, (f / N_FRAMES) * max_gen))
end
println("✓  s07_gp_convergence.mp4")
