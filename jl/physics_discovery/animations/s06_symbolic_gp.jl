# s06_symbolic_gp.jl: Symbolic GP: evolving expression trees
# Slide: "7. Symbolic Regression I: Expression trees"
#
# Animated: GP fitness curve + best expression string updating each generation

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 8 * FPS

# Run GP on kinetic energy E = ½mv²
println("Running GP on kinetic energy ...")
X, y, y_clean = feynman_data(1; N=300, noise_σ=0.0, seed=42)
best, r2, n_nodes, best_hist, pop_hist = gp_run(X, y, 2; n_gen=80, pop_size=120, seed=42)
eq_name, eq_formula, eq_vars = FEYNMAN_EQNS[1][1], FEYNMAN_EQNS[1][2], FEYNMAN_EQNS[1][3]

best_str = tree_string(best, eq_vars)

n_gens_total = length(best_hist)
y_pred = eval_tree(best, X)
xlims_sc = extrema(y_clean)

fig = Figure(size = RES)
Label(fig[0,:], L"\text{Symbolic Regression via Genetic Programming: Kinetic Energy }E=\frac{1}{2}mv^2";
    color=C_TXT, fontsize=22)

# Fitness curve: revealed progressively
ax1 = Axis(fig[1,1:2];
    title  = L"\text{Best fitness vs generation}",
    xlabel = L"\text{generation}",
    ylabel = L"\mathcal{F} = \text{NMSE} + c_p |T|",
    yscale = log10,
    limits = (1, n_gens_total, nothing, nothing))

gen_obs = Observable(1)
lines!(ax1, @lift(1:$gen_obs), @lift(best_hist[1:$gen_obs]);
    color=C_LIB, linewidth=3.0)
# Final star: appears only when all generations are revealed
scatter!(ax1, @lift($gen_obs >= n_gens_total ? [Float64(n_gens_total)] : Float64[]),
              @lift($gen_obs >= n_gens_total ? [best_hist[end]] : Float64[]);
    color=C_RECON, markersize=14, marker=:star5)
text!(ax1, n_gens_total*0.6, best_hist[end]*5.0;
    text=@lift($gen_obs >= n_gens_total ? "R² = $(round(r2;digits=4))" : ""),
    color=C_RECON, fontsize=18)

# Final prediction scatter: fades in during last 30% of animation
ax2 = Axis(fig[1,3];
    title  = L"\hat{y} \text{ vs true } y \text{ at final generation}",
    xlabel = L"y_{\text{true}}",
    ylabel = L"\hat{y} = f(m, v)")
n_scatter = @lift round(Int, max(0, ($gen_obs / n_gens_total - 0.7) / 0.3 * length(y_clean)))
scatter!(ax2, @lift(y_clean[1:$n_scatter]), @lift(y_pred[1:$n_scatter]);
    color=(C_LIB,0.6), markersize=5)
lines!(ax2, collect(xlims_sc), collect(xlims_sc); color=C_THEORY, linestyle=:dash, linewidth=1.5)
text!(ax2, xlims_sc[1]+0.05*(xlims_sc[2]-xlims_sc[1]),
           xlims_sc[2]-0.1*(xlims_sc[2]-xlims_sc[1]);
    text=@lift($gen_obs >= n_gens_total ? "R² = $(round(r2; digits=4))" : ""),
    color=C_THEORY, fontsize=16)

# Tree string display: appears in final quarter
ax3 = Axis(fig[2,1:3];
    title  = L"\text{Discovered expression}",
    aspect = DataAspect())
hidedecorations!(ax3); hidespines!(ax3)
text!(ax3, 0.5, 0.7;
    text=@lift($gen_obs >= round(Int, 0.75*n_gens_total) ? "Found: " * best_str : ""),
    color=C_RECON, fontsize=18, align=(:center,:center))
text!(ax3, 0.5, 0.4; text="True:  " * eq_formula,
    color=C_THEORY, fontsize=18, align=(:center,:center))
text!(ax3, 0.5, 0.1;
    text=@lift($gen_obs >= n_gens_total ? "Nodes: $(n_nodes)   Generations: $(n_gens_total)" : "gen $($gen_obs) / $(n_gens_total)"),
    color=C_DIM, fontsize=16, align=(:center,:center))

Label(fig[3,:],
    L"\text{GP searches expression tree space with tournament selection + crossover + mutation.}\;\text{Parsimony (}c_p\cdot|T|\text{) prevents bloat.}";
    color=C_DIM, fontsize=14)

record(fig, out("s06_symbolic_gp.mp4"), 1:N_FRAMES; framerate=FPS) do frame
    gen_obs[] = max(1, round(Int, 1 + (n_gens_total-1)*(frame-1)/(N_FRAMES-1)))
end
println("✓  s06_symbolic_gp.mp4")
