# s08_comparison.jl: Three methods on the same Kepler data
# Slide: "9. Method Comparison"
#
# Three panels: SINDy phase portrait, kernel H(t), GP prediction scatter

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 6 * FPS
TRAIL    = 60

println("Running all three methods on Kepler orbit ...")

# Kepler data
ts, states, H_true = kepler_data(; ecc=0.3, n=500)
dt = ts[2] - ts[1]

# 1. SINDy on [x, y, vx, vy]
dX   = finite_diff(states, dt)
Θ, names = build_library(states, 2, false)
xi   = stlsq(Θ, dX, 0.05)
n_active = count(any(xi .!= 0; dims=2))
rmse = sqrt(mean((Θ*xi .- dX).^2))

# Reconstruct first two coordinates (x, y)
u0 = states[1, :]
recon = sindy_reconstruct(xi, u0, size(states,1), dt, 2, false)
nv    = sum(.!isnan.(recon[:,1]))

# 2. Kernel conservation law
println("  kernel discovery ...")
H_k = discover_invariant(states, dt; γ=0.5, α=1e-5)
H_k_n  = (H_k .- mean(H_k)) ./ (std(H_k)+1e-12)
H_t_n  = (H_true .- mean(H_true)) ./ (std(H_true)+1e-12)
ρ_k    = cor(H_true, H_k)

# 3. Symbolic GP on kinetic energy (reuse precomputed)
println("  GP on kinetic energy ...")
X_ke, y_ke, _ = feynman_data(1; N=300, seed=42)
best, r2_gp, _, _, _ = gp_run(X_ke, y_ke, 2; n_gen=60, pop_size=100, seed=42)
best_str = tree_string(best, FEYNMAN_EQNS[1][3])

fig = Figure(size = RES)
Label(fig[0,:], L"\text{Three Methods: Same Data, Different Answers}";
    color=C_TXT, fontsize=22)

# SINDy: animated phase portrait
step = Observable(1)

ax1 = Axis(fig[1,1];
    title  = L"\text{SINDy: Kepler orbit reconstruction (}x,y\text{)}",
    xlabel = L"x", ylabel = L"y")
lines!(ax1, states[:,1], states[:,2]; color=(C_DIM,0.25), linewidth=0.8)
nv > 10 && lines!(ax1, recon[1:nv,1], recon[1:nv,2]; color=(C_RECON,0.3), linewidth=0.8)
trl_x = @lift states[max(1,$step-TRAIL):min($step,size(states,1)), 1]
trl_y = @lift states[max(1,$step-TRAIL):min($step,size(states,1)), 2]
lines!(ax1, trl_x, trl_y; color=C_TRAJ, linewidth=2.5)
dot_x = @lift [states[min($step,size(states,1)),1]]
dot_y = @lift [states[min($step,size(states,1)),2]]
scatter!(ax1, dot_x, dot_y; color=C_LIB, markersize=14)
scatter!(ax1, [0.0], [0.0]; color=C_THEORY, markersize=18, marker=:star5)
text!(ax1, -0.5, 1.4; text="$(n_active) active\nRMSE=$(round(rmse;sigdigits=2))",
    color=C_RECON, fontsize=15)

# Kernel: H(t) trace
ax2 = Axis(fig[1,2];
    title  = "Kernel: H_kernel(t) vs true energy (ρ = $(round(ρ_k; digits=3)))",
    xlabel = L"t", ylabel = L"H \text{ (normalized)}")
lines!(ax2, ts, H_t_n;  color=C_THEORY, linewidth=2.0, linestyle=:dash, label=L"H_{\text{true}}")
lines!(ax2, ts, H_k_n;  color=C_CONS,   linewidth=2.0, label=L"H_{\text{kernel}}")
cursor_x = @lift [ts[min($step,length(ts))]]
cursor_y = @lift [H_k_n[min($step,length(H_k_n))]]
scatter!(ax2, cursor_x, cursor_y; color=C_LIB, markersize=12)
axislegend(ax2; position=:rt, labelsize=15)

# GP: prediction scatter
ax3 = Axis(fig[1,3];
    title  = "GP (KE): R² = $(round(r2_gp; digits=4))",
    xlabel = L"E_{\text{true}}", ylabel = L"\hat{E}_{\text{GP}}")
y_pred_gp = eval_tree(best, X_ke)
scatter!(ax3, y_ke, y_pred_gp; color=(C_LIB,0.4), markersize=5)
xlims = extrema(y_ke)
lines!(ax3, collect(xlims), collect(xlims); color=C_THEORY, linestyle=:dash)
text!(ax3, xlims[1]+0.1*(xlims[2]-xlims[1]),
           xlims[2]-0.15*(xlims[2]-xlims[1]);
    text=best_str, color=C_LIB, fontsize=14)

# Summary comparison table
ax4 = Axis(fig[2,:]; title=L"\text{Comparison summary}", aspect=DataAspect())
hidedecorations!(ax4); hidespines!(ax4)
rows = [
    ("Method",       "Output",              "This run",                  "Limitation"),
    ("SINDy",        "Identified ODE",      "$(n_active) terms, RMSE=$(round(rmse;sigdigits=2))", "Needs derivative accuracy"),
    ("Kernel ∇H",    "Conserved function",  "ρ=$(round(ρ_k;digits=3)) vs true H",  "Abstract, needs poly projection"),
    ("Symbolic GP",  "Closed-form expr",    "$(best_str)  R²=$(round(r2_gp;digits=3))",    "Slow, stochastic"),
]
cols_x = [0.0, 0.25, 0.52, 0.75]
for (ri, row) in enumerate(rows)
    y_pos = 1.0 - (ri-1)*0.3
    for (ci, cell) in enumerate(row)
        clr = ri == 1 ? C_TXT : (ci == 3 ? C_RECON : C_DIM)
        text!(ax4, cols_x[ci], y_pos; text=cell, color=clr, fontsize=14, align=(:left,:center))
    end
end

record(fig, out("s08_comparison.mp4"), 1:N_FRAMES; framerate=FPS) do frame
    step[] = round(Int, 1 + (size(states,1)-1)*(frame-1)/(N_FRAMES-1))
end
println("✓  s08_comparison.mp4")
