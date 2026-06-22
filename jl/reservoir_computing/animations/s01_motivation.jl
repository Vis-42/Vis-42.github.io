# s01_motivation.jl: Lorenz sensitivity: two nearby trajectories diverge
# Slide: "Motivation"
#
# Left:  x(t) for two Lorenz trajectories starting ε=0.001 apart
# Right: |δ(t)| on log scale with slope = λ_max and Lyapunov time marked

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 9 * FPS   # 9 s

# ── Pre-compute trajectories ──────────────────────────────────────────────────
T_SIM = 8.0   # time units (≈7.2 Lyapunov times)
EPS0  = 0.001

ts, traj_A = generate_lorenz(T_SIM; u0 = [1.0, 0.0, 0.0], warmup = 5.0)
_, traj_B  = generate_lorenz(T_SIM; u0 = [1.0 + EPS0, 0.0, 0.0], warmup = 5.0)

xA = traj_A[:, 1]
xB = traj_B[:, 1]

# |δ(t)| = distance between the two trajectories
delta = [norm(traj_A[i, :] .- traj_B[i, :]) for i in eachindex(ts)]

# Theoretical exponential growth: ε₀ * exp(λ_max * t)
delta_theory = EPS0 .* exp.(λ_max .* ts)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_x = Axis(fig[1, 1];
    title  = L"x(t)\text{: two trajectories, }\varepsilon_0 = 0.001",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"x(t)",
    limits = (0.0, T_SIM, -25.0, 25.0))

ax_d = Axis(fig[1, 2];
    title  = L"|\delta(t)|\text{: separation (log scale)}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"|\delta(t)|",
    yscale = log10,
    limits = (0.0, T_SIM, 5e-4, 100.0))

# Static reference lines
lines!(ax_d, ts, delta_theory;
    color = (C_DIM, 0.55), linewidth = 1.5, linestyle = :dash,
    label = L"\varepsilon_0 \exp(\lambda_{\max} t)")
vlines!(ax_d, [T_lyapunov];
    color = (C_ATT, 0.6), linewidth = 1.5, linestyle = :dash)
text!(ax_d, T_lyapunov + 0.08, 5e-3;
    text = L"T_\lambda \approx 1.1", color = C_ATT, fontsize = 18)

# λ_max slope annotation
text!(ax_d, 1.0, 0.08;
    text = L"\text{slope} = \lambda_{\max} = 0.906", color = C_DIM, fontsize = 17)

# Legend
axislegend(ax_d; position = :rb, labelsize = 17)

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)

ts_obs    = @lift ts[1:$n_show]
xA_obs    = @lift xA[1:$n_show]
xB_obs    = @lift xB[1:$n_show]
delta_obs = @lift delta[1:$n_show]

lines!(ax_x, ts_obs, xA_obs; color = C_TRUE,  linewidth = 2.0, label = L"u_A(0)")
lines!(ax_x, ts_obs, xB_obs; color = C_PRED,  linewidth = 2.0, label = L"u_B(0) = u_A(0)+\varepsilon_0")
axislegend(ax_x; position = :rt, labelsize = 17)

lines!(ax_d, ts_obs, delta_obs; color = C_TRUE, linewidth = 2.5)

# Time label
t_label = @lift "t = $(round(ts[$n_show]; digits=2))  ($(round(ts[$n_show]/T_lyapunov; digits=1)) T_λ)"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

# ── Record ────────────────────────────────────────────────────────────────────
N_pts = length(ts)
record(fig, out("s01_motivation.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_pts))
end
println("✓  s01_motivation.mp4")
