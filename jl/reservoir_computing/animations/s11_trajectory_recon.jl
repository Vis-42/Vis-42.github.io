# s11_trajectory_recon.jl: 3D trajectory reconstruction
# Slide: "Results: Trajectory Reconstruction"
#
# 3D plot: true Lorenz trajectory (blue, first 15 time units) and
# ESN autonomous trajectory (amber): they agree for ~2LT then diverge
# but both remain on the attractor

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 10 * FPS

# ── Train ESN ─────────────────────────────────────────────────────────────────
T_TRAIN_RECON = 50.0
_, traj_all_r = generate_lorenz(T_TRAIN_RECON + 15.0; u0 = [1.0, 0.0, 0.0], warmup = 5.0)
N_TRAIN_R = round(Int, T_TRAIN_RECON / DT) + 1
traj_train_r = traj_all_r[1:N_TRAIN_R, :]

esn_r = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.05,
                                     spectral_radius = 0.95,
                                     input_scaling = 0.5,
                                     ridge_alpha = 1e-6, seed = 42))
train_esn!(esn_r, traj_train_r, 100)

# ── Autonomous prediction ─────────────────────────────────────────────────────
T_SHOW_RECON = 15.0   # time units to display
N_SHOW_R = round(Int, T_SHOW_RECON / DT)

preds_r = predict_esn(esn_r, N_SHOW_R)
N_AVAIL = min(N_SHOW_R, size(preds_r, 1), size(traj_all_r, 1) - N_TRAIN_R)
true_r  = traj_all_r[N_TRAIN_R+1:N_TRAIN_R+N_AVAIL, :]
preds_r = preds_r[1:N_AVAIL, :]

ts_r = collect(range(0.0, DT * (N_AVAIL - 1); length = N_AVAIL))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax3 = Axis3(fig[1, 1];
    title  = L"\text{Lorenz attractor: true (blue) vs ESN prediction (amber)}",
    xlabel = L"x",
    ylabel = L"y",
    zlabel = L"z")

xlims!(ax3, -25.0, 25.0)
ylims!(ax3, -35.0, 35.0)
zlims!(ax3,   0.0, 55.0)

# Legend box
elem_true = LineElement(color = C_TRUE,  linewidth = 2.5)
elem_pred = LineElement(color = C_PRED,  linewidth = 2.5)
Legend(fig[1, 2], [elem_true, elem_pred],
    [L"\text{True Lorenz}", L"\text{ESN prediction}"];
    labelsize = 15, framecolor = RGBf(0.22, 0.22, 0.28),
    backgroundcolor = RGBf(0.08, 0.08, 0.10))

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)

lines!(ax3,
    @lift(true_r[1:$n_show, 1]),
    @lift(true_r[1:$n_show, 2]),
    @lift(true_r[1:$n_show, 3]);
    color = C_TRUE, linewidth = 2.2)

lines!(ax3,
    @lift(preds_r[1:$n_show, 1]),
    @lift(preds_r[1:$n_show, 2]),
    @lift(preds_r[1:$n_show, 3]);
    color = C_PRED, linewidth = 2.2, linestyle = :dash)

t_label = @lift "t = $(round(ts_r[$n_show]; digits=1)) time units  ($(round(ts_r[$n_show]/T_lyapunov; digits=1)) T_λ)"
Label(fig[2, 1], t_label; color = C_DIM, fontsize = 14)

record(fig, out("s11_trajectory_recon.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_AVAIL))
    ax3.azimuth[] = 1.3 + frac * 0.8
end
println("✓  s11_trajectory_recon.mp4")
