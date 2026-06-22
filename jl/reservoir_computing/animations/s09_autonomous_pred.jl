# s09_autonomous_pred.jl: Autonomous prediction: ESN free-runs on attractor
# Slide: "Methods: Autonomous Prediction"
#
# Left:  true x(t) (blue) vs ESN autonomous prediction x̂(t) (amber)
# Right: normalized error ‖û-u‖/‖u‖_rms vs t/T_λ, PH = 1.96 LT marked

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))
using Statistics

set_theme!(slide_theme())

N_FRAMES = 9 * FPS

# ── Train ESN ─────────────────────────────────────────────────────────────────
T_TRAIN = 50.0
_, traj_all = generate_lorenz(T_TRAIN + 10.0; u0 = [1.0, 0.0, 0.0], warmup = 5.0)
N_ALL = size(traj_all, 1)
N_TRAIN_STEPS = round(Int, T_TRAIN / DT) + 1
traj_train_data = traj_all[1:N_TRAIN_STEPS, :]

esn = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.05,
                                   spectral_radius = 0.95,
                                   input_scaling = 0.5,
                                   ridge_alpha = 1e-6, seed = 42))
train_esn!(esn, traj_train_data, 100)

# ── Autonomous prediction ─────────────────────────────────────────────────────
T_TEST = 8.0  # 8 time units of prediction ≈ 7.2 LT
N_TEST = round(Int, T_TEST / DT)
preds_auto = predict_esn(esn, N_TEST)

# True trajectory (continuation from end of training)
true_test = traj_all[N_TRAIN_STEPS+1:N_TRAIN_STEPS+N_TEST, :]
# clamp to available data
N_USE = min(N_TEST, size(true_test, 1), size(preds_auto, 1))
true_test  = true_test[1:N_USE, :]
preds_auto = preds_auto[1:N_USE, :]

ts_test = collect(range(0.0, DT * (N_USE - 1); length = N_USE))

# Normalized error
u_rms = sqrt(mean(true_test.^2))
norm_err = [norm(preds_auto[i, :] .- true_test[i, :]) / u_rms for i in 1:N_USE]
ts_lya = ts_test ./ T_lyapunov  # in Lyapunov times

TAU = 0.5
# Compute PH from the actual error curve
ph_idx = findfirst(x -> x > TAU, norm_err)
PH = ph_idx === nothing ? last(ts_lya) : ts_lya[ph_idx]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_x = Axis(fig[1, 1];
    title  = L"x(t)\text{: true vs ESN autonomous prediction}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"x(t)",
    limits = (0.0, last(ts_test), -25.0, 25.0))

ax_e = Axis(fig[1, 2];
    title  = L"\text{Normalized error vs }t/T_\lambda",
    xlabel = L"t/T_\lambda\;\text{(Lyapunov times)}",
    ylabel = L"|u_{pred} - u| / |u|_{rms}",
    yscale = log10,
    limits = (0.0, last(ts_lya), 1e-3, 10.0))

# Threshold line
hlines!(ax_e, [TAU];
    color = (C_DIV, 0.6), linewidth = 1.5, linestyle = :dash)
text!(ax_e, 0.1, TAU * 1.5;
    text = L"\tau = 0.5", color = C_DIV, fontsize = 17)

# PH marker (computed from data)
vlines!(ax_e, [PH];
    color = (C_ATT, 0.7), linewidth = 1.5, linestyle = :dash)
text!(ax_e, PH + 0.05, 3.0;
    text = "PH = $(round(PH; digits=2)) T_λ", color = C_ATT, fontsize = 17)

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)
ts_obs   = @lift ts_test[1:$n_show]
ts_lya_obs = @lift ts_lya[1:$n_show]

lines!(ax_x, ts_obs, @lift(true_test[1:$n_show, 1]);
    color = C_TRUE, linewidth = 2.0, label = L"x_{true}(t)")
lines!(ax_x, ts_obs, @lift(preds_auto[1:$n_show, 1]);
    color = C_PRED, linewidth = 2.0, linestyle = :dash, label = L"x_{pred}(t)")
axislegend(ax_x; position = :rt, labelsize = 17)

norm_safe = max.(norm_err, 1e-4)
lines!(ax_e, ts_lya_obs, @lift(norm_safe[1:$n_show]);
    color = C_PRED, linewidth = 2.5)

t_label = @lift "t = $(round(ts_test[$n_show]; digits=2)) time units  ($(round(ts_lya[$n_show]; digits=1)) T_λ)"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

record(fig, out("s09_autonomous_pred.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_USE))
end
println("✓  s09_autonomous_pred.mp4")
