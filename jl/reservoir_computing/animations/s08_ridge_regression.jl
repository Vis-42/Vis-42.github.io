# s08_ridge_regression.jl: Ridge regression readout training
# Slide: "Methods: Ridge Regression"
#
# Left:  training RMSE vs log₁₀(λ) showing regularization tradeoff
# Right: true x(t) vs one-step-ahead prediction during training

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))
using Statistics

set_theme!(slide_theme())

N_FRAMES = 8 * FPS

# ── Generate training data ────────────────────────────────────────────────────
T_TRAIN_SHOW = 10.0   # 10 time units to show
_, traj_train = generate_lorenz(T_TRAIN_SHOW; u0 = [1.0, 0.0, 0.0], warmup = 10.0)
N_TR = size(traj_train, 1)

# ── Compute RMSE vs ridge alpha curve ─────────────────────────────────────────
log_alphas = range(-8, 2; length = 40)
alphas_vec = 10.0 .^ log_alphas

function training_rmse(alpha)
    esn = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                       spectral_radius = 0.7,
                                       input_scaling = 0.5,
                                       ridge_alpha = alpha, seed = 42))
    washout = 100
    train_esn!(esn, traj_train, washout)

    # One-step-ahead predictions on training set (after washout)
    esn2 = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                        spectral_radius = 0.7,
                                        input_scaling = 0.5,
                                        ridge_alpha = alpha, seed = 42))
    esn2.W_out = esn.W_out
    esn2.state .= 0.0
    N_res = 50
    d = 3
    preds_tr = zeros(N_TR - washout - 1, d)
    for t in 1:N_TR - 1
        esn_step!(esn2, traj_train[t, :])
        if t > washout
            ext = vcat(esn2.state, traj_train[t, :])
            preds_tr[t - washout, :] .= esn2.W_out * ext
        end
    end
    targets_tr = traj_train[washout+2:end, :]
    sqrt(mean((preds_tr .- targets_tr).^2))
end

rmse_curve = [training_rmse(a) for a in alphas_vec]

# ── Best model one-step predictions ──────────────────────────────────────────
best_alpha = 1e-6
esn_best = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                        spectral_radius = 0.7,
                                        input_scaling = 0.5,
                                        ridge_alpha = best_alpha, seed = 42))
washout = 100
train_esn!(esn_best, traj_train, washout)

esn_pred = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                        spectral_radius = 0.7,
                                        input_scaling = 0.5,
                                        ridge_alpha = best_alpha, seed = 42))
esn_pred.W_out = esn_best.W_out
esn_pred.state .= 0.0

T_pred_show = N_TR - washout - 1
preds_1step = zeros(T_pred_show, 3)
for t in 1:N_TR - 1
    esn_step!(esn_pred, traj_train[t, :])
    if t > washout
        ext = vcat(esn_pred.state, traj_train[t, :])
        preds_1step[t - washout, :] .= esn_pred.W_out * ext
    end
end
ts_pred_show = collect(range(0.0, T_TRAIN_SHOW * (T_pred_show / N_TR); length = T_pred_show))
true_x = traj_train[washout+2:end, 1]
pred_x = preds_1step[:, 1]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_r = Axis(fig[1, 1];
    title  = L"\text{Training RMSE vs regularization } \lambda",
    xlabel = L"\log_{10}(\lambda)",
    ylabel = L"\text{RMSE}",
    yscale = log10,
    limits = (-8.2, 2.2, 1e-2, 1e5))

ax_p = Axis(fig[1, 2];
    title  = L"x(t)\text{: true vs one-step-ahead prediction}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"x(t)",
    limits = (0.0, last(ts_pred_show), -25.0, 25.0))

# RMSE curve (static: computed upfront)
lines!(ax_r, collect(log_alphas), rmse_curve;
    color = C_TRUE, linewidth = 2.5)
vlines!(ax_r, [log10(best_alpha)];
    color = (C_ATT, 0.7), linewidth = 1.5, linestyle = :dash)
text!(ax_r, log10(best_alpha) + 0.2, 0.05;
    text = L"\lambda = 10^{-6}", color = C_ATT, fontsize = 17)
scatter!(ax_r, [log10(best_alpha)], [training_rmse(best_alpha)];
    color = C_ATT, markersize = 12)

# Prediction traces: animate progressively
n_show = Observable(2)
ts_obs   = @lift ts_pred_show[1:$n_show]
true_obs = @lift true_x[1:$n_show]
pred_obs = @lift pred_x[1:$n_show]

lines!(ax_p, ts_obs, true_obs; color = C_TRUE, linewidth = 2.0, label = L"x_{true}(t)")
lines!(ax_p, ts_obs, pred_obs; color = C_PRED, linewidth = 1.8, linestyle = :dash,
    label = L"\hat{x}_{1-step}(t)")
axislegend(ax_p; position = :rt, labelsize = 17)

t_label = @lift "t = $(round(ts_pred_show[$n_show]; digits=2)) time units"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

N_pts = length(ts_pred_show)
record(fig, out("s08_ridge_regression.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_pts))
end
println("✓  s08_ridge_regression.mp4")
