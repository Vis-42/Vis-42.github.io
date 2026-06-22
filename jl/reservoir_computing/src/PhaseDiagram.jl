module PhaseDiagram

include("ChaoticSystems.jl")
include("ESN.jl")
using .ChaoticSystems
using .ESN
using Statistics

export PhasePoint, sweep_phase_diagram

struct PhasePoint
    sparsity::Float64
    spectral_radius::Float64
    prediction_horizon::Float64   # in Lyapunov times
    rmse::Float64
end

# Valid prediction time: the first step where the normalised forecast error
# exceeds `tol`, expressed in Lyapunov times (the standard horizon metric for
# chaotic forecasting). `dt_eff` is the per-step time (dt for flows, 1 for maps).
function prediction_horizon_val(true_traj::Matrix{Float64}, pred_traj::Matrix{Float64};
                                  tol::Float64 = 0.5, lya_time::Float64 = 1.0,
                                  dt_eff::Float64 = 0.02)
    n = min(size(true_traj, 1), size(pred_traj, 1))
    norm_scale = sqrt(mean(true_traj.^2)) + 1e-12
    for t in 1:n
        err = sqrt(sum((true_traj[t,:] .- pred_traj[t,:]).^2)) / norm_scale
        err > tol && return t * dt_eff / lya_time
    end
    n * dt_eff / lya_time
end

function sweep_phase_diagram(system::String, sparsity_range, spectral_range;
                              N_reservoir::Int = 50, T_train::Float64 = 50.0,
                              T_test::Float64 = 10.0, dt::Float64 = 0.02,
                              washout::Int = 100, tol::Float64 = 0.5)
    params = Dict{Symbol,Float64}()
    # Compute the Lyapunov time directly from the dynamics (self-consistent),
    # rather than hardcoding a literature constant.
    lya    = lyapunov_time(system, params; dt=dt)
    dt_eff = ChaoticSystems.effective_dt(system, dt)
    _, traj_full = generate_trajectory(system, params, T_train + T_test + dt*washout, dt)
    train_len = round(Int, T_train/dt)
    test_len  = round(Int, T_test/dt)

    results = PhasePoint[]
    for sp in sparsity_range
        for sr in spectral_range
            cfg = ESNConfig(N_reservoir=N_reservoir, sparsity=sp, spectral_radius=sr,
                            input_scaling=0.5, leak_rate=1.0, ridge_alpha=1e-6)
            esn = build_esn(cfg; n_input=size(traj_full, 2))
            train_data = traj_full[1:train_len+washout, :]
            train_esn!(esn, train_data, washout)
            pred = predict_esn(esn, test_len)
            true_test = traj_full[train_len+washout+1:train_len+washout+test_len, :]
            n_cmp = min(size(pred,1), size(true_test,1))
            rmse = sqrt(mean((pred[1:n_cmp,:] .- true_test[1:n_cmp,:]).^2))
            ph   = prediction_horizon_val(true_test[1:n_cmp,:], pred[1:n_cmp,:];
                                          tol=tol, lya_time=lya, dt_eff=dt_eff)
            push!(results, PhasePoint(sp, sr, ph, rmse))
        end
    end
    results
end

end
