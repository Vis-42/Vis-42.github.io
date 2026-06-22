module Analysis

using Statistics

export prediction_horizon, save_summary

function prediction_horizon(true_traj::Matrix{Float64}, pred_traj::Matrix{Float64};
                              tol::Float64 = 0.5, dt::Float64 = 0.02, lya_time::Float64 = 1.0)
    n = min(size(true_traj, 1), size(pred_traj, 1))
    norm_scale = sqrt(mean(true_traj.^2)) + 1e-12
    for t in 1:n
        err = sqrt(sum((true_traj[t,:] .- pred_traj[t,:]).^2)) / norm_scale
        err > tol && return t * dt / lya_time
    end
    n * dt / lya_time
end

function save_summary(results, path::String)
    open(path, "w") do io
        println(io, "sparsity,spectral_radius,prediction_horizon_lya,rmse")
        for r in results
            println(io, "$(r.sparsity),$(r.spectral_radius),$(r.prediction_horizon),$(r.rmse)")
        end
    end
end

end
