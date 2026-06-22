module GradientCalc

using ..PendulumSim

export trajectory_loss, finite_diff_gradient

# Compute MSE loss between simulated trajectory and observed data
function trajectory_loss(params_vec::Vector{Float64},
                         observed::Matrix{Float64},
                         u0::Vector{Float64},
                         tspan::Tuple{Float64,Float64},
                         dt::Float64)::Float64
    m1, m2, L1, L2 = clamp.(params_vec, 0.05, 20.0)
    p = PendulumParams(m1=m1, m2=m2, L1=L1, L2=L2, g=9.81)
    try
        _, traj = PendulumSim.simulate(p, u0, tspan, dt)
        n  = min(size(traj,1), size(observed,1))
        mean_sq = 0.0
        for i in 1:n, j in 1:4
            d = traj[i,j] - observed[i,j]
            mean_sq += d^2
        end
        mean_sq / (n * 4)
    catch
        1e10
    end
end

function mean_sq_fallback(traj, observed)
    n = min(size(traj,1), size(observed,1))
    s = 0.0
    for i in 1:n, j in 1:4
        s += (traj[i,j] - observed[i,j])^2
    end
    s / (n*4)
end

# Finite-difference gradient w.r.t. params_vec
function finite_diff_gradient(params_vec::Vector{Float64},
                               observed::Matrix{Float64},
                               u0::Vector{Float64},
                               tspan::Tuple{Float64,Float64},
                               dt::Float64;
                               ε::Float64=1e-5)::Vector{Float64}
    f0   = trajectory_loss(params_vec, observed, u0, tspan, dt)
    grad = similar(params_vec)
    for i in eachindex(params_vec)
        pv_hi = copy(params_vec); pv_hi[i] += ε
        pv_lo = copy(params_vec); pv_lo[i] -= ε
        f_hi = trajectory_loss(pv_hi, observed, u0, tspan, dt)
        f_lo = trajectory_loss(pv_lo, observed, u0, tspan, dt)
        grad[i] = (f_hi - f_lo) / (2ε)
    end
    grad
end

end
