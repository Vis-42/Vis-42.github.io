module ParameterInference

import Random
using ..PendulumSim, ..GradientCalc
using Statistics

export InferenceConfig, InferenceResult, infer_parameters

Base.@kwdef struct InferenceConfig
    lr::Float64          = 0.01
    n_steps::Int         = 100
    ε_fd::Float64        = 1e-5
    momentum::Float64    = 0.9
    verbose::Bool        = false
end

struct InferenceResult
    inferred_params::PendulumParams
    loss_history::Vector{Float64}
    param_history::Matrix{Float64}   # n_steps × 4
    final_loss::Float64
    param_error_pct::Vector{Float64}
end

function infer_parameters(observed::Matrix{Float64},
                           u0::Vector{Float64},
                           true_params::PendulumParams,
                           tspan::Tuple{Float64,Float64},
                           dt::Float64;
                           cfg::InferenceConfig=InferenceConfig(),
                           init_params::Vector{Float64}=Float64[])
    # Initial guess: perturb true params by ±30%
    if isempty(init_params)
        rng = Random.MersenneTwister(7)
        tv  = [true_params.m1, true_params.m2, true_params.L1, true_params.L2]
        init_params = tv .* (1.0 .+ 0.3 .* Random.randn(rng, 4))
        init_params = clamp.(init_params, 0.1, 10.0)
    end

    pv       = copy(init_params)
    velocity = zeros(4)
    history  = Float64[]
    ph       = zeros(cfg.n_steps, 4)

    for step in 1:cfg.n_steps
        loss = GradientCalc.trajectory_loss(pv, observed, u0, tspan, dt)
        grad = GradientCalc.finite_diff_gradient(pv, observed, u0, tspan, dt; ε=cfg.ε_fd)
        push!(history, loss)
        ph[step, :] = pv

        # Momentum gradient descent
        velocity = cfg.momentum .* velocity .+ (1.0 - cfg.momentum) .* grad
        pv .-= cfg.lr .* velocity
        pv   = clamp.(pv, 0.05, 20.0)

        cfg.verbose && step % 10 == 0 && println("  step=$step  loss=$(round(loss;sigdigits=4))  pv=$(round.(pv;digits=3))")
    end

    tv    = [true_params.m1, true_params.m2, true_params.L1, true_params.L2]
    err   = abs.(pv .- tv) ./ (abs.(tv) .+ 1e-12) .* 100

    InferenceResult(
        PendulumParams(m1=pv[1], m2=pv[2], L1=pv[3], L2=pv[4]),
        history,
        ph,
        history[end],
        err,
    )
end

end
