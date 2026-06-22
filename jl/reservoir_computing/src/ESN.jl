module ESN

using LinearAlgebra
using Random
using Statistics

export ESNConfig, EchoStateNetwork, build_esn, train_esn!, predict_esn

Base.@kwdef struct ESNConfig
    N_reservoir::Int    = 100
    sparsity::Float64   = 0.1
    spectral_radius::Float64 = 0.9
    input_scaling::Float64   = 0.5
    leak_rate::Float64       = 1.0
    ridge_alpha::Float64     = 1e-6
    seed::Int           = 42
end

mutable struct EchoStateNetwork
    cfg::ESNConfig
    W_in::Matrix{Float64}    # N_res × n_in
    W_res::Matrix{Float64}   # N_res × N_res
    W_out::Union{Matrix{Float64}, Nothing}  # n_out × (N_res + n_in)
    state::Vector{Float64}   # current reservoir state
    last_input::Vector{Float64}  # last training input (for autonomous prediction)
end

function build_esn(cfg::ESNConfig; n_input::Int = 3)
    rng = MersenneTwister(cfg.seed)
    N   = cfg.N_reservoir

    W_in  = (2rand(rng, N, n_input) .- 1) .* cfg.input_scaling

    # Sparse random reservoir. Sample distinct entries without replacement so the
    # realised connection density equals `sparsity` exactly — the sweep axis of
    # the phase diagram. (Sampling with replacement loses ~5% of edges to
    # collisions, biasing the very quantity being measured.)
    W_res = zeros(Float64, N, N)
    n_edges = round(Int, N * N * cfg.sparsity)
    for k in randperm(rng, N * N)[1:n_edges]
        W_res[k] = randn(rng)
    end
    # Rescale to desired spectral radius
    ρ = maximum(abs.(eigvals(W_res)))
    ρ > 1e-8 && (W_res .*= cfg.spectral_radius / ρ)

    EchoStateNetwork(cfg, W_in, W_res, nothing, zeros(N), zeros(n_input))
end

function _reservoir_step!(esn::EchoStateNetwork, u::Vector{Float64})
    cfg = esn.cfg
    pre = esn.W_res * esn.state .+ esn.W_in * u
    new_state = tanh.(pre)
    esn.state .= (1 - cfg.leak_rate) .* esn.state .+ cfg.leak_rate .* new_state
    esn.state
end

function train_esn!(esn::EchoStateNetwork, data::Matrix{Float64}, washout::Int)
    n, d = size(data)
    N    = esn.cfg.N_reservoir
    esn.state .= 0.0

    # Collect reservoir states
    states_collected = zeros(Float64, n - washout - 1, N + d)
    for t in 1:n-1
        _reservoir_step!(esn, data[t, :])
        if t > washout
            states_collected[t - washout, 1:N] .= esn.state
            states_collected[t - washout, N+1:end] .= data[t, :]
        end
    end

    targets = data[washout+2:end, :]
    alpha   = esn.cfg.ridge_alpha
    A = states_collected' * states_collected + alpha * I(N + d)
    B = states_collected' * targets
    esn.W_out = (A \ B)'
    # Advance one more step so state matches end-of-training
    _reservoir_step!(esn, data[end, :])
    # Save last training input for prediction initialisation
    esn.last_input = copy(data[end, :])
    nothing
end

function predict_esn(esn::EchoStateNetwork, n_steps::Int)
    isnothing(esn.W_out) && error("ESN not trained. Call train_esn! first.")
    N = esn.cfg.N_reservoir
    d = size(esn.W_out, 1)
    preds = zeros(n_steps, d)
    inp   = copy(esn.last_input)   # seed from last training point
    for t in 1:n_steps
        _reservoir_step!(esn, inp)
        extended = vcat(esn.state, inp)
        out = esn.W_out * extended
        preds[t, :] .= out
        inp .= out
    end
    preds
end

end
