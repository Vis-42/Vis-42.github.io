module Network

using LinearAlgebra, Random, Statistics

export MLPConfig, MLP, forward, loss, gradient_step!, hessian_diagonal

Base.@kwdef mutable struct MLPConfig
    n_in::Int      = 2
    n_hidden::Int  = 20
    n_out::Int     = 2
    lr::Float64    = 0.01
end

mutable struct MLP
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    cfg::MLPConfig
end

function MLP(cfg::MLPConfig; seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    scale1 = sqrt(2.0 / cfg.n_in)
    scale2 = sqrt(2.0 / cfg.n_hidden)
    MLP(
        randn(rng, cfg.n_hidden, cfg.n_in) .* scale1,
        zeros(cfg.n_hidden),
        randn(rng, cfg.n_out, cfg.n_hidden) .* scale2,
        zeros(cfg.n_out),
        cfg,
    )
end

relu(x) = max(0.0, x)
drelu(x) = x > 0.0 ? 1.0 : 0.0

function forward(mlp::MLP, X::AbstractMatrix)
    h = relu.(mlp.W1 * X' .+ mlp.b1)   # n_hidden × N
    o = mlp.W2 * h .+ mlp.b2            # n_out × N
    o'   # N × n_out
end

function softmax_loss(logits::Matrix{Float64}, y::Vector{Int})
    N = size(logits, 1)
    l = 0.0
    for i in 1:N
        m = maximum(logits[i, :])
        lse = log(sum(exp(logits[i, j] - m) for j in 1:size(logits, 2))) + m
        l -= logits[i, y[i]] - lse
    end
    l / N
end

function loss(mlp::MLP, X::AbstractMatrix, y::AbstractVector{Int})
    logits = forward(mlp, X)
    softmax_loss(Matrix{Float64}(logits), y)
end

function gradient_step!(mlp::MLP, X::AbstractMatrix, y::AbstractVector{Int}; lr::Float64 = mlp.cfg.lr)
    N = size(X, 1)
    # Forward
    H_pre = mlp.W1 * X' .+ mlp.b1      # n_h × N
    H     = relu.(H_pre)
    O     = mlp.W2 * H .+ mlp.b2       # n_out × N

    # Softmax + CE gradient
    exp_O = exp.(O .- maximum(O; dims=1))
    probs = exp_O ./ sum(exp_O; dims=1)  # n_out × N
    dO = probs ./ N
    for i in 1:N; dO[y[i], i] -= 1.0/N; end

    dW2 = dO * H'
    db2 = vec(sum(dO; dims=2))
    dH  = mlp.W2' * dO
    dH_pre = dH .* drelu.(H_pre)
    dW1 = dH_pre * X
    db1 = vec(sum(dH_pre; dims=2))

    mlp.W1 .-= lr .* dW1; mlp.b1 .-= lr .* db1
    mlp.W2 .-= lr .* dW2; mlp.b2 .-= lr .* db2
end

"""
    hessian_diagonal(mlp, X, y; eps=1e-3)

Estimate diagonal of Hessian via finite differences of the gradient.
"""
function hessian_diagonal(mlp::MLP, X::AbstractMatrix, y::AbstractVector{Int}; eps::Float64 = 1e-3)
    params = vcat(vec(mlp.W1), vec(mlp.b1), vec(mlp.W2), vec(mlp.b2))
    n      = length(params)
    diag_H = zeros(n)

    base_loss = loss(mlp, X, y)

    function set_params!(m, p)
        nh, ni = size(m.W1); no = size(m.W2, 1)
        i = 1
        m.W1 .= reshape(p[i:i+nh*ni-1], nh, ni); i += nh*ni
        m.b1 .= p[i:i+nh-1];                      i += nh
        m.W2 .= reshape(p[i:i+no*nh-1], no, nh);  i += no*nh
        m.b2 .= p[i:end]
    end

    for k in 1:n
        p_plus  = copy(params); p_plus[k]  += eps
        p_minus = copy(params); p_minus[k] -= eps
        m1 = deepcopy(mlp); m2 = deepcopy(mlp)
        set_params!(m1, p_plus); set_params!(m2, p_minus)
        l_plus  = loss(m1, X, y)
        l_minus = loss(m2, X, y)
        diag_H[k] = (l_plus - 2base_loss + l_minus) / eps^2
    end
    set_params!(mlp, params)
    diag_H
end

end
