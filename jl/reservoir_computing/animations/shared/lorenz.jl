# shared/lorenz.jl — Lorenz-63 physics and ESN infrastructure (standalone, no module)

using LinearAlgebra
using Random
using Statistics

# ── Physical constants ────────────────────────────────────────────────────────
const σ_L      = 10.0
const ρ_L      = 28.0
const β_L      = 8.0 / 3.0
const λ_max    = 0.906
const T_lyapunov = 1.0 / λ_max   # ≈ 1.104 time units
const DT       = 0.02             # integration timestep

# ── Lorenz-63 ODE ─────────────────────────────────────────────────────────────
function lorenz_deriv(u::Vector{Float64})
    x, y, z = u
    return [σ_L * (y - x), x * (ρ_L - z) - y, x * y - β_L * z]
end

function rk4_step_lorenz(u::Vector{Float64}, dt::Float64 = DT)
    k1 = lorenz_deriv(u)
    k2 = lorenz_deriv(u .+ 0.5dt .* k1)
    k3 = lorenz_deriv(u .+ 0.5dt .* k2)
    k4 = lorenz_deriv(u .+ dt .* k3)
    u .+ (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

"""
    generate_lorenz(T, dt; u0, warmup) -> (ts, traj)

Integrate the Lorenz-63 system for T time units with step dt.
If warmup > 0, discard the first warmup time units.
Returns:
  ts   — Vector of length N (time axis)
  traj — N×3 matrix [x y z]
"""
function generate_lorenz(T::Float64, dt::Float64 = DT;
                          u0::Vector{Float64} = [1.0, 0.0, 0.0],
                          warmup::Float64 = 0.0)
    # Warmup phase (discard)
    u = copy(u0)
    n_warmup = round(Int, warmup / dt)
    for _ in 1:n_warmup
        u = rk4_step_lorenz(u, dt)
    end

    # Main trajectory
    n = round(Int, T / dt) + 1
    traj = zeros(Float64, n, 3)
    traj[1, :] .= u
    for i in 2:n
        u = rk4_step_lorenz(u, dt)
        traj[i, :] .= u
    end
    ts = collect(range(0.0, T; length = n))
    return ts, traj
end

# ── ESN infrastructure ────────────────────────────────────────────────────────
struct ESNParams
    N_res::Int
    sparsity::Float64
    spectral_radius::Float64
    input_scaling::Float64
    ridge_alpha::Float64
    seed::Int
end

ESNParams(; N_res=50, sparsity=0.15, spectral_radius=0.7,
            input_scaling=0.5, ridge_alpha=1e-6, seed=42) =
    ESNParams(N_res, sparsity, spectral_radius, input_scaling, ridge_alpha, seed)

mutable struct SimpleESN
    params::ESNParams
    W_in::Matrix{Float64}    # N_res × n_in
    W_res::Matrix{Float64}   # N_res × N_res
    W_out::Union{Matrix{Float64}, Nothing}   # n_out × (N_res + n_in)
    state::Vector{Float64}
    last_input::Vector{Float64}
end

function build_simple_esn(p::ESNParams; n_input::Int = 3)
    rng = MersenneTwister(p.seed)
    N   = p.N_res

    W_in = (2rand(rng, N, n_input) .- 1) .* p.input_scaling

    W_res = zeros(Float64, N, N)
    n_edges = round(Int, N * N * p.sparsity)
    for _ in 1:n_edges
        i, j = rand(rng, 1:N), rand(rng, 1:N)
        W_res[i, j] = randn(rng)
    end
    rho_cur = maximum(abs.(eigvals(W_res)))
    rho_cur > 1e-8 && (W_res .*= p.spectral_radius / rho_cur)

    SimpleESN(p, W_in, W_res, nothing, zeros(N), zeros(n_input))
end

function esn_step!(esn::SimpleESN, u::Vector{Float64})
    pre        = esn.W_res * esn.state .+ esn.W_in * u
    esn.state .= tanh.(pre)
    esn.state
end

function train_esn!(esn::SimpleESN, data::Matrix{Float64}, washout::Int = 100)
    n, d = size(data)
    N    = esn.params.N_res
    esn.state .= 0.0

    T_collect = n - washout - 1
    X = zeros(Float64, T_collect, N + d)
    for t in 1:n-1
        esn_step!(esn, data[t, :])
        if t > washout
            idx = t - washout
            X[idx, 1:N]    .= esn.state
            X[idx, N+1:end] .= data[t, :]
        end
    end

    Y     = data[washout+2:end, :]
    alpha = esn.params.ridge_alpha
    A = X' * X + alpha * I(N + d)
    B = X' * Y
    esn.W_out = (A \ B)'
    esn_step!(esn, data[end, :])
    esn.last_input = copy(data[end, :])
    nothing
end

function predict_esn(esn::SimpleESN, n_steps::Int)
    isnothing(esn.W_out) && error("ESN not trained.")
    N = esn.params.N_res
    d = size(esn.W_out, 1)
    preds = zeros(n_steps, d)
    inp   = copy(esn.last_input)
    for t in 1:n_steps
        esn_step!(esn, inp)
        extended  = vcat(esn.state, inp)
        out       = esn.W_out * extended
        preds[t, :] .= out
        inp       .= out
    end
    preds
end

"""Return eigenvalues of W_res (for visualization)."""
esn_eigenvalues(esn::SimpleESN) = eigvals(esn.W_res)
