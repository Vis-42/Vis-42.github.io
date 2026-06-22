# shared/vicsek.jl — Vicsek model and run-and-tumble dynamics

using Random, Statistics, LinearAlgebra

# ── Vicsek model ─────────────────────────────────────────────────────────────

struct VicsekState
    x::Vector{Float64}
    y::Vector{Float64}
    θ::Vector{Float64}
    L::Float64
end

function init_vicsek(N::Int, ρ::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    L = sqrt(N / ρ)
    x = rand(rng, N) .* L
    y = rand(rng, N) .* L
    θ = rand(rng, N) .* 2π .- π
    VicsekState(x, y, θ, L)
end

function vicsek_step!(state::VicsekState, η::Float64; v0::Float64=0.03, r::Float64=1.0, seed_offset::Int=0)
    N = length(state.x)
    L = state.L
    rng = MersenneTwister(seed_offset * N)

    θ_new = copy(state.θ)

    @inbounds for i in 1:N
        # Accumulate neighbor headings
        sx, sy = 0.0, 0.0
        for j in 1:N
            dx = state.x[j] - state.x[i]
            dy = state.y[j] - state.y[i]
            # Minimum image convention
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            if dx^2 + dy^2 <= r^2
                sx += cos(state.θ[j])
                sy += sin(state.θ[j])
            end
        end
        θ_avg = atan(sy, sx)
        noise = (rand(rng) - 0.5) * η
        θ_new[i] = θ_avg + noise
    end

    @inbounds for i in 1:N
        state.θ[i] = θ_new[i]
        state.x[i] = mod(state.x[i] + v0 * cos(state.θ[i]), L)
        state.y[i] = mod(state.y[i] + v0 * sin(state.θ[i]), L)
    end
    nothing
end

function order_parameter(θ::Vector{Float64})
    N = length(θ)
    abs(sum(exp.(im .* θ))) / N
end

"""Run Vicsek model for n_steps, return time series of φ."""
function simulate_vicsek(N::Int, η::Float64;
                         ρ::Float64=1.5, v0::Float64=0.03, r::Float64=1.0,
                         n_warm::Int=200, n_run::Int=300, seed::Int=42)
    state = init_vicsek(N, ρ; seed=seed)
    for step in 1:n_warm
        vicsek_step!(state, η; v0=v0, r=r, seed_offset=step)
    end
    φ_trace = zeros(n_run)
    for step in 1:n_run
        vicsek_step!(state, η; v0=v0, r=r, seed_offset=n_warm+step)
        φ_trace[step] = order_parameter(state.θ)
    end
    state, φ_trace
end

# ── Run-and-tumble ────────────────────────────────────────────────────────────

struct RTState
    x::Vector{Float64}
    y::Vector{Float64}
    θ::Vector{Float64}
    L::Float64
end

function init_runtumble(N::Int; L::Float64=20.0, seed::Int=42)
    rng = MersenneTwister(seed)
    x = rand(rng, N) .* L
    y = rand(rng, N) .* L
    θ = rand(rng, N) .* 2π
    RTState(x, y, θ, L)
end

function runtumble_step!(state::RTState, rng::MersenneTwister;
                         v::Float64, λ::Float64, dt::Float64, κ::Float64=0.0)
    N = length(state.x)
    @inbounds for i in 1:N
        if rand(rng) < λ * dt
            state.θ[i] = rand(rng) * 2π
        end
        state.x[i] += v * dt * cos(state.θ[i])
        state.y[i] += v * dt * sin(state.θ[i])
        # Optional harmonic confinement
        if κ > 0
            cx, cy = state.L / 2, state.L / 2
            state.x[i] -= κ * dt * (state.x[i] - cx)
            state.y[i] -= κ * dt * (state.y[i] - cy)
        end
        state.x[i] = mod(state.x[i], state.L)
        state.y[i] = mod(state.y[i], state.L)
    end
    nothing
end

"""Theoretical MSD for 2D run-and-tumble."""
function msd_theory(t::AbstractVector{Float64}; v::Float64, λ::Float64)
    @. (v^2 / λ^2) * (2λ * t - 2 * (1 - exp(-λ * t)))
end

"""Simulate run-and-tumble, compute ensemble MSD (unwrapped displacement)."""
function simulate_runtumble_msd(N::Int;
                                 v::Float64=1.0, λ::Float64=0.5,
                                 dt::Float64=0.05, T::Float64=30.0,
                                 seed::Int=42)
    rng = MersenneTwister(seed)
    n_steps = round(Int, T / dt)
    θ = rand(rng, N) .* 2π   # initial angles
    rx = zeros(N)             # cumulative x-displacement (unwrapped)
    ry = zeros(N)             # cumulative y-displacement

    ts   = Float64[]
    msds = Float64[]
    sample_every = max(1, n_steps ÷ 200)

    for step in 1:n_steps
        for i in 1:N
            if rand(rng) < λ * dt
                θ[i] = rand(rng) * 2π
            end
            rx[i] += v * dt * cos(θ[i])
            ry[i] += v * dt * sin(θ[i])
        end
        if step % sample_every == 0
            push!(ts,   step * dt)
            push!(msds, mean(rx.^2 .+ ry.^2))
        end
    end
    ts, msds
end
