module PendulumSim

import Random

export PendulumParams, pendulum_ode, simulate, simulate_noisy

Base.@kwdef struct PendulumParams
    m1::Float64 = 1.0
    m2::Float64 = 1.0
    L1::Float64 = 1.0
    L2::Float64 = 1.0
    g::Float64  = 9.81
end

# Double pendulum ODE — state = [θ1, ω1, θ2, ω2]
function pendulum_ode(u::AbstractVector{Float64}, p::PendulumParams, t::Float64)
    θ1, ω1, θ2, ω2 = u
    Δ  = θ2 - θ1
    sΔ = sin(Δ); cΔ = cos(Δ)
    g  = p.g; m1 = p.m1; m2 = p.m2; L1 = p.L1; L2 = p.L2
    denom1 = (2m1 + m2 - m2*cos(2Δ)) * L1
    denom2 = L2 / L1 * denom1

    # Sign: Δ = θ2-θ1, so sin(θ1-θ2) = -sΔ; the standard formula has -2sin(θ1-θ2)·m2·(...)
    # which becomes +2sΔ·m2·(...) after substitution (and -2sΔ·(...) in α2).
    α1 = (-g*(2m1+m2)*sin(θ1) - m2*g*sin(θ1-2θ2)
          + 2sΔ*m2*(ω2^2*L2 + ω1^2*L1*cΔ)) / denom1
    α2 = (-2sΔ*(ω1^2*L1*(m1+m2) + g*(m1+m2)*cos(θ1)
          + ω2^2*L2*m2*cΔ)) / denom2

    [ω1, α1, ω2, α2]
end

# Runge-Kutta 4 integrator
function rk4_step(u, p, t, dt)
    k1 = pendulum_ode(u,         p, t)
    k2 = pendulum_ode(u .+ dt/2 .* k1, p, t + dt/2)
    k3 = pendulum_ode(u .+ dt/2 .* k2, p, t + dt/2)
    k4 = pendulum_ode(u .+ dt   .* k3, p, t + dt)
    u .+ dt/6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function simulate(params::PendulumParams, u0::Vector{Float64},
                  tspan::Tuple{Float64,Float64}, dt::Float64)
    t0, t1 = tspan
    n = max(2, round(Int, (t1 - t0) / dt) + 1)
    times = range(t0, t1; length=n)
    traj  = zeros(n, 4)
    traj[1, :] = u0
    t = t0
    for i in 2:n
        traj[i, :] = rk4_step(traj[i-1, :], params, t, dt)
        t += dt
    end
    collect(times), traj
end

function simulate_noisy(params::PendulumParams, u0::Vector{Float64},
                        tspan::Tuple{Float64,Float64}, dt::Float64;
                        σ::Float64=0.05, seed::Int=42)
    times, traj = simulate(params, u0, tspan, dt)
    rng = Random.MersenneTwister(seed)
    noise = σ .* Random.randn(rng, size(traj))
    times, traj .+ noise, traj
end

end
