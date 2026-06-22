module ChaoticSystems

export lorenz63, rossler, henon, generate_trajectory,
       largest_lyapunov_exponent, lyapunov_time, is_discrete

# ── Dynamical systems ────────────────────────────────────────────────────────
# Continuous flows return a *derivative* du/dt (integrated by RK4).
# Discrete maps return the *next state* u_{n+1} directly (iterated, no RK4).

function lorenz63(u::Vector{Float64}, p, t)
    x, y, z = u
    sigma = get(p, :sigma, 10.0)
    rho   = get(p, :rho,   28.0)
    beta  = get(p, :beta,  8.0/3.0)
    [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
end

function rossler(u::Vector{Float64}, p, t)
    x, y, z = u
    a = get(p, :a, 0.2)
    b = get(p, :b, 0.2)
    c = get(p, :c, 5.7)
    [-y - z, x + a*y, b + z*(x - c)]
end

# Hénon is a discrete-time map u_{n+1} = f(u_n), NOT a vector field.
# It must be iterated directly; integrating it with RK4 is meaningless.
function henon(u::Vector{Float64}, p, t)
    x, y = u
    a = get(p, :a, 1.4)
    b = get(p, :b, 0.3)
    [1.0 - a*x^2 + y, b*x]
end

const DISCRETE_SYSTEMS = Set(["henon"])
is_discrete(system::String) = system in DISCRETE_SYSTEMS

const SYSTEM_FN = Dict{String,Function}(
    "lorenz63" => lorenz63, "rossler" => rossler, "henon" => henon)

const DEFAULT_U0 = Dict{String,Vector{Float64}}(
    "lorenz63" => [1.0, 0.0, 0.0], "rossler" => [1.0, 0.0, 0.0], "henon" => [0.1, 0.0])

function rk4_step(f, u::Vector{Float64}, t::Float64, dt::Float64, p)
    k1 = f(u, p, t)
    k2 = f(u .+ 0.5dt.*k1, p, t + 0.5dt)
    k3 = f(u .+ 0.5dt.*k2, p, t + 0.5dt)
    k4 = f(u .+ dt.*k3,    p, t + dt)
    u .+ (dt/6.0).*(k1 .+ 2k2 .+ 2k3 .+ k4)
end

# One step of the system: RK4 for flows, direct iteration for maps.
function step_system(system::String, f, u::Vector{Float64}, t::Float64, dt::Float64, p)
    is_discrete(system) ? f(u, p, t) : rk4_step(f, u, t, dt, p)
end

# For flows, one step advances time by dt; for maps, by one iteration (dt_eff = 1).
effective_dt(system::String, dt::Float64) = is_discrete(system) ? 1.0 : dt

function generate_trajectory(system::String, params::Dict, T::Float64, dt::Float64;
                              u0::Vector{Float64} = Float64[])
    f = SYSTEM_FN[system]
    n = round(Int, T/dt) + 1
    u0_use = isempty(u0) ? DEFAULT_U0[system] : u0
    traj = zeros(n, length(u0_use))
    traj[1, :] .= u0_use
    t = 0.0
    for i in 2:n
        traj[i, :] .= step_system(system, f, traj[i-1, :], t, dt, params)
        t += dt
    end
    dt_eff = effective_dt(system, dt)
    collect(0.0:dt_eff:dt_eff*(n-1)), traj
end

# ── Largest Lyapunov exponent (Benettin two-trajectory method) ───────────────
# Evolve a reference and a shadow trajectory separated by d0; after each step
# measure the growth of the separation, accumulate log(d/d0), and renormalise
# the shadow back to distance d0 along the current separation direction.
# λ_max = (1/total_time) Σ log(dₙ/d0).  Works uniformly for flows and maps.
function largest_lyapunov_exponent(system::String, params::Dict;
                                    dt::Float64 = 0.01, n_steps::Int = 100_000,
                                    transient::Int = 1_000, d0::Float64 = 1e-9,
                                    u0::Vector{Float64} = Float64[])
    f      = SYSTEM_FN[system]
    dt_eff = effective_dt(system, dt)
    u = copy(isempty(u0) ? DEFAULT_U0[system] : u0)

    # Settle onto the attractor before measuring.
    t = 0.0
    for _ in 1:transient
        u = step_system(system, f, u, t, dt, params); t += dt
    end

    # Shadow trajectory, perturbed along the first coordinate.
    v = copy(u); v[1] += d0
    accum = 0.0
    for _ in 1:n_steps
        u = step_system(system, f, u, t, dt, params)
        v = step_system(system, f, v, t, dt, params)
        t += dt
        d = sqrt(sum((v .- u).^2))
        d < 1e-300 && continue
        accum += log(d / d0)
        v = u .+ (d0 / d) .* (v .- u)   # renormalise separation back to d0
    end
    accum / (n_steps * dt_eff)
end

# Lyapunov time = 1 / λ_max, computed directly from the dynamics (self-consistent).
# Literature reference values (for validation): Lorenz ≈ 0.906, Rössler ≈ 0.071,
# Hénon ≈ 0.419 (per iteration).
function lyapunov_time(system::String, params::Dict = Dict{Symbol,Float64}(); kwargs...)
    λ = largest_lyapunov_exponent(system, params; kwargs...)
    1.0 / λ
end

end
