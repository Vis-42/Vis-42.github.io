# shared/discovery.jl — SINDy, kernel conservation laws, symbolic GP
# All physics self-contained; no external ODE library.

using Random, Statistics, LinearAlgebra

# ── RK4 integrator ────────────────────────────────────────────────────────────

function rk4_step(f, state::Vector{Float64}, t::Float64, dt::Float64, p)
    k1 = f(state, t, p)
    k2 = f(state .+ 0.5dt .* k1, t + 0.5dt, p)
    k3 = f(state .+ 0.5dt .* k2, t + 0.5dt, p)
    k4 = f(state .+ dt   .* k3, t + dt,      p)
    state .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function integrate(f, u0::Vector{Float64}, tspan::Tuple{Float64,Float64},
                   n::Int, p=nothing)
    dt = (tspan[2] - tspan[1]) / (n - 1)
    states = zeros(n, length(u0))
    states[1, :] .= u0
    t = tspan[1]
    for i in 2:n
        states[i, :] .= rk4_step(f, states[i-1, :], t, dt, p)
        t += dt
    end
    range(tspan[1], tspan[2]; length=n) |> collect, states
end

# ── ODEs ──────────────────────────────────────────────────────────────────────

function ode_van_der_pol(s, t, p)
    μ = get(p, :mu, 1.0)
    [s[2], μ*(1 - s[1]^2)*s[2] - s[1]]
end

function ode_duffing(s, t, p)
    [s[2], p[:gamma]*cos(p[:omega]*t) - p[:delta]*s[2] - p[:alpha]*s[1] - p[:beta]*s[1]^3]
end

function ode_pendulum_driven(s, t, p)
    [s[2], -p[:b]*s[2] - sin(s[1]) + p[:A]*cos(p[:omega]*t)]
end

function ode_pendulum(s, t, p)
    [s[2], -9.81*sin(s[1])]
end

function ode_kepler(s, t, p)
    r = sqrt(s[1]^2 + s[2]^2)
    [s[3], s[4], -s[1]/r^3, -s[2]/r^3]
end

function ode_duffing_free(s, t, p)
    [s[2], s[1] - s[1]^3]
end

# ── Trajectory generators ────────────────────────────────────────────────────

const SYS_MAP = Dict(
    "van_der_pol"     => (ode_van_der_pol, Dict(:mu=>1.0), [2.0, 0.0], (0.0, 25.0), 800),
    "duffing"         => (ode_duffing, Dict(:alpha=>-1.0,:beta=>1.0,:delta=>0.2,:gamma=>0.3,:omega=>1.2), [0.5,0.0], (0.0,30.0), 900),
    "driven_pendulum" => (ode_pendulum_driven, Dict(:b=>0.2,:A=>1.2,:omega=>2.0/3.0), [0.1,0.0], (0.0,40.0), 1200),
)

"""Generate trajectory for a named system, optionally with measurement noise."""
function gen_trajectory(sysname::String; noise::Float64=0.0, seed::Int=42)
    f, p, u0, tspan, n = SYS_MAP[sysname]
    ts, states = integrate(f, Float64.(u0), tspan, n, p)
    dt = ts[2] - ts[1]
    if noise > 0
        rng = MersenneTwister(seed)
        states .+= noise .* randn(rng, size(states)...)
    end
    ts, states, dt
end

# ── SINDy: library construction ──────────────────────────────────────────────

function poly_exponents(d::Int, max_deg::Int)
    exps = Vector{Vector{Int}}()
    function rec(dim, rem, cur)
        if dim > d
            sum(cur) > 0 && push!(exps, copy(cur))
            return
        end
        for e in 0:rem
            cur[dim] = e
            rec(dim+1, rem-e, cur)
        end
    end
    rec(1, max_deg, zeros(Int, d))
    exps
end

"""Build polynomial (+ optional trig) library matrix from state snapshots."""
function build_library(states::Matrix{Float64}, max_deg::Int, inc_trig::Bool)
    n, d = size(states)
    cols  = [ones(n)]
    names = String["1"]
    for exps in poly_exponents(d, max_deg)
        col = ones(n)
        nm  = String[]
        for (j, e) in enumerate(exps)
            e == 1 && (push!(nm, "x$j"); col .*= states[:, j])
            e > 1  && (push!(nm, "x$j^$e"); col .*= states[:, j].^e)
        end
        push!(cols, col); push!(names, join(nm, "·"))
    end
    if inc_trig
        for j in 1:d
            push!(cols, sin.(states[:, j])); push!(names, "sin(x$j)")
            push!(cols, cos.(states[:, j])); push!(names, "cos(x$j)")
        end
    end
    reduce(hcat, cols), names
end

"""Centered finite-difference time derivatives, O(Δt²) interior."""
function finite_diff(states::Matrix{Float64}, dt::Float64)
    n, d = size(states)
    dX = similar(states)
    for i in 2:n-1
        dX[i, :] .= (states[i+1, :] .- states[i-1, :]) ./ (2dt)
    end
    dX[1,   :] .= (states[2,   :] .- states[1,   :]) ./ dt
    dX[end, :] .= (states[end, :] .- states[end-1,:]) ./ dt
    dX
end

"""STLSQ sparse regression: iterate least-squares + hard threshold λ."""
function stlsq(Θ::Matrix{Float64}, dX::Matrix{Float64},
               λ::Float64, max_iter::Int=20)
    xi = Θ \ dX
    for _ in 1:max_iter
        small = abs.(xi) .< λ
        xi[small] .= 0.0
        for j in 1:size(dX, 2)
            idx = .!small[:, j]
            any(idx) && (xi[idx, j] .= Θ[:, idx] \ dX[:, j])
        end
    end
    xi
end

"""Integrate the SINDy-identified model forward from u0."""
function sindy_reconstruct(xi::Matrix{Float64}, u0::Vector{Float64},
                            n::Int, dt::Float64, max_deg::Int, inc_trig::Bool)
    d    = length(u0)
    traj = fill(NaN64, n, d)
    traj[1, :] .= u0
    function fhat(sv)
        sm   = reshape(sv, 1, d)
        Th,_ = build_library(sm, max_deg, inc_trig)
        vec(Th * xi)
    end
    for i in 2:n
        s = traj[i-1, :]
        any(isnan.(s)) && break
        k1 = fhat(s); k2 = fhat(s .+ 0.5dt .* k1)
        k3 = fhat(s .+ 0.5dt .* k2); k4 = fhat(s .+ dt .* k3)
        ns = s .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        (any(isnan.(ns)) || any(isinf.(ns)) || maximum(abs.(ns)) > 200) && break
        traj[i, :] .= ns
    end
    traj
end

# ── Conservation law discovery via kernel generalized eigenproblem ────────────

"""
Discover a conserved quantity from trajectory data.
Returns the kernel-discovered invariant f(xᵢ) for each snapshot i.
"""
function discover_invariant(states::Matrix{Float64}, dt::Float64;
                            γ::Float64=1.0, α::Float64=1e-5, n_anchors::Int=200)
    n  = size(states, 1)
    m  = min(n_anchors, n)
    idx = round.(Int, range(1, n; length=m))

    # RBF feature matrix
    Φ = zeros(Float64, n, m)
    @inbounds for i in 1:n, j in 1:m
        Φ[i, j] = exp(-γ * sum((states[i, :] .- states[idx[j], :]).^2))
    end
    Φc  = Φ .- mean(Φ; dims=1)
    dΦ  = diff(Φ; dims=1) ./ dt

    A = Symmetric(dΦ' * dΦ  + α   * I(m))
    B = Symmetric(Φc' * Φc  + 1e-8 * I(m))

    ev = eigen(A, B)
    c  = ev.vectors[:, argmin(real.(ev.values))]
    f  = vec(Φc * real.(c))
    f .-= f[1]
    f
end

"""Project kernel invariant onto polynomial monomials."""
function poly_fit_invariant(states::Matrix{Float64}, inv_vec::Vector{Float64})
    n, d = size(states)
    cols = [ones(n)]; nms = String["1"]
    for i in 1:d
        push!(cols, states[:, i]);          push!(nms, "x$i")
        push!(cols, states[:, i].^2);       push!(nms, "x$i²")
    end
    for i in 1:d, j in i+1:d
        push!(cols, states[:, i] .* states[:, j]); push!(nms, "x$(i)·x$j")
    end
    X = reduce(hcat, cols)
    β = X \ inv_vec
    terms = ["$(round(c; sigdigits=3)) $(nm)"
             for (c, nm) in zip(β, nms) if abs(c) > 1e-3]
    join(isempty(terms) ? ["0"] : terms, " + "), mean((X*β .- inv_vec).^2)
end

# Conservative system generators
function pendulum_data(; θ₀::Float64=1.2, n::Int=600)
    ts, s = integrate(ode_pendulum, [θ₀, 0.0], (0.0, 20.0), n)
    ts, s, [0.5*s[i,2]^2 - 9.81*cos(s[i,1]) for i in axes(s,1)]
end

function kepler_data(; ecc::Float64=0.25, n::Int=600)
    ecc = clamp(ecc, 0.0, 0.92)
    u0  = [1.0-ecc, 0.0, 0.0, sqrt((1.0+ecc)/(1.0-ecc))]
    ts, s = integrate(ode_kepler, u0, (0.0, 4*2π), n)
    ts, s, [0.5*(s[i,3]^2+s[i,4]^2) - 1/hypot(s[i,1],s[i,2]) for i in axes(s,1)]
end

function duffing_data(; x₀::Float64=0.6, n::Int=600)
    ts, s = integrate(ode_duffing_free, [x₀, 0.0], (0.0, 40.0), n)
    ts, s, [0.5*s[i,2]^2 + 0.25*s[i,1]^4 - 0.5*s[i,1]^2 for i in axes(s,1)]
end

# ── Symbolic regression via genetic programming ───────────────────────────────

const BIN_OPS = Symbol[:+, :-, :*, :/]
const UN_OPS  = Symbol[:sqrt, :sin, :cos]

mutable struct ExprNode
    op      ::Symbol
    value   ::Float64
    var_idx ::Int
    children::Vector{ExprNode}
end
ExprNode(op::Symbol) = ExprNode(op, 0.0, 0, ExprNode[])
ExprNode(v::Float64) = ExprNode(:const, v, 0, ExprNode[])
ExprNode(i::Int)     = ExprNode(:var,  0.0, i, ExprNode[])

function copy_tree(n::ExprNode)
    ExprNode(n.op, n.value, n.var_idx, [copy_tree(c) for c in n.children])
end

count_nodes(n::ExprNode) = 1 + sum(count_nodes(c) for c in n.children; init=0)

function tree_string(n::ExprNode, vars::Vector{String})
    n.op == :const && return string(round(n.value; sigdigits=3))
    n.op == :var   && return (n.var_idx <= length(vars) ? vars[n.var_idx] : "x$(n.var_idx)")
    if n.op in BIN_OPS
        return "($(tree_string(n.children[1],vars)) $(n.op) $(tree_string(n.children[2],vars)))"
    end
    "$(n.op)($(tree_string(n.children[1],vars)))"
end

function random_tree(nv::Int, rng::AbstractRNG; depth::Int=0, max_depth::Int=4)
    if depth >= max_depth || (depth > 1 && rand(rng) < 0.4)
        return rand(rng) < 0.6 ? ExprNode(rand(rng, 1:nv)) : ExprNode(randn(rng) * 2.0)
    end
    op   = rand(rng, vcat(BIN_OPS, UN_OPS))
    node = ExprNode(op)
    nch  = op in BIN_OPS ? 2 : 1
    for _ in 1:nch
        push!(node.children, random_tree(nv, rng; depth=depth+1, max_depth=max_depth))
    end
    node
end

function eval_tree(n::ExprNode, X::AbstractMatrix{Float64})
    N = size(X, 1)
    n.op == :const && return fill(n.value, N)
    n.op == :var   && return X[:, clamp(n.var_idx, 1, size(X,2))]
    n.op == :+  && return eval_tree(n.children[1],X) .+ eval_tree(n.children[2],X)
    n.op == :-  && return eval_tree(n.children[1],X) .- eval_tree(n.children[2],X)
    n.op == :*  && return eval_tree(n.children[1],X) .* eval_tree(n.children[2],X)
    n.op == :/  && return eval_tree(n.children[1],X) ./ (abs.(eval_tree(n.children[2],X)) .+ 1e-10)
    n.op == :sqrt && return sqrt.(abs.(eval_tree(n.children[1],X)))
    n.op == :sin  && return sin.(eval_tree(n.children[1],X))
    n.op == :cos  && return cos.(eval_tree(n.children[1],X))
    zeros(N)
end

function fitness(n::ExprNode, X::Matrix{Float64}, y::Vector{Float64}; cp::Float64=0.002)
    try
        pred = eval_tree(n, X)
        (any(isnan, pred) || any(isinf, pred)) && return 1e12
        mse  = mean((pred .- y).^2)
        mse / max(var(y), 1e-12) + cp * count_nodes(n)
    catch
        1e12
    end
end

function collect_nodes!(n::ExprNode, acc::Vector{ExprNode})
    push!(acc, n)
    for c in n.children; collect_nodes!(c, acc); end
end

function rand_parent(n::ExprNode, rng::AbstractRNG)
    !isempty(n.children) && rand(rng) < 0.45 && return n, rand(rng, 1:length(n.children))
    for c in n.children
        p, i = rand_parent(c, rng)
        !isnothing(p) && return p, i
    end
    nothing, 0
end

function mutate!(n::ExprNode, nv::Int, rng::AbstractRNG; rate::Float64=0.2)
    if rand(rng) < rate
        if     n.op == :const;  n.value   += randn(rng) * 0.3
        elseif n.op == :var;    n.var_idx  = rand(rng, 1:nv)
        elseif n.op in BIN_OPS; n.op       = rand(rng, BIN_OPS)
        else;                   n.op       = rand(rng, UN_OPS)
        end
    end
    for c in n.children; mutate!(c, nv, rng; rate=rate); end
end

function crossover(t1::ExprNode, t2::ExprNode, rng::AbstractRNG)
    t1 = copy_tree(t1); t2 = copy_tree(t2)
    all_t2 = ExprNode[]; collect_nodes!(t2, all_t2)
    isempty(all_t2) && return t1
    src  = copy_tree(rand(rng, all_t2))
    p, i = rand_parent(t1, rng)
    !isnothing(p) && (p.children[i] = src)
    t1
end

"""Run genetic programming. Returns (best_tree, R², node_count, best_history, pop_fitnesses)."""
function gp_run(X::Matrix{Float64}, y::Vector{Float64}, nv::Int;
                n_gen::Int=80, pop_size::Int=120, seed::Int=42)
    rng  = MersenneTwister(seed)
    pop  = [random_tree(nv, rng) for _ in 1:pop_size]
    fits = [fitness(t, X, y) for t in pop]
    best_hist   = Float64[]
    pop_hist    = Vector{Vector{Float64}}()

    for gen in 1:n_gen
        push!(best_hist, minimum(fits))
        push!(pop_hist, copy(sort(fits)[1:min(20, pop_size)]))

        elite_n = max(1, pop_size ÷ 20)
        ord     = sortperm(fits)
        new_pop = ExprNode[copy_tree(pop[ord[i]]) for i in 1:elite_n]
        new_fits = Float64[fits[ord[i]] for i in 1:elite_n]

        while length(new_pop) < pop_size
            i1, i2 = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            p1 = fits[i1] < fits[i2] ? pop[i1] : pop[i2]
            i3, i4 = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            p2 = fits[i3] < fits[i4] ? pop[i3] : pop[i4]

            child = rand(rng) < 0.7 ? crossover(p1, p2, rng) : copy_tree(p1)
            mutate!(child, nv, rng)
            count_nodes(child) > 30 && (child = random_tree(nv, rng; max_depth=3))
            push!(new_pop, child)
            push!(new_fits, fitness(child, X, y))
        end
        pop = new_pop; fits = new_fits
    end

    best_idx = argmin(fits)
    best     = pop[best_idx]
    pred     = eval_tree(best, X)
    ȳ        = mean(y)
    r2       = max(0.0, 1 - sum((y .- pred).^2) / max(sum((y .- ȳ).^2), 1e-12))
    best, r2, count_nodes(best), best_hist, pop_hist
end

# Feynman benchmark data generators
const FEYNMAN_EQNS = [
    ("Kinetic Energy",   "E = ½mv²",     ["m","v"],
     (X -> 0.5 .* X[:,1] .* X[:,2].^2),
     (rng, N) -> hcat(0.1 .+ 9.9 .* rand(rng, N), 0.1 .+ 9.9 .* rand(rng, N))),
    ("Pendulum Period",  "T = 2π√(L/g)", ["L"],
     (X -> 2π .* sqrt.(X[:,1] ./ 9.81)),
     (rng, N) -> hcat(0.1 .+ 4.9 .* rand(rng, N))),
    ("Ohm's Law",        "V = IR",        ["I","R"],
     (X -> X[:,1] .* X[:,2]),
     (rng, N) -> hcat(0.01 .+ 9.99 .* rand(rng, N), 1.0 .+ 99.0 .* rand(rng, N))),
    ("Hooke's Law",      "F = kx",        ["k","x"],
     (X -> X[:,1] .* X[:,2]),
     (rng, N) -> hcat(1.0 .+ 99.0 .* rand(rng, N), 0.01 .+ 0.99 .* rand(rng, N))),
    ("Gravitational PE", "U = mgh",       ["m","g","h"],
     (X -> X[:,1] .* X[:,2] .* X[:,3]),
     (rng, N) -> hcat(0.1 .+ 9.9 .* rand(rng, N), fill(9.81, N), 0.1 .+ 9.9 .* rand(rng, N))),
]

function feynman_data(eq_idx::Int; N::Int=300, noise_σ::Float64=0.0, seed::Int=42)
    rng = MersenneTwister(seed)
    _, _, _, fn, sampler = FEYNMAN_EQNS[eq_idx]
    X       = sampler(rng, N)
    ndims(X) == 1 && (X = reshape(X, N, 1))
    y_clean = fn(X)
    σ_y     = max(std(y_clean), 1e-12)
    y       = y_clean .+ noise_σ .* σ_y .* randn(rng, N)
    X, y, y_clean
end
