module MonteCarlo

using ..GoModel
using Random, Statistics

export MCConfig, mc_step!, replica_exchange_mc, ReplicaResult

Base.@kwdef struct MCConfig
    n_steps::Int    = 10000
    step_size::Float64 = 0.05
    swap_freq::Int  = 100
end

struct ReplicaResult
    T::Float64
    Q_traj::Vector{Float64}   # fraction native contacts trajectory
    E_traj::Vector{Float64}   # energy trajectory
    coords_final::Matrix{Float64}
end

function mc_step!(coords::Matrix{Float64}, cfg::ProteinConfig, T::Float64,
                  rng::AbstractRNG, step_size::Float64)::Bool
    n = size(coords, 1)
    i = rand(rng, 1:n)
    δ = (rand(rng, 3) .- 0.5) .* 2step_size

    E_old = go_energy(coords, cfg)
    coords[i,:] .+= δ
    E_new = go_energy(coords, cfg)

    ΔE = E_new - E_old
    if ΔE <= 0.0 || rand(rng) < exp(-ΔE / max(T, 1e-6))
        return true   # accepted
    else
        coords[i,:] .-= δ   # reject
        return false
    end
end

function replica_exchange_mc(cfg::ProteinConfig, T_ladder::Vector{Float64};
                              n_steps::Int=5000, swap_freq::Int=200, seed::Int=42)::Vector{ReplicaResult}
    n_rep = length(T_ladder)
    rng   = MersenneTwister(seed)
    step_size = 0.05

    # Initialize each replica near native structure with small perturbations
    replicas = [cfg.native_coords .+ 0.05.*randn(rng, size(cfg.native_coords)) for _ in 1:n_rep]
    energies = [go_energy(r, cfg) for r in replicas]

    # Store trajectories (sampled every 50 steps)
    sample_freq = max(1, n_steps ÷ 200)
    Q_trajs = [Float64[] for _ in 1:n_rep]
    E_trajs = [Float64[] for _ in 1:n_rep]

    n_swap_attempts = 0; n_swaps = 0

    for step in 1:n_steps
        # MC sweep for each replica
        for r in 1:n_rep
            mc_step!(replicas[r], cfg, T_ladder[r], rng, step_size)
        end

        # Sample
        if step % sample_freq == 0
            for r in 1:n_rep
                push!(Q_trajs[r], fraction_native_contacts(replicas[r], cfg))
                push!(E_trajs[r], go_energy(replicas[r], cfg))
            end
        end

        # Replica exchange
        if step % swap_freq == 0
            for r in 1:(n_rep-1)
                r1, r2 = r, r+1
                E1 = go_energy(replicas[r1], cfg)
                E2 = go_energy(replicas[r2], cfg)
                β1 = 1.0 / T_ladder[r1]; β2 = 1.0 / T_ladder[r2]
                # Parallel-tempering swap acceptance: Δ = (β1-β2)(E1-E2);
                # accept if Δ ≥ 0 or rand < exp(Δ). (Was (β1-β2)(E2-E1) = -Δ,
                # which inverted the criterion and swapped almost every attempt.)
                ΔΔ = (β1 - β2) * (E1 - E2)
                n_swap_attempts += 1
                if ΔΔ ≥ 0 || rand(rng) < exp(ΔΔ)
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    n_swaps += 1
                end
            end
        end
    end

    [ReplicaResult(T_ladder[r], Q_trajs[r], E_trajs[r], copy(replicas[r])) for r in 1:n_rep]
end

end
