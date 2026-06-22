module FreeEnergy

using ..MonteCarlo
using Statistics

export free_energy_profile, weighted_histogram, wham_simple

# Simple free energy from probability histogram: F(Q) = -kT ln P(Q)
function free_energy_profile(Q_traj::Vector{Float64}, T::Float64;
                              n_bins::Int=30)::Tuple{Vector{Float64},Vector{Float64}}
    isempty(Q_traj) && return zeros(n_bins), zeros(n_bins)
    q_min, q_max = 0.0, 1.0
    edges = range(q_min, q_max; length=n_bins+1)
    centers = [(edges[i]+edges[i+1])/2 for i in 1:n_bins]
    counts = zeros(n_bins)
    for q in Q_traj
        b = searchsortedlast(collect(edges), clamp(q, q_min, q_max - 1e-10))
        b = clamp(b, 1, n_bins)
        counts[b] += 1
    end
    # Smooth counts
    counts .+= 0.5   # add pseudocount to avoid log(0)
    P = counts ./ sum(counts)
    F = -T .* log.(P)
    F .-= minimum(F)
    centers, F
end

# Combined free energy profile from multiple temperatures using simple reweighting
function wham_simple(results::Vector{MonteCarlo.ReplicaResult}; n_bins::Int=30)::Tuple{Vector{Float64},Vector{Float64}}
    # Use the lowest-T replica for best resolution
    best_idx = argmin([r.T for r in results])
    r = results[best_idx]
    free_energy_profile(r.Q_traj, r.T; n_bins=n_bins)
end

function weighted_histogram(energy_traj::Vector{Float64}, T_range::AbstractRange; n_bins::Int=30)
    e_min = minimum(energy_traj) - 0.1
    e_max = maximum(energy_traj) + 0.1
    edges = range(e_min, e_max; length=n_bins+1)
    centers = [(edges[i]+edges[i+1])/2 for i in 1:n_bins]
    H = zeros(length(T_range), n_bins)
    for (ti, T) in enumerate(T_range)
        for e in energy_traj
            b = clamp(searchsortedlast(collect(edges), e), 1, n_bins)
            H[ti, b] += exp(-e / T)
        end
        s = sum(H[ti,:]) + 1e-10
        H[ti,:] ./= s
    end
    centers, H
end

end
