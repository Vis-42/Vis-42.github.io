module PhaseDiagram

using Statistics

export sweep_percolation, PercolationResult, estimate_pc

struct PercolationResult
    p_range::Vector{Float64}
    S::Vector{Float64}       # giant component fraction
    chi::Vector{Float64}     # susceptibility
    dimension::Int
    N::Int
end

"""
    sweep_percolation(complex, p_range; dimension, n_samples)

Sweep over bond occupation probabilities, compute S(p) and χ(p).
"""
function sweep_percolation(complex::Dict, p_range;
                            dimension::Int = 1, n_samples::Int = 5)
    include(joinpath(@__DIR__, "Percolation.jl")); using .Percolation
    N = length(complex[:vertices])
    S   = [giant_component_size(complex, p; dimension=dimension, n_samples=n_samples) for p in p_range]
    chi = susceptibility(complex, p_range; dimension=dimension, n_samples=n_samples)
    PercolationResult(collect(Float64, p_range), S, chi, dimension, N)
end

"""
    estimate_pc(result)

Estimate p_c as location of maximum susceptibility.
"""
function estimate_pc(result::PercolationResult)
    idx = argmax(result.chi)
    result.p_range[idx]
end

end
