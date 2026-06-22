module Percolation

using Random, Statistics

export percolate, giant_component_size, susceptibility

"""
    percolate(complex, p; dimension=1, seed=42)

Bond percolation on k-simplices: each simplex of given dimension is
occupied with probability p. Returns the giant component size as fraction.
For dimension=1: edge percolation (standard). For dimension >= 2: the
vertices incident to at least one occupied higher simplex form the active set.
"""
function percolate(complex::Dict, p::Float64;
                   dimension::Int = 1, seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    N = length(complex[:vertices])

    if dimension == 1
        edges = complex[:edges]
        adj = falses(N, N)
        for (i,j) in edges
            rand(rng) < p && (adj[i,j] = adj[j,i] = true)
        end
    elseif dimension == 2
        triangles = get(complex, :triangles, [])
        adj = falses(N, N)
        for (i,j,k) in triangles
            if rand(rng) < p
                adj[i,j]=adj[j,i]=adj[i,k]=adj[k,i]=adj[j,k]=adj[k,j]=true
            end
        end
    elseif dimension == 3
        tets = get(complex, :tetrahedra, [])
        adj = falses(N, N)
        for (i,j,k,l) in tets
            if rand(rng) < p
                for (a,b) in [(i,j),(i,k),(i,l),(j,k),(j,l),(k,l)]
                    adj[a,b]=adj[b,a]=true
                end
            end
        end
    else
        return 0.0
    end

    # BFS to find components
    comp = zeros(Int, N); c = 0
    for start in 1:N
        comp[start] != 0 && continue
        c += 1; queue = [start]; comp[start] = c
        while !isempty(queue)
            v = popfirst!(queue)
            for u in 1:N
                adj[v,u] && comp[u]==0 && (comp[u]=c; push!(queue,u))
            end
        end
    end

    counts = zeros(Int, c)
    for v in 1:N; comp[v] > 0 && (counts[comp[v]] += 1); end
    isempty(counts) ? 0.0 : maximum(counts) / N
end

"""
    giant_component_size(complex, p; dimension, n_samples)

Average giant component fraction over n_samples realisations.
"""
function giant_component_size(complex::Dict, p::Float64;
                               dimension::Int = 1, n_samples::Int = 5)
    mean(percolate(complex, p; dimension=dimension, seed=s) for s in 1:n_samples)
end

"""
    susceptibility(complex, p_range; dimension, n_samples)

Compute χ(p) = N * (<s²> - <s>²) / <s> where s = giant component size.
Peak of χ signals p_c.
"""
function susceptibility(complex::Dict, p_range;
                         dimension::Int = 1, n_samples::Int = 5)
    N = length(complex[:vertices])
    chi = Float64[]
    for p in p_range
        samples = [percolate(complex, p; dimension=dimension, seed=s) for s in 1:n_samples]
        mean_s = mean(samples); mean_s2 = mean(s^2 for s in samples)
        c = mean_s > 0 ? N * (mean_s2 - mean_s^2) / mean_s : 0.0
        push!(chi, max(0.0, c))
    end
    chi
end

end
