module SimplicialComplex

using Random

export SimplicialConfig, generate_clique_complex, generate_hypergraph,
       connected_components_1skeleton, simplices_of_dim

Base.@kwdef struct SimplicialConfig
    N::Int         = 200
    p_edge::Float64 = 0.05
    dim_max::Int   = 3
end

"""
    generate_clique_complex(N, p_edge; dim_max=3, seed=42)

Generate an Erdős-Rényi clique complex on N vertices:
1. Generate G(N, p_edge) random graph
2. Find all k-cliques (k-simplices) up to dim_max

Returns Dict with keys :vertices, :edges, :triangles, :tetrahedra (as lists of tuples).
"""
function generate_clique_complex(N::Int, p_edge::Float64;
                                  dim_max::Int = 3, seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    adj = falses(N, N)
    edges = Tuple{Int,Int}[]

    for i in 1:N, j in i+1:N
        if rand(rng) < p_edge
            adj[i,j] = adj[j,i] = true
            push!(edges, (i, j))
        end
    end

    triangles = Tuple{Int,Int,Int}[]
    if dim_max >= 2
        for (i,j) in edges, k in j+1:N
            adj[i,k] && adj[j,k] && push!(triangles, (i,j,k))
        end
    end

    tetrahedra = Tuple{Int,Int,Int,Int}[]
    if dim_max >= 3
        for (i,j,k) in triangles, l in k+1:N
            adj[i,l] && adj[j,l] && adj[k,l] && push!(tetrahedra, (i,j,k,l))
        end
    end

    Dict(
        :vertices    => collect(1:N),
        :edges       => edges,
        :triangles   => triangles,
        :tetrahedra  => tetrahedra,
        :adj         => adj,
    )
end

"""
    generate_hypergraph(N, n_hyperedges, k; seed=42)

Generate a random hypergraph with n_hyperedges k-uniform hyperedges on N vertices.
"""
function generate_hypergraph(N::Int, n_hyperedges::Int, k::Int; seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    hyperedges = [sort(rand(rng, 1:N, k)) for _ in 1:n_hyperedges]
    Dict(:vertices => collect(1:N), :hyperedges => hyperedges, :k => k)
end

"""
    connected_components_1skeleton(adj)

Return connected components of the 1-skeleton (standard graph) via BFS.
"""
function connected_components_1skeleton(adj::BitMatrix)
    N = size(adj, 1)
    comp = zeros(Int, N)
    c = 0
    for start in 1:N
        comp[start] != 0 && continue
        c += 1
        queue = [start]
        comp[start] = c
        while !isempty(queue)
            v = popfirst!(queue)
            for u in 1:N
                adj[v,u] && comp[u] == 0 && (comp[u] = c; push!(queue, u))
            end
        end
    end
    comp, c
end

"""
    simplices_of_dim(complex, dim)

Return the simplices of a given dimension (0=vertices, 1=edges, 2=triangles, 3=tetra).
"""
function simplices_of_dim(complex::Dict, dim::Int)
    dim == 0 && return complex[:vertices]
    dim == 1 && return complex[:edges]
    dim == 2 && return get(complex, :triangles, [])
    dim == 3 && return get(complex, :tetrahedra, [])
    []
end

end
