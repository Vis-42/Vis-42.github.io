module DataGeneration

using Random

export make_spiral, make_checkerboard, make_xor

"""
    make_spiral(n_per_class; noise=0.1, seed=42)

Two-class spiral dataset. Returns (X, y) where X is N×2, y is 1-indexed class labels.
"""
function make_spiral(n_per_class::Int; noise::Float64 = 0.1, seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    n   = n_per_class * 2
    X   = zeros(n, 2); y = zeros(Int, n)
    for c in 0:1
        ix = c*n_per_class+1:(c+1)*n_per_class
        r  = range(0.0, 1.0, length=n_per_class)
        t  = range(c*4, (c+2)*4, length=n_per_class) .+ randn(rng, n_per_class) .* noise
        X[ix, 1] .= r .* sin.(t)
        X[ix, 2] .= r .* cos.(t)
        y[ix]    .= c + 1
    end
    X, y
end

"""
    make_checkerboard(N; noise=0.05, seed=42)

2×2 checkerboard classification dataset.
"""
function make_checkerboard(N::Int; noise::Float64 = 0.05, seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    X = randn(rng, N, 2); y = zeros(Int, N)
    for i in 1:N
        xi = floor(Int, X[i,1]) + floor(Int, X[i,2])
        y[i] = (xi % 2 == 0) ? 1 : 2
    end
    X, y
end

"""
    make_xor(N; noise=0.1, seed=42)

XOR binary classification: quadrant-based.
"""
function make_xor(N::Int; noise::Float64 = 0.1, seed::Int = 42)
    rng = Random.MersenneTwister(seed)
    X = 2rand(rng, N, 2) .- 1
    X .+= randn(rng, N, 2) .* noise
    y = [((X[i,1] > 0) == (X[i,2] > 0)) ? 1 : 2 for i in 1:N]
    X, y
end

end
