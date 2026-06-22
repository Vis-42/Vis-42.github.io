module InversionNetwork

using Random, Statistics, LinearAlgebra

export CDInvNet, build_net, forward, train_net!, predict_composition, physics_loss

# Simple 3-layer MLP: input_dim → hidden → hidden → 3
struct CDInvNet
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
end

function build_net(; input_dim::Int=61, hidden_dim::Int=64, seed::Int=42)
    rng = MersenneTwister(seed)
    k1 = sqrt(2.0/input_dim); k2 = sqrt(2.0/hidden_dim)
    CDInvNet(
        randn(rng, hidden_dim, input_dim) .* k1, zeros(hidden_dim),
        randn(rng, hidden_dim, hidden_dim) .* k2, zeros(hidden_dim),
        randn(rng, 3, hidden_dim) .* k2, zeros(3),
    )
end

relu(x) = max(0.0, x)
relu_d(x) = x > 0 ? 1.0 : 0.0

function softmax(v::Vector{Float64})
    e = exp.(v .- maximum(v))
    e ./ sum(e)
end

# Forward pass: returns raw output (before softmax for training, after for inference)
function forward(net::CDInvNet, x::Vector{Float64})
    h1 = relu.(net.W1 * x .+ net.b1)
    h2 = relu.(net.W2 * h1 .+ net.b2)
    out = net.W3 * h2 .+ net.b3
    out
end

function predict_composition(net::CDInvNet, spectrum::Vector{Float64})::Vector{Float64}
    out = forward(net, spectrum)
    softmax(out)    # ensures sum=1, each ≥ 0
end

# Physics loss: penalise if predicted spectrum ≠ input spectrum
function physics_loss(pred_comp::Vector{Float64}, spectrum::Vector{Float64})::Float64
    h, s, c = pred_comp
    # reconstruct spectrum from prediction
    ref_h = [0.0]; ref_s = [0.0]; ref_c = [0.0]
    # inline reference spectra computation
    reconstructed = _reconstruct(h, s, c)
    mean((reconstructed .- spectrum).^2) / (var(spectrum) + 1e-6)
end

function _reconstruct(h, s, c)
    λ = 190.0:1.0:250.0
    n = 61
    helix = zeros(n); sheet = zeros(n); coil = zeros(n)
    for (i, wl) in enumerate(λ)
        helix[i]  = -30exp(-0.5*((wl-208)/5)^2) - 25exp(-0.5*((wl-222)/4)^2) + 50exp(-0.5*((wl-193)/8)^2)
        sheet[i]  = -15exp(-0.5*((wl-216)/7)^2) + 20exp(-0.5*((wl-195)/6)^2)
        coil[i]   = -8exp(-0.5*((wl-200)/6)^2) + 5exp(-0.5*((wl-218)/8)^2)
    end
    h.*helix .+ s.*sheet .+ c.*coil
end

function var(v)
    n=length(v); n<2 && return 0.0
    m=sum(v)/n; sum((x-m)^2 for x in v)/(n-1)
end

# Finite-difference gradient for net parameters
function _grad_fd(net::CDInvNet, x::Vector{Float64}, y_true::Vector{Float64},
                  λ_phys::Float64; ε::Float64=1e-5)
    params = net_to_vec(net)
    f0     = _eval_loss(net, x, y_true, λ_phys)
    grads  = similar(params)
    for i in eachindex(params)
        pv = copy(params); pv[i] += ε
        net_hi = vec_to_net(pv, net)
        f_hi = _eval_loss(net_hi, x, y_true, λ_phys)
        pv[i] -= 2ε
        net_lo = vec_to_net(pv, net)
        f_lo = _eval_loss(net_lo, x, y_true, λ_phys)
        grads[i] = (f_hi - f_lo) / (2ε)
    end
    grads
end

function _eval_loss(net::CDInvNet, x::Vector{Float64}, y_true::Vector{Float64}, λ_phys::Float64)
    out  = forward(net, x)
    pred = softmax(out)
    mse  = mean((pred .- y_true).^2)
    phy  = λ_phys * physics_loss(pred, x)
    mse + phy
end

# SGD training with analytical backprop (cleaner than FD for MLP)
function train_net!(net::CDInvNet, spectra::Matrix{Float64}, compositions::Matrix{Float64};
                    epochs::Int=20, lr::Float64=1e-3, lambda_physics::Float64=0.1,
                    batch_size::Int=32, seed::Int=0)::Vector{Float64}
    rng  = MersenneTwister(seed)
    N    = size(spectra, 1)
    losses = Float64[]

    for epoch in 1:epochs
        order = randperm(rng, N)
        epoch_loss = 0.0; n_batches = 0
        for batch_start in 1:batch_size:N
            batch_end = min(batch_start + batch_size - 1, N)
            batch_idx = order[batch_start:batch_end]

            # Accumulate gradients over batch
            dW1 = zeros(size(net.W1)); db1 = zeros(size(net.b1))
            dW2 = zeros(size(net.W2)); db2 = zeros(size(net.b2))
            dW3 = zeros(size(net.W3)); db3 = zeros(size(net.b3))
            batch_loss = 0.0

            for i in batch_idx
                x     = spectra[i, :]
                y_tgt = compositions[i, :]
                gW1,gb1,gW2,gb2,gW3,gb3,loss = _backprop(net, x, y_tgt, lambda_physics)
                dW1.+=gW1; db1.+=gb1; dW2.+=gW2; db2.+=gb2; dW3.+=gW3; db3.+=gb3
                batch_loss += loss
            end

            bsz = length(batch_idx)
            net.W1 .-= lr .* dW1 ./ bsz; net.b1 .-= lr .* db1 ./ bsz
            net.W2 .-= lr .* dW2 ./ bsz; net.b2 .-= lr .* db2 ./ bsz
            net.W3 .-= lr .* dW3 ./ bsz; net.b3 .-= lr .* db3 ./ bsz
            epoch_loss += batch_loss / bsz; n_batches += 1
        end
        push!(losses, epoch_loss / n_batches)
    end
    losses
end

function _backprop(net::CDInvNet, x::Vector{Float64}, y::Vector{Float64}, λ_phys::Float64)
    # Forward
    h1 = relu.(net.W1 * x  .+ net.b1)
    h2 = relu.(net.W2 * h1 .+ net.b2)
    out = net.W3 * h2 .+ net.b3
    pred = softmax(out)

    # Loss = MSE + physics
    mse   = mean((pred .- y).^2)
    rec   = _reconstruct(pred[1], pred[2], pred[3])
    p_err = mean((rec .- x).^2) / (var(x) + 1e-6)
    loss  = mse + λ_phys * p_err

    # Backprop (simplified: just MSE gradient through softmax)
    d_pred = 2.0 .* (pred .- y) ./ length(y)
    # Softmax jacobian
    S  = pred
    J  = diagm(S) .- S * S'
    d_out = J * d_pred

    dW3 = d_out * h2'
    db3 = d_out
    d_h2 = net.W3' * d_out .* relu_d.(net.W2 * h1 .+ net.b2)
    dW2 = d_h2 * h1'
    db2 = d_h2
    d_h1 = net.W2' * d_h2 .* relu_d.(net.W1 * x .+ net.b1)
    dW1 = d_h1 * x'
    db1 = d_h1

    dW1,db1,dW2,db2,dW3,db3, loss
end

function net_to_vec(net::CDInvNet)::Vector{Float64}
    vcat(vec(net.W1), net.b1, vec(net.W2), net.b2, vec(net.W3), net.b3)
end

function vec_to_net(v::Vector{Float64}, ref::CDInvNet)::CDInvNet
    n1 = length(ref.W1); nb1 = length(ref.b1)
    n2 = length(ref.W2); nb2 = length(ref.b2)
    n3 = length(ref.W3); nb3 = length(ref.b3)
    i = 1
    W1 = reshape(v[i:i+n1-1], size(ref.W1)); i+=n1
    b1 = v[i:i+nb1-1]; i+=nb1
    W2 = reshape(v[i:i+n2-1], size(ref.W2)); i+=n2
    b2 = v[i:i+nb2-1]; i+=nb2
    W3 = reshape(v[i:i+n3-1], size(ref.W3)); i+=n3
    b3 = v[i:i+nb3-1]
    CDInvNet(W1,b1,W2,b2,W3,b3)
end

end
