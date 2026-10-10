# shared/physics.jl — CN solver, wavepacket, potentials (N=500 for animation speed)

using LinearAlgebra

const ħ  = 1.054571817e-34   # J·s
const mₑ = 9.1093837015e-31  # kg

# Spatial grid
const N   = 500
const L   = 20e-9            # 20 nm domain
const dx  = L / N            # 40 pm
const xs  = collect(range(0.0, L; length=N))
const xnm = xs .* 1e9        # positions in nm

# Timestep: Fourier number r = ħΔt/(mΔx²) = 0.4
const dt = 0.4 * mₑ * dx^2 / ħ   # ≈ 5.54e-18 s

# Default packet parameters
const k0_def = 3e10          # m⁻¹
const σ_def  = 2e-10         # 0.2 nm
const x0_def = L / 3

const BW = N ÷ 20            # absorbing layer width (5% each side)

# Kinetic energy at default k₀
Ek(k0=k0_def) = ħ^2 * k0^2 / (2mₑ)

"""Normalised Gaussian wavepacket ψ(x,0)."""
function wavepacket(k0=k0_def, x0=x0_def, σ=σ_def)
    ψ = @. exp(-(xs - x0)^2 / (2σ^2)) * exp(im * k0 * xs) + 0im
    ψ ./= sqrt(sum(abs2.(ψ)) * dx)
    ψ
end

"""Build CN matrices A, B for V(x)."""
function cn_matrices(V::Vector{Float64})
    r  = im * ħ * dt / (2mₑ * dx^2)
    od = fill(-im * ħ * dt / (4mₑ * dx^2), N - 1)
    ad = ComplexF64.(1.0 .+ r .+ im .* (dt ./ (2ħ)) .* V)
    bd = ComplexF64.(1.0 .- r .- im .* (dt ./ (2ħ)) .* V)
    Tridiagonal(copy(od), ad, copy(od)), Tridiagonal(conj.(od), bd, conj.(od))
end

"""Gaussian absorbing boundary (in-place)."""
function absorb!(ψ::Vector{ComplexF64})
    @inbounds for i in 1:BW
        α = exp(-((BW - i) / BW)^2)
        ψ[i]         *= α
        ψ[N - i + 1] *= α
    end
end

"""One CN step: A ψⁿ⁺¹ = B ψⁿ, then absorb."""
function cn_step!(ψ::Vector{ComplexF64}, A, B)
    ψ .= A \ (B * ψ)
    absorb!(ψ)
end

# ── Potentials ──────────────────────────────────────────────────────────────

V_free() = zeros(Float64, N)

function V_barrier(A_amp, Ek_val=Ek(); hw_frac=0.12)
    V = zeros(N)
    hw = clamp(round(Int, hw_frac * N / 2), 1, N ÷ 4)
    ci = N ÷ 2
    V[max(1, ci-hw):min(N, ci+hw)] .= A_amp * Ek_val
    V
end

function V_well(A_amp, Ek_val=Ek(); hw_frac=0.12)
    V = zeros(N)
    hw = clamp(round(Int, hw_frac * N / 2), 1, N ÷ 4)
    ci = N ÷ 2
    V[max(1, ci-hw):min(N, ci+hw)] .= -A_amp * Ek_val
    V
end

function V_harmonic(A_amp, Ek_val=Ek(); hw_frac=0.30)
    xc     = L / 2
    half_w = hw_frac * L / 2
    ks     = A_amp < 1e-60 ? 0.0 : 2.0 * A_amp * Ek_val / half_w^2
    V      = @. 0.5 * ks * (xs - xc)^2
    clamp!(V, 0.0, 30 * max(A_amp * Ek_val, 1e-60))
    V
end

function V_double(A_amp, Ek_val=Ek(); hw_frac=0.055)
    V  = zeros(N)
    hw = clamp(round(Int, hw_frac * N / 2), 1, N ÷ 8)
    c1, c2 = N ÷ 3, 2 * (N ÷ 3)
    V[max(1, c1-hw):min(N, c1+hw)] .= A_amp * Ek_val
    V[max(1, c2-hw):min(N, c2+hw)] .= A_amp * Ek_val
    V
end

function V_step(A_amp, Ek_val=Ek())
    V = zeros(N)
    V[N÷2:end] .= A_amp * Ek_val
    V
end

# ── Simulation runner ───────────────────────────────────────────────────────

"""
Pre-run CN simulation, storing wavefunction every n_sub steps.
Returns Vector of N-element ComplexF64 arrays, one per frame.
"""
function simulate(V::Vector{Float64}; n_frames=180, n_sub=960,
                  k0=k0_def, x0=x0_def, σ=σ_def)
    ψ    = wavepacket(k0, x0, σ)
    A, B = cn_matrices(V)
    traj = Vector{Vector{ComplexF64}}(undef, n_frames)
    @inbounds for f in 1:n_frames
        for _ in 1:n_sub
            cn_step!(ψ, A, B)
        end
        traj[f] = copy(ψ)
    end
    traj
end
