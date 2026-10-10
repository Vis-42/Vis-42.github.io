# s03_5 — Wavepacket spreading: σ(t) growing, norm conserved
# Slide: "Spreading in Time"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 7 * FPS
t_star   = 2mₑ * σ_def^2 / ħ   # spreading time
T_total  = 2.5 * t_star
vg       = ħ * k0_def / mₑ

# Compute explicit y limit for ψ panel
ρ_max_0 = 1.2 * maximum(@. exp(-(xs - x0_def)^2 / σ_def^2) / (sqrt(π) * σ_def))

fig = Figure(size = RES)

ax_ψ = Axis(fig[1, 1];
    title  = L"|\psi(x,t)|^2\ \text{— packet spreads at constant norm}",
    xlabel = L"x\ \text{(nm)}",
    ylabel = L"|\psi|^2",
    limits = (0, L*1e9, 0, ρ_max_0))

ax_σ = Axis(fig[1, 2];
    title  = L"\text{Width }\ \sigma(t)\ \text{vs time}",
    xlabel = L"t / t^*",
    ylabel = L"\sigma(t) / \sigma_0",
    limits = (0, 2.8, 0.9, 2.9))

# Pre-compute σ(t) curve
t_range = range(0, T_total; length=300)
σ_curve = @. σ_def * sqrt(1 + (ħ * t_range / (2mₑ * σ_def^2))^2)

t_obs = Observable(0.0)

# Analytical Gaussian envelope
function ρ_gauss(t)
    σt = σ_def * sqrt(1 + (ħ*t/(2mₑ*σ_def^2))^2)
    xc = x0_def + vg * t
    @. exp(-(xs - xc)^2 / σt^2) / (sqrt(π) * σt)
end

lines!(ax_ψ, xnm, @lift(ρ_gauss($t_obs)); color = C_PR, linewidth = 2.5)

# Norm label (stays at 1) — text positioned inside limits
text!(ax_ψ, L*1e9*0.45, ρ_max_0 * 0.90;
    text = L"\int|\psi|^2\,dx = 1\ \text{(constant)}", color = C_IM, fontsize = 13)

# σ(t) curve — reveal progressively
lines!(ax_σ, t_range ./ t_star, σ_curve ./ σ_def;
    color = (C_PR, 0.25), linewidth = 2.0)   # full curve, dim

n_shown = @lift round(Int, clamp($t_obs / T_total * 300, 1, 300))
lines!(ax_σ,
    @lift(t_range[1:$n_shown] ./ t_star),
    @lift(σ_curve[1:$n_shown] ./ σ_def);
    color = C_PR, linewidth = 2.5)

# Current point
σ_now = @lift σ_def * sqrt(1 + (ħ * $t_obs / (2mₑ * σ_def^2))^2)
scatter!(ax_σ, @lift([$t_obs / t_star]), @lift([$σ_now / σ_def]);
    color = C_PR, markersize = 12)

# t* line
vlines!(ax_σ, [1.0]; color = (C_DIM, 0.6), linestyle = :dash, linewidth = 1.5)
text!(ax_σ, 1.06, 2.75;
    text = L"t = t^*\ \text{(doubled width)}", color = C_DIM, fontsize = 12)

record(fig, out("s03_5_spreading.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * T_total
end
println("✓  s03_5_spreading.mp4")
