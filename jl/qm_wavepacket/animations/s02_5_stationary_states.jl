# s02_5 — Stationary states: static |ψ|², rotating phase; superposition shows dynamics
# Slide: "Stationary States and Why They Are Not Enough"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 7 * FPS

# Infinite square well eigenfunctions on [0,L]
E1 = (ħ * π / L)^2 / (2mₑ)
E2 = 4E1   # E₂ = 4E₁

φ1(x) = sqrt(2/L) .* sin.(π .* x ./ L)
φ2(x) = sqrt(2/L) .* sin.(2π .* x ./ L)

ψ1(x, t) = φ1(x) .* exp.(-im .* E1 .* t ./ ħ)
ψ2(x, t) = φ2(x) .* exp.(-im .* E2 .* t ./ ħ)
ψ_super(x, t) = (ψ1(x,t) .+ ψ2(x,t)) ./ sqrt(2)

x_inf = range(0.0, L; length=500)

# Compute explicit y-limits
ymax_sup = 1.3 * 2 * (2/L)   # max density of superposition

fig = Figure(size = RES)

# Left: ψ₁ — static |ψ|² but rotating (Re, Im)
ax1 = Axis(fig[1, 1];
    title  = L"Energy eigenstate $\psi_1$  —  $|\psi_1|^2$ is static",
    xlabel = L"x (nm)",
    ylabel = "amplitude",
    limits = (0, L*1e9, -1.8, 1.8))

# Right: superposition — |ψ|² oscillates
ax2 = Axis(fig[1, 2];
    title  = L"Superposition $(\psi_1+\psi_2)/\sqrt{2}$  —  $|\psi|^2$ oscillates",
    xlabel = L"x (nm)",
    ylabel = "density",
    limits = (0, L*1e9, 0, ymax_sup))

t_obs = Observable(0.0)

# Left panel: Re, Im, and |ψ|²  (|ψ₁|² stays constant)
re1 = @lift real.(ψ1(x_inf, $t_obs))
im1 = @lift imag.(ψ1(x_inf, $t_obs))
ρ1  = @lift abs2.(ψ1(x_inf, $t_obs))

lines!(ax1, x_inf.*1e9, re1; color = C_RE, linewidth = 2.0, label = L"\mathrm{Re}(\psi_1)")
lines!(ax1, x_inf.*1e9, im1; color = C_IM, linewidth = 2.0, label = L"\mathrm{Im}(\psi_1)")
lines!(ax1, x_inf.*1e9, ρ1;  color = C_PR, linewidth = 2.8, label = L"|\psi_1|^2")
axislegend(ax1; position = :rt, labelsize = 13)

# Right panel: |ψ_super|²
ρ_sup = @lift abs2.(ψ_super(x_inf, $t_obs))
lines!(ax2, x_inf.*1e9, ρ_sup; color = C_HX, linewidth = 2.5)

T_beat = 2π * ħ / (E2 - E1)   # beat period

record(fig, out("s02_5_stationary_states.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * 2T_beat
end
println("✓  s02_5_stationary_states.mp4")
