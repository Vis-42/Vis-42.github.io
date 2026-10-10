# s06_4 — Exact vs CN phase: 2arctan(ωΔt/2) vs ωΔt, marking r=0.4 operating point
# Slide: "Timestep: Phase Accuracy, Not Stability"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 5 * FPS

# Dimensionless parameter: ξ = ω Δt (phase per step)
# For dominant mode k=k₀: ξ_dominant = ħk₀²Δt/(2m)
ξ_dom = ħ * k0_def^2 * dt / (2mₑ)   # operating point
# For Nyquist mode k=π/Δx:
ξ_ny  = ħ * (π/dx)^2  * dt / (2mₑ)  # should be ≈ 5 × ξ_dom with r=0.4

ξ_range = range(0, 4.0; length=400)

exact_phase = ξ_range                              # ωΔt
cn_phase    = 2 .* atan.(ξ_range ./ 2)            # 2arctan(ωΔt/2)
err_pct     = abs.(cn_phase .- exact_phase) ./ max.(exact_phase, 1e-10) .* 100

# Compute error axis upper limit
err_max = ceil(maximum(err_pct[2:end]) * 1.2)

fig = Figure(size = RES)

ax_ph = Axis(fig[1, 1];
    title  = L"\text{CN phase accuracy vs exact phase per step}",
    xlabel = L"\xi = \omega\Delta t\ \text{(exact phase per step)}",
    ylabel = L"\text{phase (rad)}",
    limits = (0, 4, 0, 4.5))

ax_er = Axis(fig[2, 1];
    title  = L"\text{Phase error (\%)}",
    xlabel = L"\xi = \omega\Delta t",
    ylabel = L"\text{error (\%)}",
    limits = (0, 4, 0, err_max))

n_show = Observable(1)

lines!(ax_ph, @lift(ξ_range[1:$n_show]),  @lift(exact_phase[1:$n_show]);
    color = C_TXT, linewidth = 2.5, label = L"\text{Exact}\ \varphi = \omega\Delta t")
lines!(ax_ph, @lift(ξ_range[1:$n_show]),  @lift(cn_phase[1:$n_show]);
    color = C_IM,  linewidth = 2.5, linestyle = :dash,
    label = L"\text{CN}\ \varphi = 2\arctan(\omega\Delta t/2)")
axislegend(ax_ph; position = :lt)

lines!(ax_er, @lift(ξ_range[1:$n_show]),  @lift(err_pct[1:$n_show]);
    color = RGBf(0.90, 0.35, 0.25), linewidth = 2.2)

# Operating point markers — text inside limits
vlines!(ax_ph, [ξ_dom]; color = (C_PR, 0.7), linestyle = :dot, linewidth = 1.8)
vlines!(ax_er, [ξ_dom]; color = (C_PR, 0.7), linestyle = :dot, linewidth = 1.8)
text!(ax_ph, ξ_dom + 0.05, 4.3;
    text = "k0 mode\nxi = $(round(ξ_dom, sigdigits=2))",
    color = C_PR, fontsize = 11)

vlines!(ax_ph, [ξ_ny]; color = (C_V, 0.5), linestyle = :dot, linewidth = 1.5)
vlines!(ax_er, [ξ_ny]; color = (C_V, 0.5), linestyle = :dot, linewidth = 1.5)
text!(ax_ph, ξ_ny + 0.05, 2.0;
    text = "Nyquist mode\nxi = $(round(ξ_ny, sigdigits=2))",
    color = C_V, fontsize = 11)

hlines!(ax_er, [1.0]; color = (C_DIM, 0.4), linestyle = :dash, linewidth = 1.2)
text!(ax_er, 0.1, 1.0 + err_max * 0.05;
    text = L"\text{1\% error threshold}", color = C_DIM, fontsize = 11)

record(fig, out("s06_4_phase_accuracy.mp4"), 1:N_FRAMES; framerate = FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * length(ξ_range)))
end
println("✓  s06_4_phase_accuracy.mp4")
