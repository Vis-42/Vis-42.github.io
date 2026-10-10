# s06_2 — Möbius transform (1-iθ)/(1+iθ) maps real axis to unit circle
# Slide: "The Crank-Nicolson Scheme"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS

fig = Figure(size = RES)

# Left: θ on the real axis → complex plane trace
ax_c = Axis(fig[1, 1];
    title  = L"u_k = \frac{1 - i\theta}{1 + i\theta}\ \text{maps real }\theta\text{ to unit circle}",
    xlabel = L"\mathrm{Re}(u_k)",
    ylabel = L"\mathrm{Im}(u_k)",
    aspect = DataAspect(),
    limits = (-1.5, 1.5, -1.5, 1.5))

# Full unit circle for reference
θs = range(0, 2π; length=300)
lines!(ax_c, cos.(θs), sin.(θs); color = (C_DIM, 0.25), linewidth = 1.2)
hlines!(ax_c, [0.0]; color = (C_DIM, 0.15), linewidth = 0.8)
vlines!(ax_c, [0.0]; color = (C_DIM, 0.15), linewidth = 0.8)

θ_range = range(-6.0, 6.0; length=300)
uk_vals  = @. (1 - im*θ_range) / (1 + im*θ_range)

n_obs  = Observable(1)
trace  = @lift uk_vals[1:$n_obs]
θ_now  = @lift θ_range[$n_obs]
uk_now = @lift uk_vals[$n_obs]

lines!(ax_c, @lift(real.($trace)), @lift(imag.($trace));
    color = C_HX, linewidth = 2.5)
scatter!(ax_c, @lift([real($uk_now)]), @lift([imag($uk_now)]);
    color = C_HX, markersize = 14)

# Dynamic labels — plain strings with interpolation
text!(ax_c, -1.45, 1.35;
    text = @lift("theta = DtE_k/2hbar = $( round($θ_now, digits=2) )"),
    color = C_TXT, fontsize = 13)
text!(ax_c, -1.45, 1.15;
    text = @lift("|u_k| = $( round(abs($uk_now), digits=4) )"),
    color = C_IM, fontsize = 13)

# Right: |u_k| vs θ — always = 1
ax_r = Axis(fig[1, 2];
    title  = L"|u_k|^2 = 1\ \text{for all real }\theta\ \text{— norm exactly preserved}",
    xlabel = L"\theta = \Delta t E_k / 2\hbar",
    ylabel = L"|u_k|^2",
    limits = (-6.5, 6.5, 0.0, 1.5))

lines!(ax_r, θ_range, ones(length(θ_range));
    color = C_IM, linewidth = 2.8)
hlines!(ax_r, [1.0]; color = (C_DIM, 0.35), linewidth = 1.0, linestyle = :dash)

θ_now_r = @lift [θ_range[$n_obs]]
vlines!(ax_r, θ_now_r; color = (C_HX, 0.7), linewidth = 2.0)

# Static annotations inside limits y ∈ [0, 1.5]
text!(ax_r, -6.3, 1.43;
    text = L"|1 + i\theta|^2 = 1 + \theta^2 > 1\ \text{(FE, unstable)}",
    color = RGBf(0.90, 0.35, 0.25), fontsize = 12)
text!(ax_r, -6.3, 1.30;
    text = L"1/|1 + i\theta|^2 < 1\ \text{(BE, dissipative)}",
    color = C_HX, fontsize = 12)
text!(ax_r, -6.3, 1.17;
    text = L"\left|\frac{1-i\theta}{1+i\theta}\right|^2 = 1\ \text{(CN, exact)}",
    color = C_IM, fontsize = 12)

record(fig, out("s06_2_unit_circle.mp4"), 1:N_FRAMES; framerate = FPS) do f
    n_obs[] = max(1, round(Int, (f / N_FRAMES) * length(θ_range)))
end
println("✓  s06_2_unit_circle.mp4")
