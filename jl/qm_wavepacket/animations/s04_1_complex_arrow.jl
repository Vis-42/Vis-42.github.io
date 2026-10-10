# s04_1 — Complex arrow rotating in (Re ψ, Im ψ) plane at a fixed position x
# Slide: "The Complex Arrow at Each Position"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 5 * FPS

fig = Figure(size = RES)

# Left: complex plane (Re, Im) with rotating arrow
ax_c = Axis(fig[1, 1];
    title  = L"\psi(x_0,\,t)\ \text{in the complex plane}",
    xlabel = L"\mathrm{Re}(\psi)",
    ylabel = L"\mathrm{Im}(\psi)",
    aspect = DataAspect(),
    limits = (-1.4, 1.4, -1.4, 1.4))

# Unit circle for reference
θs = range(0, 2π; length=200)
lines!(ax_c, cos.(θs), sin.(θs); color = (C_DIM, 0.3), linewidth = 1.2)
hlines!(ax_c, [0.0]; color = (C_DIM, 0.2), linewidth = 0.8)
vlines!(ax_c, [0.0]; color = (C_DIM, 0.2), linewidth = 0.8)

# Annotation labels
text!(ax_c,  1.02, 0.08; text = L"\mathrm{Re}(\psi)", color = C_DIM, fontsize = 12)
text!(ax_c, -0.18, 1.25; text = L"\mathrm{Im}(\psi)", color = C_DIM, fontsize = 12)

t_obs = Observable(0.0)
A     = 0.9   # amplitude of the arrow

# Probability density |ψ|² = A² (constant for plane wave)
θ_obs  = @lift k0_def * 5e-10 - (1.054e-34 * k0_def^2 / (2 * 9.109e-31)) * $t_obs
re_obs = @lift A * cos($θ_obs)
im_obs = @lift A * sin($θ_obs)

# Arrow: origin → (Re, Im) drawn as line shaft + tip marker
shaft_obs = @lift [Point2f(0.0, 0.0), Point2f($re_obs, $im_obs)]
lines!(ax_c, shaft_obs; color = C_HX, linewidth = 3.0)
scatter!(ax_c, @lift([$re_obs]), @lift([$im_obs]);
    color = C_HX, markersize = 18, marker = :circle)

# |ψ| = amplitude label (radius) — inside limits
text!(ax_c, -1.35, -1.20; text = L"|\psi| = A\ \text{(arrow length = }\sqrt{\text{probability}}\text{)}",
    color = C_TXT, fontsize = 12)
# Dynamic θ label — plain string with interpolation
θ_label = @lift "arg(psi) = $(round(mod($θ_obs, 2π), digits=2)) rad"
text!(ax_c, -1.35, -1.05; text = θ_label, color = C_DIM, fontsize = 12)

# Right: show how θ encodes local momentum (k increases → faster rotation)
ax_k = Axis(fig[1, 2];
    title  = L"\text{Phase }\ \theta(x)\ \text{for three }k\text{ values}",
    xlabel = L"x\ \text{(nm)}",
    ylabel = L"\text{phase }\ \theta\ \text{(rad)}",
    limits = (0, 5.0, -2π, 2π))

x_short = range(0.0, 5e-9; length=300)
for (k_val, col, lbl) in [(k0_def*0.5, C_IM, L"\text{low }k"),
                           (k0_def,     C_RE, L"k_0"),
                           (k0_def*2.0, C_HX, L"\text{high }k")]
    lines!(ax_k, x_short.*1e9, k_val .* x_short;
        color = col, linewidth = 2.0, label = lbl)
end
axislegend(ax_k; position = :rb, labelsize = 13)
# Annotation inside axis limits (y in [-2π, 2π] ≈ [-6.28, 6.28])
text!(ax_k, 0.2, 5.8;
    text = L"\text{steeper slope = shorter }\lambda\text{ = higher }k",
    color = C_DIM, fontsize = 12)

record(fig, out("s04_1_complex_arrow.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / N_FRAMES * 4π / (1.054e-34 * k0_def^2 / (2 * 9.109e-31))
end
println("✓  s04_1_complex_arrow.mp4")
