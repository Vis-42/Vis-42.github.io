# s07 — Absorbing boundary: Gaussian damping profile α(x)
# Slide: "Boundary Conditions: Absorbing Layers"
#
# Shows the profile α(x) and contrasts it with sharp/linear alternatives.
# The actual packet being absorbed is shown via notebook screen recording.

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS

# Build the three boundary profiles for comparison
x_01 = range(0.0, 1.0; length=300)   # normalised position within the layer

α_gaussian = @. exp(-(1.0 - x_01)^2)                # C∞ profile (what we use)
α_linear   = x_01                                     # linear: discontinuous ∂α/∂x at x=1
α_sharp    = [xi > 0.5 ? 1.0 : 0.0 for xi in x_01]  # sharp step: Gibbs ringing

fig = Figure(size = RES)

ax_prof = Axis(fig[1, 1];
    title  = L"\text{Absorbing layer profiles (outer 5\% of domain, each side)}",
    xlabel = L"\text{normalised position within layer}\ (0 = \text{edge},\ 1 = \text{interior})",
    ylabel = L"\alpha\ \text{(damping factor)}",
    limits = (0, 1, -0.05, 1.15))

hlines!(ax_prof, [1.0]; color = (C_DIM, 0.35), linewidth = 1.0, linestyle = :dash)
text!(ax_prof, 0.05, 1.08; text = "interior (no damping)", color = C_DIM, fontsize = 13)

n_show = Observable(1)
lines!(ax_prof, @lift(x_01[1:$n_show]), @lift(α_sharp[1:$n_show]);
    color = RGBf(0.85, 0.35, 0.25), linewidth = 2.2, linestyle = :dot,
    label = L"\text{Sharp cutoff} \rightarrow \text{Gibbs ringing}")
lines!(ax_prof, @lift(x_01[1:$n_show]), @lift(α_linear[1:$n_show]);
    color = C_HX, linewidth = 2.2, linestyle = :dash,
    label = L"\text{Linear ramp} \rightarrow \partial\alpha/\partial x\ \text{discontinuous}")
lines!(ax_prof, @lift(x_01[1:$n_show]), @lift(α_gaussian[1:$n_show]);
    color = C_IM, linewidth = 3.0,
    label = L"\text{Gaussian}\ \alpha = \exp(-((BW-i)/BW)^2) \rightarrow C^\infty")

axislegend(ax_prof; position = :lt, labelsize = 12)

# Bottom panel: full domain showing where the layers sit
ax_dom = Axis(fig[2, 1];
    title  = L"\text{Full 20 nm domain: absorbing regions (5\% each side)}",
    xlabel = L"x\ \text{(nm)}",
    ylabel = L"\alpha(x)",
    limits = (0, L*1e9, -0.05, 1.15))

α_full = ones(N)
for i in 1:BW
    a = exp(-((BW - i) / BW)^2)
    α_full[i]         = a
    α_full[N - i + 1] = a
end

lines!(ax_dom, xnm, α_full; color = C_IM, linewidth = 2.5)
hlines!(ax_dom, [1.0]; color = (C_DIM, 0.25), linewidth = 1.0, linestyle = :dash)

bl = xnm[1:BW]; br = xnm[N-BW+1:N]
band!(ax_dom, bl, zeros(BW), α_full[1:BW];        color = (C_V, 0.15))
band!(ax_dom, br, zeros(BW), α_full[N-BW+1:N];    color = (C_V, 0.15))
text!(ax_dom, xnm[BW÷2],  0.45; text = "absorbing\n$(BW) pts", color = C_V,
    fontsize = 12, align = (:center, :center))
text!(ax_dom, xnm[N-BW÷2], 0.45; text = "absorbing\n$(BW) pts", color = C_V,
    fontsize = 12, align = (:center, :center))
text!(ax_dom, L*1e9/2, 1.07; text = "propagating region", color = C_DIM,
    fontsize = 13, align = (:center, :center))

rowsize!(fig.layout, 1, Relative(0.55))
rowsize!(fig.layout, 2, Relative(0.38))

record(fig, out("s07_absorbing.mp4"), 1:N_FRAMES; framerate = FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * length(x_01)))
end
println("✓  s07_absorbing.mp4")
