# s03_1 — Gaussian as minimum-uncertainty state: position/momentum Fourier pair
# Slide: "The Gaussian as the Unique Minimum-Uncertainty State"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS

# σ sweeps from narrow to wide
σ_min = 0.5 * σ_def
σ_max = 3.0 * σ_def

# k-range for momentum space panel
k_range = range(-6e10, 6e10; length=500)

# Compute y-limits at σ_min (narrowest → tallest peaks)
ρ_x_max = 1.0 / (sqrt(π) * σ_min) * 1.25
ρ_k_max = (2 * σ_max^2 / sqrt(π)) * 1.25   # widest σ → tallest k-space peak

fig = Figure(size = RES)

ax_x = Axis(fig[1, 1];
    title  = L"Position space  $|\psi(x)|^2$",
    xlabel = L"x (nm)",
    ylabel = L"|\psi|^2",
    limits = (0, L*1e9, 0, ρ_x_max))

ax_k = Axis(fig[1, 2];
    title  = L"Momentum space  $|\tilde{\psi}(k)|^2$",
    xlabel = L"k  (\times 10^{10}\ \mathrm{m}^{-1})",
    ylabel = L"|\tilde{\psi}|^2",
    limits = (-6, 6, 0, ρ_k_max))

σ_obs = Observable(σ_def)

# Position-space profile (centred Gaussian)
ρ_x = @lift @. exp(-(xs - L/2)^2 / $σ_obs^2) / (sqrt(π) * $σ_obs)

# Momentum-space profile (Gaussian, FT of Gaussian is Gaussian)
ρ_k = @lift @. exp(-2k_range.^2 .* $σ_obs^2) .* (2 * $σ_obs^2 / sqrt(π))

lines!(ax_x, xnm, ρ_x; color = C_PR, linewidth = 2.5)
lines!(ax_k, k_range ./ 1e10, ρ_k; color = C_RE, linewidth = 2.5)

# Dynamic labels (plain strings — contain interpolated values)
Δx_lbl  = @lift "Dx = $(round($σ_obs / sqrt(2) * 1e10, digits=1)) pm"
Δk_lbl  = @lift "Dk = $(round(1/(2*$σ_obs) / 1e10, digits=2)) x10^10 m^-1"

text!(ax_x, L*1e9*0.05, ρ_x_max * 0.88; text = Δx_lbl,
    color = C_PR, fontsize = 14)
text!(ax_k, -5.8, ρ_k_max * 0.88; text = Δk_lbl,
    color = C_RE, fontsize = 14)

# Static bottom label
Label(fig[2, :], L"\Delta x \cdot \Delta k = 0.50 = \hbar/2  \checkmark";
    color = C_IM, fontsize = 18, tellwidth = false)

# σ annotation lines on position panel
vlines!(ax_x, @lift [L*1e9/2 - $σ_obs/sqrt(2)*1e9,
                      L*1e9/2 + $σ_obs/sqrt(2)*1e9];
    color = (C_PR, 0.5), linewidth = 1.5, linestyle = :dash)

record(fig, out("s03_1_min_uncertainty.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac = (f - 1) / (N_FRAMES - 1)
    # σ sweeps: σ_min → σ_max → σ_min
    phase = frac * 2π
    σ_obs[] = σ_def + (σ_max - σ_min)/2 * sin(phase) * 0.85
end
println("✓  s03_1_min_uncertainty.mp4")
