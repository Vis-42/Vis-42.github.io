# s02_3 — Plane wave exp(i(kx−ωt)): phase fronts and de Broglie wavelength
# Slide: "From de Broglie to the TDSE"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 5 * FPS

k = k0_def
ω = ħ * k^2 / (2mₑ)
λ = 2π / k   # de Broglie wavelength ≈ 2.09 nm

# Show just 3 full de Broglie wavelengths
x_domain = range(0.0, 3λ; length=600)
xnm_d = x_domain .* 1e9

fig = Figure(size=RES)

ax = Axis(fig[1, 1];
    title  = L"\psi = \exp(i(k_0 x - \omega t))  \text{  — plane wave}",
    xlabel = L"x (nm)",
    ylabel = "amplitude",
    limits = (0, 3λ * 1e9, -2, 2))

t_obs = Observable(0.0)

re_obs = @lift cos.(k .* x_domain .- ω .* $t_obs)
im_obs = @lift sin.(k .* x_domain .- ω .* $t_obs)

lines!(ax, xnm_d, re_obs; color=C_RE, linewidth=2.5, label=L"\mathrm{Re}(\psi)")
lines!(ax, xnm_d, im_obs; color=C_IM, linewidth=2.5, label=L"\mathrm{Im}(\psi)")
axislegend(ax; position=:rt)

# Mark one wavelength with a horizontal bracket annotation
hlines!(ax, [0.0]; color=(C_DIM, 0.3), linewidth=1)

# λ_dB annotation: horizontal line with end ticks at y = -1.35
lines!(ax, [0.0, λ * 1e9], [-1.35, -1.35]; color=C_V, linewidth=2.0)
lines!(ax, [0.0, 0.0], [-1.30, -1.40]; color=C_V, linewidth=2.0)
lines!(ax, [λ * 1e9, λ * 1e9], [-1.30, -1.40]; color=C_V, linewidth=2.0)
# Dynamic value in plain string is fine; static label uses LaTeX
text!(ax, λ * 1e9 / 2, -1.47;
    text = L"\lambda_\text{dB} = 2\pi/k_0 \approx " * "$(round(λ*1e9, digits=2)) nm",
    align=(:center, :top), color=C_V, fontsize=14)

# phase velocity label (static — no observable needed)
text!(ax, 3λ * 1e9 * 0.5, 1.9;
    text = "v_phi = omega/k = $(round(ω/k, sigdigits=3)) m/s",
    align=(:center, :top), color=C_DIM, fontsize=13)

T_period = 2π / ω

record(fig, out("s02_3_plane_wave.mp4"), 1:N_FRAMES; framerate=FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * 2T_period
end
println("✓  s02_3_plane_wave.mp4")
