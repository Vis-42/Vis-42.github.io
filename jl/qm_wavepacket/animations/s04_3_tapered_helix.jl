# s04_3 — Gaussian packet as a tapered Argand helix translating in x
# Slide: "The Gaussian Packet: A Tapered Helix"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 8 * FPS
vg      = ħ * k0_def / mₑ
ω       = ħ * k0_def^2 / (2mₑ)
T_total = L / (2vg)

# Build helix on the full domain
xs_h  = xs
xh_nm = xnm

# Fix axis limits based on initial amplitude
amp0 = 1.0 / (π^0.25 * sqrt(σ_def))
lim  = 1.4 * amp0

fig = Figure(size = RES)
ax  = Axis3(fig[1, 1];
    title     = L"\text{Gaussian wavepacket}\ \psi(x,t)\ \text{— tapered Argand helix}",
    xlabel    = L"x\ \text{(nm)}",
    ylabel    = L"\mathrm{Im}(\psi)",
    zlabel    = L"\mathrm{Re}(\psi)",
    azimuth   = π / 5,
    elevation = π / 7)

xlims!(ax, 0, L*1e9)
ylims!(ax, -lim, lim)
zlims!(ax, -lim, lim)

t_obs = Observable(0.0)

# Analytical wavefunction
function psi_helix(t)
    σt  = σ_def * sqrt(1 + (ħ*t/(2mₑ*σ_def^2))^2)
    xc  = x0_def + vg * t
    amp = @. exp(-(xs_h - xc)^2 / (2σt^2)) / (π^0.25 * sqrt(σt))
    re  = @. amp * cos(k0_def * xs_h - ω * t)
    im_ = @. amp * sin(k0_def * xs_h - ω * t)
    ρ   = @. amp^2
    re, im_, ρ
end

re_obs = @lift psi_helix($t_obs)[1]
im_obs = @lift psi_helix($t_obs)[2]
ρ_obs  = @lift psi_helix($t_obs)[3]

zeros_h = zeros(N)

# Projections on floor planes
lines!(ax, xh_nm, zeros_h, re_obs;
    color = (C_RE, 0.50), linewidth = 1.8)
lines!(ax, xh_nm, im_obs, zeros_h;
    color = (C_IM, 0.50), linewidth = 1.8)
# |ψ|² on z=0 plane (scaled to fit in [-lim, lim])
ρ_scale = lim / (amp0^2) * 0.7
lines!(ax, xh_nm, zeros_h, @lift($ρ_obs .* ρ_scale);
    color = (C_PR, 0.70), linewidth = 2.0)

# The helix itself
lines!(ax, xh_nm, im_obs, re_obs;
    color = (C_HX, 0.90), linewidth = 2.5)

# V=0 floor baseline
lines!(ax, xh_nm, zeros_h, zeros_h;
    color = (C_DIM, 0.25), linewidth = 0.8)

# λ annotation inside z limits
λ_nm = 2π / k0_def * 1e9
text!(ax, x0_def*1e9, 0.0, -lim*0.80;
    text = L"\text{pitch} = \lambda_\text{dB} = " * "$(round(λ_nm, digits=2)) nm",
    color = C_V, fontsize = 13)

# Slowly rotate the camera while packet moves
record(fig, out("s04_3_tapered_helix.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / N_FRAMES
    t_obs[]  = frac * T_total
    ax.azimuth[] = π/5 + frac * π/3   # slow camera orbit
end
println("✓  s04_3_tapered_helix.mp4")
