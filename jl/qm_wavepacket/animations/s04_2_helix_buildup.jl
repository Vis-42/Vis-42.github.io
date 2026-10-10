# s04_2 — Building the Argand helix step by step: 2D → 3D
# Slide: "Why a Plane Wave Traces a Helix"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 9 * FPS

# Show 4 de Broglie wavelengths
n_λ  = 4
λ_dB = 2π / k0_def
x_h  = range(0.0, n_λ * λ_dB; length=400)
xh_nm = x_h .* 1e9

fig = Figure(size = RES)
ax  = Axis3(fig[1, 1];
    title    = L"\psi = \exp(i(k_0 x - \omega t))\ \text{— the Argand helix}",
    xlabel   = L"x\ \text{(nm)}",
    ylabel   = L"\mathrm{Im}(\psi)",
    zlabel   = L"\mathrm{Re}(\psi)",
    azimuth  = π / 6,
    elevation = π / 8)

xlims!(ax, 0, n_λ * λ_dB * 1e9)
ylims!(ax, -1.3, 1.3)
zlims!(ax, -1.3, 1.3)

t_obs   = Observable(0.0)
stage   = Observable(1)     # 1=x-axis, 2=Re added, 3=Im added, 4=helix

ω = ħ * k0_def^2 / (2mₑ)

# Stage-dependent curves
re_curve = @lift begin
    t = $t_obs
    cos.(k0_def .* x_h .- ω .* t)
end
im_curve = @lift begin
    t = $t_obs
    sin.(k0_def .* x_h .- ω .* t)
end

zeros_h = zeros(length(x_h))

# Always show x-axis as a thin reference
lines!(ax, xh_nm, zeros_h, zeros_h;
    color = (C_DIM, 0.4), linewidth = 1.0)

# Re(ψ) — visible after stage 2
re_curve_vis = @lift $stage >= 2 ? re_curve[] : zeros_h
lines!(ax, xh_nm, zeros_h, re_curve_vis;
    color = C_RE, linewidth = 2.5)

# Im(ψ) — visible after stage 3
im_curve_vis = @lift $stage >= 3 ? im_curve[] : zeros_h
lines!(ax, xh_nm, im_curve_vis, zeros_h;
    color = C_IM, linewidth = 2.5)

# Helix — visible after stage 4
hx_re_vis = @lift $stage >= 4 ? re_curve[] : zeros_h
hx_im_vis = @lift $stage >= 4 ? im_curve[] : zeros_h
lines!(ax, xh_nm, hx_im_vis, hx_re_vis;
    color = C_HX, linewidth = 2.8)

# λ_dB annotation (always visible)
lines!(ax, [0.0, λ_dB*1e9], zeros_h[1:2], fill(-1.25, 2);
    color = (C_V, 0.7), linewidth = 2.0)
text!(ax, λ_dB*1e9/2, 0.0, -1.30;
    text = L"\lambda_\text{dB} = 2\pi/k_0", color = C_V, fontsize = 13)

# Stage transitions: 2s each
record(fig, out("s04_2_helix_buildup.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac = (f - 1) / N_FRAMES
    stage[] = clamp(floor(Int, frac * 5) + 1, 1, 4)
    t_obs[] = frac * 6π / ω   # helix slowly rotates
end
println("✓  s04_2_helix_buildup.mp4")
