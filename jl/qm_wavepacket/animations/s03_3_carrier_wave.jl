# s03_3 — Carrier wave: bare Gaussian (p=0) vs with carrier (p=ħk₀)
# Slide: "The Carrier Wave and Mean Momentum"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 7 * FPS
vg = ħ * k0_def / mₑ
T_total = (L / 3) / vg

# k-space distributions (analytical Gaussians, both width = 1/(2σ))
k_range = range(-6e10, 12e10; length=600)
ρ̃_bare = @. exp(-2k_range.^2 * σ_def^2) .* (2σ_def^2 / sqrt(π))
ρ̃_carr = @. exp(-2(k_range .- k0_def).^2 * σ_def^2) .* (2σ_def^2 / sqrt(π))
k_ymax = maximum(ρ̃_carr) * 1.25

fig = Figure(size = RES)

ax_bare = Axis(fig[1, 1];
    title  = L"\text{Bare Gaussian}  \langle p \rangle = 0",
    xlabel = L"x (nm)",
    ylabel = L"\mathrm{Re}(\psi)\ /\ |\psi|^2",
    limits = (0, L*1e9, -0.8, 1.1))

ax_carr = Axis(fig[1, 2];
    title  = L"\text{With carrier}\ e^{ik_0 x}\ \rightarrow\ \langle p \rangle = \hbar k_0",
    xlabel = L"x (nm)",
    ylabel = L"\mathrm{Re}(\psi)\ /\ |\psi|^2",
    limits = (0, L*1e9, -0.8, 1.1))

ax_ksp  = Axis(fig[2, :];
    title  = L"Momentum distribution $|\tilde{\psi}(k)|^2$",
    xlabel = L"k\ (\times 10^{10}\ \mathrm{m}^{-1})",
    ylabel = L"|\tilde{\psi}|^2",
    limits = (-6, 12, 0, k_ymax))

t_obs = Observable(0.0)

# Analytical free-particle propagation
function psi_bare(t)
    σt = σ_def * sqrt(1 + (ħ*t/(2mₑ*σ_def^2))^2)
    xc = L / 2   # bare Gaussian stays centred
    ρ  = @. exp(-(xs - xc)^2 / σt^2) / (sqrt(π) * σt)
    # Re(ψ) for bare: envelope with zero carrier
    re = @. sqrt(ρ)
    re, ρ
end

function psi_carr(t)
    σt = σ_def * sqrt(1 + (ħ*t/(2mₑ*σ_def^2))^2)
    xc = L / 3 + vg * t
    ρ  = @. exp(-(xs - xc)^2 / σt^2) / (sqrt(π) * σt)
    re = @. exp(-(xs - xc)^2 / (2*σt^2)) * cos(k0_def*(xs - xc)) / (π^0.25 * sqrt(σt))
    re, ρ
end

re_bare_obs = @lift psi_bare($t_obs)[1]
ρ_bare_obs  = @lift psi_bare($t_obs)[2]
re_carr_obs = @lift psi_carr($t_obs)[1]
ρ_carr_obs  = @lift psi_carr($t_obs)[2]

lines!(ax_bare, xnm, re_bare_obs; color = C_RE, linewidth = 2.0, label = L"\mathrm{Re}(\psi)")
lines!(ax_bare, xnm, ρ_bare_obs;  color = C_PR, linewidth = 2.5, label = L"|\psi|^2")
axislegend(ax_bare; position = :rt, labelsize = 13)

lines!(ax_carr, xnm, re_carr_obs; color = C_RE, linewidth = 2.0, label = L"\mathrm{Re}(\psi)")
lines!(ax_carr, xnm, ρ_carr_obs;  color = C_PR, linewidth = 2.5, label = L"|\psi|^2")
axislegend(ax_carr; position = :rt, labelsize = 13)

lines!(ax_ksp, k_range./1e10, ρ̃_bare; color = C_IM,   linewidth = 2.0, label = L"\text{bare } (k=0)")
lines!(ax_ksp, k_range./1e10, ρ̃_carr; color = C_HX,   linewidth = 2.0, label = L"\text{carrier } (k=k_0)")
vlines!(ax_ksp, [k0_def/1e10]; color = (C_HX, 0.5), linestyle = :dash, linewidth = 1.5)
text!(ax_ksp, k0_def/1e10 + 0.3, maximum(ρ̃_carr)*0.88; text = L"k_0", color = C_HX, fontsize = 14)
axislegend(ax_ksp; position = :rt, labelsize = 13)

record(fig, out("s03_3_carrier_wave.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * T_total
end
println("✓  s03_3_carrier_wave.mp4")
