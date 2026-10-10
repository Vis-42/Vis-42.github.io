# s03_4 — Phase velocity vs group velocity: v_g = 2 v_φ for free electron
# Slide: "Phase Velocity, Group Velocity, and the Dispersion Relation"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 8 * FPS

vφ = ħ * k0_def / (2mₑ)
vg = ħ * k0_def / mₑ   # = 2 vφ
T_total = L / (2vg)

fig = Figure(size = RES)

ax = Axis(fig[1, 1];
    title  = L"\text{Free Gaussian: phase crests at }v_\varphi,\text{ envelope at }v_g = 2v_\varphi",
    xlabel = L"x\ \text{(nm)}",
    ylabel = L"\mathrm{Re}(\psi)\ /\ |\psi|^2",
    limits = (0, L*1e9, -0.9, 1.1))

t_obs = Observable(0.0)

# Analytical: Re(ψ) and |ψ|²
function psi_re_and_ρ(t)
    σt = σ_def * sqrt(1 + (ħ*t/(2mₑ*σ_def^2))^2)
    xc = x0_def + vg * t   # envelope centre
    ρ   = @. exp(-(xs - xc)^2 / σt^2) / (sqrt(π) * σt)
    # Phase includes both group and phase velocity
    re  = @. exp(-(xs - xc)^2 / (2σt^2)) *
              cos(k0_def*xs - (ħ*k0_def^2/(2mₑ))*t) / (π^0.25 * sqrt(σt))
    re, ρ
end

re_obs = @lift psi_re_and_ρ($t_obs)[1]
ρ_obs  = @lift psi_re_and_ρ($t_obs)[2]

lines!(ax, xnm, re_obs; color = C_RE, linewidth = 2.0, label = L"\mathrm{Re}(\psi)\ \text{— phase crests}")
lines!(ax, xnm, ρ_obs;  color = C_PR, linewidth = 3.0, label = L"|\psi|^2\ \text{— envelope}")
axislegend(ax; position = :rt)

# Moving markers
xc_envelope = @lift (x0_def + vg * $t_obs) * 1e9
# Find a phase crest: cos(k₀x - ωt) = 1  ⟹  x = (ωt + 2nπ)/k₀
x_crest_ref = @lift begin
    ωt   = (ħ * k0_def^2 / (2mₑ)) * $t_obs
    n    = round(Int, (k0_def * (x0_def + vg * $t_obs) - ωt) / (2π))
    crest = (ωt + n * 2π) / k0_def
    clamp(crest * 1e9, 0.0, L*1e9)
end

vlines!(ax, xc_envelope; color = (C_PR, 0.7), linewidth = 2.0, linestyle = :solid)
vlines!(ax, x_crest_ref;  color = (C_RE, 0.6), linewidth = 1.5, linestyle = :dash)

# Velocity labels
text!(ax, L*1e9*0.03, 1.02; text = L"v_g = \hbar k_0/m", color = C_PR, fontsize = 15)
text!(ax, L*1e9*0.03, 0.87; text = L"v_\varphi = \hbar k_0/2m = v_g/2", color = C_RE, fontsize = 14)

record(fig, out("s03_4_phase_group.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * T_total
end
println("✓  s03_4_phase_group.mp4")
