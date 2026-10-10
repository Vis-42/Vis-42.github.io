# s02_2 — Why ψ must be complex: interference vs incoherent sum
# Slide: "Why ψ Must Be Complex"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 7 * FPS

# Two counter-propagating Gaussian packets
vg = ħ * k0_def / mₑ
σp = σ_def * 2   # slightly wider for visual clarity

function psi1(t)
    xc = L / 5 + vg * t
    @. exp(-(xs - xc)^2 / (2σp^2)) * exp(im * k0_def * xs) + 0im
end
function psi2(t)
    xc = 4L / 5 - vg * t
    @. exp(-(xs - xc)^2 / (2σp^2)) * exp(-im * k0_def * xs) + 0im
end

T_total = (3L / 10) / vg   # time for them to meet in the middle

# Compute y-axis max for explicit limits
ψ1_0 = psi1(0.0)
ψ2_0 = psi2(0.0)
y_max = 1.15 * maximum(abs2.(ψ1_0) .+ abs2.(ψ2_0))
y_top = 3 * y_max

fig = Figure(size=RES)

ax_inc = Axis(fig[1, 1];
    title = L"Incoherent sum  $|\psi_1|^2 + |\psi_2|^2$",
    xlabel = L"x (nm)",
    ylabel = "density",
    limits = (0, L * 1e9, 0, y_top))

ax_coh = Axis(fig[1, 2];
    title = L"$|\psi_1 + \psi_2|^2$  (with phase interference)",
    xlabel = L"x (nm)",
    ylabel = "density",
    limits = (0, L * 1e9, 0, y_top))

t_obs = Observable(0.0)

ρ_inc = @lift abs2.($t_obs |> t -> psi1(t)) .+ abs2.($t_obs |> t -> psi2(t))
ρ_coh = @lift abs2.($t_obs |> t -> psi1(t) .+ psi2(t))

lines!(ax_inc, xnm, ρ_inc; color=C_PR, linewidth=2.5)
lines!(ax_coh, xnm, ρ_coh; color=C_HX, linewidth=2.5)

# annotation that fades in when they overlap
overlap_label = @lift begin
    t = $t_obs
    xc1 = L / 5 + vg * t
    xc2 = 4L / 5 - vg * t
    abs(xc1 - xc2) < 3σp ? "interference\nfringes" : ""
end
text!(ax_coh, L * 1e9 * 0.5, y_max * 1.5; text=overlap_label,
    align=(:center, :top), color=C_IM, fontsize=16)

record(fig, out("s02_2_interference.mp4"), 1:N_FRAMES; framerate=FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * T_total * 1.8
end
println("✓  s02_2_interference.mp4")
