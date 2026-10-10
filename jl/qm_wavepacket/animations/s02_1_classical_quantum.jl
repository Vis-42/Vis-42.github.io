# s02_1 — Classical trajectory vs quantum probability density
# Slide: "The Classical Picture and Its Replacement"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS   # 6 s

# Classical: free particle at constant momentum p = ħk₀
x_class(t) = x0_def + (ħ * k0_def / mₑ) * t   # x(t) in metres

# Quantum: analytical free Gaussian spreading
function ψ_free(t)
    σt = σ_def * sqrt(1 + (ħ * t / (2mₑ * σ_def^2))^2)
    xc = x0_def + (ħ * k0_def / mₑ) * t
    ρ  = @. exp(-(xs - xc)^2 / σt^2) / (sqrt(π) * σt)
    ρ
end

# Time axis: 0 → T such that packet crosses to ~2L/3
T_total = (L / 2) / (ħ * k0_def / mₑ)   # time to reach centre

# Compute y-axis max for quantum panel (use t=0 peak)
ρ_max = maximum(ψ_free(0.0)) * 1.3

fig = Figure(size = RES)

# Left: classical phase space (x,p) — particle as moving dot on horizontal line
ax_c = Axis(fig[1, 1];
    title  = L"Classical mechanics: definite $(x,\,p)$",
    xlabel = L"Position $x$ (nm)",
    ylabel = L"Momentum $p$  $(\hbar k_0$ units$)$",
    limits = (0, L*1e9, -0.5, 1.5))

hlines!(ax_c, [1.0]; color = (C_DIM, 0.4), linewidth = 1.5, linestyle = :dash)
text!(ax_c, L*1e9*0.05, 1.18; text = L"p = \hbar k_0", color = C_DIM, fontsize = 14)

t_obs = Observable(0.0)

xc_obs = @lift x_class($t_obs) * 1e9
scatter!(ax_c, xc_obs, Observable(1.0);
    color = C_RE, markersize = 18, marker = :circle)

# Right: quantum |ψ|²
ax_q = Axis(fig[1, 2];
    title  = L"Quantum mechanics: probability density $|\psi|^2$",
    xlabel = L"Position $x$ (nm)",
    ylabel = L"|\psi(x,t)|^2",
    limits = (0, L*1e9, 0, ρ_max))

ρ_obs = @lift ψ_free($t_obs)

lines!(ax_q, xnm, ρ_obs; color = C_PR, linewidth = 2.5)

# Shared time label (dynamic — keep as plain string)
t_label = @lift "t = $(round($t_obs * 1e15; digits=1)) fs"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 15)

record(fig, out("s02_1_classical_quantum.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[] = (f - 1) / (N_FRAMES - 1) * T_total
end
println("✓  s02_1_classical_quantum.mp4")
