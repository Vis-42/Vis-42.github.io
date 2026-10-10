# s02_4 — Physical meaning of each term: curvature = KE, i = rotation not dissipation
# Slide: "Physical Meaning of Each Term"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS

# x domain for the illustration
x_ill = range(0.0, L; length=500)

# Two wavefunctions: high-k and low-k
k_hi = k0_def * 3;  k_lo = k0_def * 0.7
ψ_hi(t) = @. cos(k_hi * x_ill - ħ * k_hi^2 / (2mₑ) * t)
ψ_lo(t) = @. cos(k_lo * x_ill - ħ * k_lo^2 / (2mₑ) * t)

# Norm evolution: diffusion equation (decaying Gaussian) vs TDSE (constant norm)
# Diffusion: norm ∝ exp(-2κt)  (just an illustrative curve)
norm_diff(t) = exp(-2.5 * t / (N_FRAMES / FPS))
norm_CN(t)   = 1.0   # always 1

fig = Figure(size = RES)

# Top row: two wavefunctions side by side
ax_hi = Axis(fig[1, 1];
    title  = L"High $k$ (large curvature) $\rightarrow$ high KE",
    xlabel = L"x",
    ylabel = L"\mathrm{Re}(\psi)",
    limits = (0, L, -1.5, 1.5))
ax_lo = Axis(fig[1, 2];
    title  = L"Low $k$ (small curvature) $\rightarrow$ low KE",
    xlabel = L"x",
    ylabel = L"\mathrm{Re}(\psi)",
    limits = (0, L, -1.5, 1.5))

t_obs = Observable(0.0)
lines!(ax_hi, x_ill, @lift(ψ_hi($t_obs)); color = C_RE, linewidth = 2.2)
lines!(ax_lo, x_ill, @lift(ψ_lo($t_obs)); color = C_IM, linewidth = 2.2)

# Label curvature
text!(ax_hi, L*0.05, 1.35;
    text = L"\text{short }\lambda \rightarrow \text{large }|\partial^2\psi/\partial x^2|",
    color = C_DIM, fontsize = 13)
text!(ax_lo, L*0.05, 1.35;
    text = L"\text{long }\lambda \rightarrow \text{small }|\partial^2\psi/\partial x^2|",
    color = C_DIM, fontsize = 13)

# Bottom row: norm over time
ax_norm = Axis(fig[2, 1:2];
    title  = L"\text{Role of }i\text{: rotation (TDSE) vs decay (diffusion)}",
    xlabel = "time step",
    ylabel = L"\int|\psi|^2\,dx",
    limits = (0, N_FRAMES, 0, 1.25))

steps_obs = Observable(1)

diff_curve = @lift (1:$steps_obs, [norm_diff((s-1)/FPS) for s in 1:$steps_obs])
cn_curve   = @lift (1:$steps_obs, ones($steps_obs))

lines!(ax_norm, @lift($diff_curve[1]), @lift($diff_curve[2]);
    color = RGBf(0.90, 0.35, 0.30), linewidth = 2.5, label = L"\text{diffusion (no }i\text{)}")
lines!(ax_norm, @lift($cn_curve[1]),   @lift($cn_curve[2]);
    color = C_IM, linewidth = 2.5, linestyle = :solid, label = L"\text{TDSE (with }i\text{)}")
axislegend(ax_norm; position = :rt)

record(fig, out("s02_4_curvature_norm.mp4"), 1:N_FRAMES; framerate = FPS) do f
    t_obs[]    = (f - 1) / FPS
    steps_obs[] = f
end
println("✓  s02_4_curvature_norm.mp4")
