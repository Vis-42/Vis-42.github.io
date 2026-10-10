# s06_1 — Forward Euler norm grows, Backward Euler shrinks, CN stays exactly 1
# Slide: "Why Explicit and Implicit Euler Both Fail"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 6 * FPS
const N_STEPS  = 120

# Simulate forward Euler, backward Euler, and CN norm evolution
Ek_v  = ħ^2 * k0_def^2 / (2mₑ)
r_    = Ek_v * dt / ħ   # dimensionless parameter for the dominant mode

norm_FE = [1.0]
norm_BE = [1.0]
norm_CN = ones(N_STEPS + 1)

for s in 1:N_STEPS
    push!(norm_FE, norm_FE[end] * (1 + r_^2)^0.5)   # |1 + iΔtE/ħ|
    push!(norm_BE, norm_BE[end] / (1 + r_^2)^0.5)   # |1/(1 + iΔtE/ħ)|
end

steps = 0:N_STEPS
fig   = Figure(size = RES)

ax = Axis(fig[1, 1];
    title  = L"\int|\psi|^2\,dx\ \text{vs step — why Euler schemes fail}",
    xlabel = "step",
    ylabel = L"\int|\psi|^2\,dx",
    limits = (0, N_STEPS, 0.0, 2.4))

hlines!(ax, [1.0]; color = (C_DIM, 0.4), linewidth = 1.2, linestyle = :dash)
text!(ax, N_STEPS*0.55, 1.10;
    text = L"\text{exact} = 1", color = C_DIM, fontsize = 13)

n_show = Observable(1)

# Three curves revealed progressively
lines!(ax,
    @lift(steps[1:$n_show]),
    @lift(norm_FE[1:$n_show]);
    color = RGBf(0.90, 0.35, 0.25), linewidth = 2.5,
    label = L"\text{Forward Euler (norm grows)}")
lines!(ax,
    @lift(steps[1:$n_show]),
    @lift(norm_BE[1:$n_show]);
    color = C_HX, linewidth = 2.5,
    label = L"\text{Backward Euler (norm shrinks)}")
lines!(ax,
    @lift(steps[1:$n_show]),
    @lift(norm_CN[1:$n_show]);
    color = C_IM, linewidth = 3.0,
    label = L"\text{Crank-Nicolson (norm = 1 exactly)}")

axislegend(ax; position = :rc, labelsize = 13)

# Annotations at end of animation — inside y limits [0, 2.4]
text!(ax, N_STEPS*0.72, min(norm_FE[end]*0.96, 2.30);
    text = @lift($n_show >= N_STEPS ? "unstable!" : ""),
    color = RGBf(0.90, 0.35, 0.25), fontsize = 14)
text!(ax, N_STEPS*0.72, norm_BE[end] + 0.06;
    text = @lift($n_show >= N_STEPS ? "leaks probability" : ""),
    color = C_HX, fontsize = 14)

record(fig, out("s06_1_euler_fails.mp4"), 1:N_FRAMES; framerate = FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * (N_STEPS + 1)))
end
println("✓  s06_1_euler_fails.mp4")
