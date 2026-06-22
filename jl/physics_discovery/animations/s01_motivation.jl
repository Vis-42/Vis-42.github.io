# s01_motivation.jl: Three methods applied to Van der Pol
# Slide: "1. Hook"
#
# Static: true trajectory + SINDy reconstruction + energy trace

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 4 * FPS

# Van der Pol data
ts, states, dt = gen_trajectory("van_der_pol"; noise=0.0)

# SINDy
dX  = finite_diff(states, dt)
Θ, names = build_library(states, 3, false)
xi  = stlsq(Θ, dX, 0.1)
n_active = count(any(xi .!= 0, dims=2))
rmse = sqrt(mean((Θ*xi .- dX).^2))
u0   = [2.0, 0.0]
recon = sindy_reconstruct(xi, u0, size(states,1), dt, 3, false)
n_valid = sum(.!isnan.(recon[:,1]))

# Conservation (pseudo-energy E = x₁² + x₂² for limit cycle)
E_true  = states[:,1].^2 .+ states[:,2].^2

fig = Figure(size = RES)
Label(fig[0, :],
    L"\text{Physics Discovery: SINDy · Kernel Invariants · Symbolic Regression}";
    color=C_TXT, fontsize=22)

# Phase portrait: true vs SINDy
ax1 = Axis(fig[1,1];
    title  = L"\text{Van der Pol: True vs SINDy (}\lambda=0.1\text{)}",
    xlabel = L"x_1", ylabel = L"x_2")
n_show = Observable(1)
N_pts = size(states, 1)
lines!(ax1, @lift(states[1:$n_show, 1]), @lift(states[1:$n_show, 2]);
    color=C_TRAJ, linewidth=1.5, label=L"\text{True}")
if n_valid > 10
    lines!(ax1, @lift(recon[1:min($n_show, n_valid), 1]), @lift(recon[1:min($n_show, n_valid), 2]);
           color=C_RECON, linewidth=2.0, linestyle=:dash, label=L"\text{SINDy}")
end
axislegend(ax1; position=:lt, labelsize=16)
text!(ax1, -1.5, 2.2; text="$(n_active) active terms\nRMSE=$(round(rmse;sigdigits=3))",
      color=C_LIB, fontsize=16)

# SINDy coefficient bar chart
active_mask = vec(any(xi .!= 0; dims=2))
ax2 = Axis(fig[1,2];
    title  = L"\text{SINDy coefficients }\xi \text{ for }\dot{x}_2",
    ylabel = L"\xi",
    xticks = (1:length(names), names),
    xticklabelrotation = 0.7)
bar_colors = [active_mask[i] ? C_LIB : C_DIM for i in 1:length(names)]
barplot!(ax2, 1:length(names), xi[:,2]; color=bar_colors)
hlines!(ax2, [0.0]; color=C_DIM, linestyle=:dash)

# "energy" trace (x₁²+x₂² on limit cycle approaches constant)
ax3 = Axis(fig[1,3];
    title  = L"\text{Amplitude envelope }(x_1^2+x_2^2)^{1/2} \text{ converges}",
    xlabel = L"t", ylabel = L"\|x\|")
lines!(ax3, @lift(ts[1:$n_show]), @lift(sqrt.(E_true)[1:$n_show]); color=C_TRAJ, linewidth=2.0)
hlines!(ax3, [2.0]; color=C_THEORY, linestyle=:dash, label=L"\text{limit cycle radius}=2")
axislegend(ax3; position=:rb, labelsize=16)

Label(fig[2,:],
    L"\text{SINDy recovers: }\dot{x}_1 = x_2,\;\dot{x}_2 = \mu(1-x_1^2)x_2 - x_1\;\text{ from data alone}";
    color=C_DIM, fontsize=15)

record(fig, out("s01_motivation.mp4"), 1:N_FRAMES; framerate=FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * N_pts))
end
println("✓  s01_motivation.mp4")
