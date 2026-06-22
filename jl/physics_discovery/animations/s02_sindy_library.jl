# s02_sindy_library.jl: Library matrix Θ heatmap + active term identification
# Slide: "2. SINDy I: Library construction"
#
# Static: Θ heatmap + STLSQ threshold sweep showing sparsification

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 4 * FPS

ts, states, dt = gen_trajectory("van_der_pol"; noise=0.0)
dX = finite_diff(states, dt)
Θ, names = build_library(states, 3, false)

# Sweep thresholds to show sparsification
λ_range  = range(0.01, 0.5; length=30)
n_active = Int[count(any(stlsq(Θ, dX, λ) .!= 0; dims=2)) for λ in λ_range]
rmse_arr = Float64[sqrt(mean((Θ*stlsq(Θ,dX,λ) .- dX).^2)) for λ in λ_range]

# Final ξ at λ=0.1
xi = stlsq(Θ, dX, 0.1)

# Subset of Θ to show (first 200 rows, all columns)
N_SHOW = min(200, size(Θ, 1))
Θ_show = Θ[1:N_SHOW, :]

fig = Figure(size = RES)
Label(fig[0,:], L"\text{SINDy: Dictionary Matrix }\Theta \text{ and STLSQ Sparsification}";
    color=C_TXT, fontsize=22)

# Θ heatmap
ax1 = Axis(fig[1,1:2];
    title  = L"\Theta(X)\text{: library evaluated on trajectory (clipped to }[-2,2]\text{)}",
    xlabel = L"\text{library term}",
    ylabel = L"\text{time snapshot}",
    xticks = (1:length(names), names),
    xticklabelrotation = 0.7,
    xticklabelsize = 14)
hm = heatmap!(ax1, 1:length(names), 1:N_SHOW,
    clamp.(Θ_show', -2.0, 2.0);
    colormap = :RdBu, colorrange=(-2,2))
Colorbar(fig[1,0], hm; label=L"\Theta_{ij}", labelcolor=C_TXT)

# Mark active columns at λ=0.1
active_mask = vec(any(xi .!= 0; dims=2))
for (j, active) in enumerate(active_mask)
    active && vlines!(ax1, [Float64(j)]; color=(C_LIB, 0.6), linewidth=2.0)
end

# Sparsity vs λ
ax2 = Axis(fig[1,3];
    title  = L"\text{Active terms vs }\lambda",
    xlabel = L"\lambda \;\text{(STLSQ threshold)}",
    ylabel = L"\text{active terms}")
n_λ = Observable(1)
λ_vec = collect(λ_range)
lines!(ax2, @lift(λ_vec[1:$n_λ]), @lift(n_active[1:$n_λ]); color=C_LIB, linewidth=2.5)
scatter!(ax2, @lift(λ_vec[1:$n_λ]), @lift(n_active[1:$n_λ]); color=C_LIB, markersize=7)
vlines!(ax2, [0.1]; color=C_THEORY, linestyle=:dash, linewidth=1.5,
    label=L"\lambda=0.1 \;\text{(default)}")
axislegend(ax2; position=:rt, labelsize=14)

# RMSE vs λ
ax3 = Axis(fig[2,3];
    title  = L"\text{RMSE vs }\lambda",
    xlabel = L"\lambda",
    ylabel = L"\text{RMSE}")
lines!(ax3, @lift(λ_vec[1:$n_λ]), @lift(rmse_arr[1:$n_λ]); color=C_ERR, linewidth=2.5)
vlines!(ax3, [0.1]; color=C_THEORY, linestyle=:dash, linewidth=1.5)

# ξ heatmap
ax4 = Axis(fig[2,1:2];
    title  = L"\xi \text{ coefficient matrix } (\dot{x}_1 \text{ left, } \dot{x}_2 \text{ right})",
    xlabel = L"\text{library term}",
    ylabel = L"\text{state}",
    xticks = (1:length(names), names),
    xticklabelrotation = 0.7,
    xticklabelsize = 14,
    yticks = ([1,2], [L"\dot{x}_1", L"\dot{x}_2"]))
hm2 = heatmap!(ax4, 1:length(names), 1:2,
    clamp.(xi', -2, 2);
    colormap = :RdBu, colorrange=(-2,2))
Colorbar(fig[2,0], hm2; label=L"\xi_{ij}", labelcolor=C_TXT)

Label(fig[3,:],
    L"\text{Highlighted columns: active at }\lambda=0.1.\;\xi_2\text{ row recovers: }\dot{x}_2 = x_2 - x_1^2 \cdot x_2 - x_1";
    color=C_DIM, fontsize=14)

record(fig, out("s02_sindy_library.mp4"), 1:N_FRAMES; framerate=FPS) do f
    n_λ[] = max(1, round(Int, (f / N_FRAMES) * length(λ_range)))
end
println("✓  s02_sindy_library.mp4")
