# s04_bifurcation.jl, Bifurcation φ(η) for two densities + susceptibility peak
# Slide: "Bifurcation Diagram and Critical Noise"
#
# 3B1B-style: sweep the two bifurcation curves on (denser flock = more robust),
# drop the predicted η_c markers, then reveal the susceptibility peak that
# locates the transition.

using CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "vicsek.jl"))

set_theme!(slide_theme())

N_VIC  = 120
ρ_vals = [1.5, 3.0]
η_range = collect(range(0.2, 3.8; length=28))
ρ_colors = [C_FLOCK, C_MSD]
ρ_labels = ["ρ = 1.5", "ρ = 3.0"]

println("Computing bifurcation for two densities ...")
results = map(ρ_vals) do ρ
    map(η_range) do η
        φ = simulate_vicsek(N_VIC, η; ρ=ρ, n_warm=200, n_run=250, seed=42)[2]
        (mean(φ), std(φ))
    end
end
χ_all = [[N_VIC * r[2]^2 for r in res] for res in results]
χmax = maximum(maximum.(χ_all)) * 1.15

N_FRAMES = 7 * FPS
N_ETA = length(η_range)
frac = Observable(0.0)

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Flocking is a phase transition: } \langle\varphi\rangle \text{ vs noise, for two densities}";
      color=@lift(fadein(C_TXT,$frac,0.0,0.06)), fontsize=22)

ax = Axis(fig[1, 1]; title=L"\text{order parameter bifurcation}",
    xlabel=L"\eta", ylabel=L"\langle\varphi\rangle", limits=(0.0, 4.0, 0.0, 1.1))
ax2 = Axis(fig[1, 2]; title=L"\chi = N(\langle\varphi^2\rangle-\langle\varphi\rangle^2) \text{ peaks at } \eta_c",
    xlabel=L"\eta", ylabel=L"\chi", limits=(0.0, 4.0, 0.0, χmax))

# Two curves sweep in (low density first, then high density)
swins = [(0.10, 0.42), (0.40, 0.72)]
for (res, χs, col, lbl, ρ, (a,b)) in zip(results, χ_all, ρ_colors, ρ_labels, ρ_vals, swins)
    ms = [r[1] for r in res]; ss = [r[2] for r in res]
    k = @lift reveal_n($frac, a, b, N_ETA)
    band!(ax, @lift(η_range[1:$k]), @lift((ms.-ss)[1:$k]), @lift((ms.+ss)[1:$k]); color=(col, 0.18))
    lines!(ax, @lift(η_range[1:$k]), @lift(ms[1:$k]); color=col, linewidth=2.8, label=lbl)
    η_c = 1.1 * sqrt(ρ)
    vlines!(ax, [η_c]; color=@lift(fadein(col,$frac,b-0.02,b+0.05)), linestyle=:dash, linewidth=1.6)
    text!(ax, η_c+0.05, 1.02; text="η_c(ρ=$ρ)", color=@lift(fadein(col,$frac,b,b+0.07)),
          fontsize=14, align=(:left,:center))
    # susceptibility peak (revealed in the last third)
    lines!(ax2, @lift(η_range[1:reveal_n($frac,0.70,0.94,N_ETA)]),
           @lift(χs[1:reveal_n($frac,0.70,0.94,N_ETA)]); color=col, linewidth=2.8, label=lbl)
end
axislegend(ax; position=:rt, labelsize=16)
text!(ax2, 0.2, χmax*0.9; text="denser flock →\nresists noise to higher η_c",
      color=@lift(fadein(C_DIM,$frac,0.86,0.96)), fontsize=15, align=(:left,:top))

Label(fig[2, :], L"N=%$(N_VIC),\;v_0=0.03,\;r=1.0;\;\text{200 warm + 250 averaging steps per point}";
      color=C_DIM, fontsize=15)

record(fig, out("s04_bifurcation.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
end
println("✓  s04_bifurcation.mp4")
