# s07_deff_comparison.jl, D_eff = v²/(2λ): theory curve vs measured points
# Slide: "Exact MSD, effective diffusion"
#
# 3B1B-style: draw the theoretical law as a smooth curve, then let each
# simulation point fall onto it one at a time, annotating the % agreement.

using CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "vicsek.jl"))

set_theme!(slide_theme())

λ_vals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
V_RT   = 1.0
D_theo = @. V_RT^2 / (2 * λ_vals)

println("Measuring D_eff at $(length(λ_vals)) tumble rates ...")
D_sim = map(λ_vals) do λ
    τ = 1.0 / λ
    ts, msds = simulate_runtumble_msd(200; v=V_RT, λ=λ, dt=0.05, T=max(25.0, 15τ), seed=77)
    n = length(ts); i0 = round(Int, 0.7n)
    tl = ts[i0:end]; ml = msds[i0:end]
    tm = sum(tl)/length(tl); mm = sum(ml)/length(ml)
    slope = sum((tl .- tm) .* (ml .- mm)) / sum((tl .- tm).^2)
    slope / 4
end

λ_fine = collect(range(0.15, 2.3; length=120))
D_fine = @. V_RT^2 / (2 * λ_fine)

N_FRAMES = 6 * FPS
nλ = length(λ_vals)
frac = Observable(0.0)

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Effective diffusion } D_\mathrm{eff} = v^2\tau/2 = v^2/(2\lambda) \text{, theory meets measurement}";
      color=@lift(fadein(C_TXT,$frac,0.0,0.06)), fontsize=22)

ax = Axis(fig[1, 1]; title=L"D_\mathrm{eff} \text{ vs tumble rate } \lambda",
    xlabel=L"\lambda", ylabel=L"D_\mathrm{eff}", limits=(0.0, 2.4, 0.0, 2.7))

# theory law draws on first
kt = @lift reveal_n($frac, 0.08, 0.34, length(λ_fine))
lines!(ax, @lift(λ_fine[1:$kt]), @lift(D_fine[1:$kt]);
    color=@lift(fadein(C_THEO,$frac,0.08,0.16)), linewidth=3, label=L"v^2/(2\lambda)\;\text{(theory)}")

# measured points land one by one, with % error labels
np = @lift reveal_n($frac, 0.36, 0.92, nλ) - 1
scatter!(ax, @lift(λ_vals[1:max(1,$np)]), @lift(D_sim[1:max(1,$np)]);
    color=C_MSD, markersize=15, marker=:circle, label="simulation")
for (i, (λ, ds, dt)) in enumerate(zip(λ_vals, D_sim, D_theo))
    err = round(abs(ds-dt)/dt*100; digits=1)
    text!(ax, λ+0.03, ds+0.07; text="$(err)%", color=C_DIM, fontsize=12,
          visible=@lift($np >= i))
end
axislegend(ax; position=:rt, labelsize=16)

# right: the physical message
ax2 = Axis(fig[1, 2]; title=L"\text{run length } \ell = v\tau = v/\lambda",
    xlabel=L"\lambda", ylabel=L"\tau = 1/\lambda", limits=(0.0, 2.4, 0.0, 6.0))
τ_fine = @. 1.0 / λ_fine
lines!(ax2, @lift(λ_fine[1:$kt]), @lift(τ_fine[1:$kt]);
    color=@lift(fadein(C_FLOCK,$frac,0.10,0.20)), linewidth=3)
text!(ax2, 0.5, 4.6;
    text="slower tumbling → longer runs →\nfar more diffusion than any\nthermal (passive) particle",
    color=@lift(fadein(C_DIM,$frac,0.6,0.78)), fontsize=16, align=(:left,:top))

Label(fig[2, :], L"N=200,\;v=%$(V_RT);\;\text{slope fit of } \langle r^2\rangle=4D_\mathrm{eff}t \text{ in the diffusive tail}";
      color=C_DIM, fontsize=15)

record(fig, out("s07_deff_comparison.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
end
println("✓  s07_deff_comparison.mp4")
