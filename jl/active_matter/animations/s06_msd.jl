# s06_msd.jl, MSD ballistic-to-diffusive crossover
# Slide: "Run-and-Tumble MSD"
#
# 3B1B-style staging: lay down the two scaling LAWS first (t² and t), then let
# the measured data draw on and visibly bend from one to the other at t = τ.

using CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "vicsek.jl"))

set_theme!(slide_theme())

const V_RT = 1.0
const Λ_RT = 0.5
const TAU  = 1.0 / Λ_RT          # = 2.0
const DEFF = V_RT^2 * TAU / 2     # = 1.0

println("Simulating run-and-tumble MSD (N=250, T=30) ...")
ts_sim, msds_sim = simulate_runtumble_msd(250; v=V_RT, λ=Λ_RT, dt=0.05, T=30.0, seed=42)
msds_theo = msd_theory(ts_sim; v=V_RT, λ=Λ_RT)

t_ref = collect(range(0.05, 30.0; length=300))
msd_ballistic = @. V_RT^2 * t_ref^2
msd_diffusive = @. 4 * DEFF * t_ref

N_FRAMES = 7 * FPS
n_pts    = length(ts_sim)
frac     = Observable(0.0)

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Run-and-Tumble MSD: from ballistic } (t^2) \text{ to diffusive } (t)";
      color = @lift(fadein(C_TXT, $frac, 0.0, 0.06)), fontsize=24)

ax = Axis(fig[1, 1];
    title=L"\langle r^2(t)\rangle \text{ (log-log)}", xlabel=L"t", ylabel=L"\langle r^2\rangle",
    xscale=log10, yscale=log10, limits=(0.05, 35.0, 1e-3, 1e4))
ax2 = Axis(fig[1, 2];
    title=L"\text{measured vs exact theory (linear)}", xlabel=L"t", ylabel=L"\langle r^2\rangle",
    limits=(0.0, 30.0, 0.0, 130.0))

# ── Stage 1: the two scaling laws draw on first (the "rulers") ────────────────
kb = @lift reveal_n($frac, 0.08, 0.30, length(t_ref))
kd = @lift reveal_n($frac, 0.22, 0.44, length(t_ref))
lines!(ax, @lift(t_ref[1:$kb]), @lift(msd_ballistic[1:$kb]);
    color = @lift(fadein(C_FLOCK, $frac, 0.08, 0.16)), linewidth=2, linestyle=:dash)
lines!(ax, @lift(t_ref[1:$kd]), @lift(msd_diffusive[1:$kd]);
    color = @lift(fadein(C_PHI, $frac, 0.22, 0.30)), linewidth=2, linestyle=:dot)
text!(ax, 0.12, 0.5; text=L"\sim t^2", color=@lift(fadein(C_FLOCK,$frac,0.16,0.24)), fontsize=22)
text!(ax, 9.0, 4.0;  text=L"\sim t",   color=@lift(fadein(C_PHI,$frac,0.30,0.38)), fontsize=22)

# ── Stage 2: measured data draws on over both panels ─────────────────────────
ksim = @lift reveal_n($frac, 0.34, 0.84, n_pts)
lines!(ax, @lift(ts_sim[1:$ksim]), @lift(msds_sim[1:$ksim]); color=C_MSD, linewidth=3)
lines!(ax2, @lift(ts_sim[1:$ksim]), @lift(msds_theo[1:$ksim]);
    color=@lift(fadein(C_THEO,$frac,0.34,0.42)), linewidth=3, label="exact theory")
lines!(ax2, @lift(ts_sim[1:$ksim]), @lift(msds_sim[1:$ksim]); color=C_MSD, linewidth=2, label="simulation")
axislegend(ax2; position=:lt, labelsize=16)

# ── Stage 3: crossover callout at t = τ ──────────────────────────────────────
vlines!(ax, [TAU]; color=@lift(fadein(C_TXT,$frac,0.70,0.80)), linestyle=:dash, linewidth=2)
text!(ax, TAU*1.15, 0.004; text=L"t=\tau=1/\lambda",
      color=@lift(fadein(C_TXT,$frac,0.74,0.84)), fontsize=18)
text!(ax, 0.07, 2.5e3;
      text="one run length:\nmotion remembers its heading\nfor τ, then forgets",
      color=@lift(fadein(C_DIM,$frac,0.82,0.94)), fontsize=15, align=(:left,:top))

ttl = @lift "t = $(round(ts_sim[max(1,$ksim)]; digits=1))   ($(round(ts_sim[max(1,$ksim)]/TAU; digits=1)) τ)"
Label(fig[2, :], ttl; color=C_DIM, fontsize=18)

record(fig, out("s06_msd.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
end
println("✓  s06_msd.mp4")
