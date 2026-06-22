# s03_order_parameter.jl, φ(t) at three noise levels, then the bifurcation
# Slide: "Order Parameter and the Flocking Transition"
#
# 3B1B-style: draw the three time-traces ONE AT A TIME, each with its own
# annotation beat, then assemble the bifurcation curve ⟨φ⟩(η) on the right.

using CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "vicsek.jl"))

set_theme!(slide_theme())

N_VIC, RHO_VIC, N_RUN = 120, 1.5, 350
ηs     = [0.5, 1.2, 3.0]
colors = [C_FLOCK, C_MSD, C_DIS]

println("Pre-computing φ(t) traces ...")
φ_traces = map(η -> simulate_vicsek(N_VIC, η; ρ=RHO_VIC, n_warm=150, n_run=N_RUN, seed=99)[2], ηs)

η_range = collect(range(0.2, 3.8; length=25))
println("Computing bifurcation ...")
means_ = Float64[]; stds_ = Float64[]
for η in η_range
    φ = simulate_vicsek(N_VIC, η; ρ=RHO_VIC, n_warm=150, n_run=200, seed=7)[2]
    push!(means_, mean(φ)); push!(stds_, std(φ))
end
N_BIF = length(η_range)

N_FRAMES = 8 * FPS
frac = Observable(0.0)

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Order parameter } \varphi = |\langle e^{i\theta}\rangle|\text{, one number for how aligned}";
      color=@lift(fadein(C_TXT,$frac,0.0,0.06)), fontsize=23)

ax = Axis(fig[1, 1]; title=L"\varphi(t) \text{ at three noise levels}",
    xlabel=L"\text{time step}", ylabel=L"\varphi(t)", limits=(0, N_RUN, 0.0, 1.05))

# 1/√N floor (random-phase value), fades in first
hlines!(ax, [1/sqrt(N_VIC)]; color=@lift(fadein(C_DIM,$frac,0.06,0.12)), linestyle=:dot, linewidth=1.5)
text!(ax, 8, 1/sqrt(N_VIC)+0.03; text=L"1/\sqrt{N}", color=@lift(fadein(C_DIM,$frac,0.08,0.14)), fontsize=15)

# Three traces, drawn one at a time with their own annotation
wins = [(0.10,0.30), (0.32,0.52), (0.54,0.72)]
notes = [(L"\eta=0.5:\;\text{flock locks in},\;\varphi\to 1", 0.86),
         (L"\eta\approx\eta_c:\;\text{critical, wild fluctuations}", 0.55),
         (L"\eta=3.0:\;\text{noise wins},\;\varphi\to 0", 0.16)]
for (φ, col, (a,b), (note,ypos)) in zip(φ_traces, colors, wins, notes)
    k = @lift reveal_n($frac, a, b, N_RUN)
    lines!(ax, @lift(1:$k), @lift(φ[1:$k]); color=col, linewidth=2.8)
    text!(ax, N_RUN*0.42, ypos; text=note, color=@lift(fadein(col,$frac,b-0.02,b+0.06)),
          fontsize=16, align=(:left,:center))
end

# Bifurcation panel assembles last
ax2 = Axis(fig[1, 2]; title=L"\langle\varphi\rangle \pm \sigma_\varphi \text{ vs noise } \eta",
    xlabel=L"\eta", ylabel=L"\langle\varphi\rangle", limits=(0.0, 4.0, 0.0, 1.1))
kb = @lift reveal_n($frac, 0.62, 0.92, N_BIF)
band!(ax2, @lift(η_range[1:$kb]), @lift((means_.-stds_)[1:$kb]), @lift((means_.+stds_)[1:$kb]);
      color=(C_FLOCK, 0.22))
lines!(ax2, @lift(η_range[1:$kb]), @lift(means_[1:$kb]); color=C_FLOCK, linewidth=2.8)
vlines!(ax2, [1.2]; color=@lift(fadein(C_TXT,$frac,0.90,0.97)), linestyle=:dash, linewidth=1.8)
text!(ax2, 1.28, 0.92; text=L"\eta_c\approx 1.2", color=@lift(fadein(C_TXT,$frac,0.92,0.98)), fontsize=17)

Label(fig[2, :], L"N=%$(N_VIC),\;\rho=%$(RHO_VIC),\;v_0=0.03"; color=C_DIM, fontsize=16)

record(fig, out("s03_order_parameter.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
end
println("✓  s03_order_parameter.mp4")
