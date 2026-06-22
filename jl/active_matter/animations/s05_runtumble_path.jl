# s05_runtumble_path.jl, Single run-and-tumble trajectory
# Slide: "Run-and-Tumble Dynamics, E. coli Swimming"
#
# 3B1B-style: the path draws on (eased); each tumble flashes a marker and the
# heading θ(t) jumps in the side panel, making "run, then randomly reorient"
# visible as straight segments punctuated by kinks.

using CairoMakie, Random
include(joinpath(@__DIR__, "shared", "style.jl"))

set_theme!(slide_theme())

const v_rt, λ_rt, dt_rt, T_rt = 1.0, 0.4, 0.05, 18.0   # τ = 1/λ = 2.5
n_steps = round(Int, T_rt / dt_rt)

rng = MersenneTwister(55)
θ = rand(rng) * 2π
px = Float64[0.0]; py = Float64[0.0]; θ_vals = Float64[θ]; tumble_steps = Int[]
for step in 1:n_steps
    if rand(rng) < λ_rt * dt_rt
        global θ = rand(rng) * 2π; push!(tumble_steps, step)
    end
    push!(px, px[end] + v_rt*dt_rt*cos(θ)); push!(py, py[end] + v_rt*dt_rt*sin(θ))
    push!(θ_vals, θ)
end
t_vals = (0:n_steps) .* dt_rt
side = max(maximum(px)-minimum(px), maximum(py)-minimum(py)) * 1.15
xc, yc = (minimum(px)+maximum(px))/2, (minimum(py)+maximum(py))/2

N_FRAMES = 7 * FPS
frac = Observable(0.0)

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Run-and-tumble: straight runs punctuated by random reorientations } (v=1,\;\lambda=0.4)";
      color=@lift(fadein(C_TXT,$frac,0.0,0.06)), fontsize=22)

ax = Axis(fig[1,1]; title=L"\text{trajectory}", xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
    limits=(xc-side/2, xc+side/2, yc-side/2, yc+side/2))
axθ = Axis(fig[1,2]; title=L"\text{heading } \theta(t) \text{, flat during a run, jumps at a tumble}",
    xlabel=L"t", ylabel=L"\theta", limits=(0.0, T_rt, 0.0, 2π))

ns = @lift max(2, reveal_n($frac, 0.05, 0.95, n_steps+1))
lines!(ax, @lift(px[1:$ns]), @lift(py[1:$ns]); color=C_PHI, linewidth=2.4)
scatter!(ax, @lift([px[$ns]]), @lift([py[$ns]]); color=C_MSD, markersize=15)
# tumble markers, visible once the path reaches them
tx = [px[s] for s in tumble_steps]; ty = [py[s] for s in tumble_steps]
for (i, s) in enumerate(tumble_steps)
    scatter!(ax, @lift($ns >= s ? [tx[i]] : Float64[]), @lift($ns >= s ? [ty[i]] : Float64[]);
        color=C_DIS, markersize=13, marker=:xcross)
end

lines!(axθ, @lift(t_vals[1:$ns]), @lift(θ_vals[1:$ns]); color=C_PHI, linewidth=2)
for s in tumble_steps
    vlines!(axθ, @lift($ns >= s ? [s*dt_rt] : Float64[]); color=(C_DIS,0.45), linewidth=1)
end
text!(axθ, T_rt*0.05, 5.7; text=L"\text{run length } \ell = v/\lambda = 2.5",
      color=@lift(fadein(C_DIM,$frac,0.3,0.45)), fontsize=17, align=(:left,:top))

lab = @lift "t = $(round(t_vals[$ns];digits=1))   |   tumbles so far: $(sum(s <= $ns for s in tumble_steps))"
Label(fig[2, :], lab; color=C_DIM, fontsize=18)

record(fig, out("s05_runtumble_path.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
end
println("✓  s05_runtumble_path.mp4")
