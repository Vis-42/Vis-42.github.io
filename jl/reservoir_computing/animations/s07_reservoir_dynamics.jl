# s07_reservoir_dynamics.jl: Lorenz input vs reservoir state traces
# Slide: "Architecture: Reservoir Dynamics"
#
# Left:  x(t), y(t), z(t) Lorenz input (first 5 time units)
# Right: 6 selected reservoir state traces h_i(t): nonlinear expansion

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 8 * FPS
T_SHOW = 5.0

# ── Generate Lorenz input and reservoir states ────────────────────────────────
ts_d, traj_d = generate_lorenz(T_SHOW; u0 = [1.0, 0.0, 0.0], warmup = 5.0)
N_D = length(ts_d)

esn = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                   spectral_radius = 0.7, seed = 42))
esn.state .= 0.0

states = zeros(N_D, 50)
for i in 1:N_D
    esn_step!(esn, traj_d[i, :])
    states[i, :] .= esn.state
end

# Pick 6 diverse nodes
node_ids = [1, 8, 15, 23, 37, 48]
res_colors = [C_TRUE, C_PRED, C_RES, C_ATT, C_DIV, RGBf(0.9, 0.85, 0.3)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_in = Axis(fig[1, 1];
    title  = L"\text{Lorenz-63 input }(x,y,z)\text{ first 5 time units}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"\text{state}",
    limits = (0.0, T_SHOW, -35.0, 55.0))

ax_res = Axis(fig[1, 2];
    title  = L"h_i(t)\text{: 6 reservoir nodes driven by Lorenz input}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"h_i(t)",
    limits = (0.0, T_SHOW, -1.1, 1.1))

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)
ts_obs = @lift ts_d[1:$n_show]

# Lorenz input components
lines!(ax_in, ts_obs, @lift(traj_d[1:$n_show, 1]);
    color = C_TRUE,  linewidth = 1.8, label = L"x(t)")
lines!(ax_in, ts_obs, @lift(traj_d[1:$n_show, 2]);
    color = C_PRED,  linewidth = 1.8, label = L"y(t)")
lines!(ax_in, ts_obs, @lift(traj_d[1:$n_show, 3]);
    color = C_ATT,   linewidth = 1.8, label = L"z(t)")
axislegend(ax_in; position = :rt, labelsize = 17)

# Reservoir state traces
lbl_map = Dict(1=>L"h_1", 8=>L"h_8", 15=>L"h_{15}", 23=>L"h_{23}", 37=>L"h_{37}", 48=>L"h_{48}")
for (k, (nid, col)) in enumerate(zip(node_ids, res_colors))
    lines!(ax_res, ts_obs, @lift(states[1:$n_show, nid]);
        color = col, linewidth = 1.6, label = lbl_map[nid])
end
axislegend(ax_res; position = :rt, labelsize = 15)

t_label = @lift "t = $(round(ts_d[$n_show]; digits=2)) time units"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

record(fig, out("s07_reservoir_dynamics.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_D))
end
println("✓  s07_reservoir_dynamics.mp4")
