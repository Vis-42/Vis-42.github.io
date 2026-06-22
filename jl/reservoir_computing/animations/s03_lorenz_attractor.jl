# s03_lorenz_attractor.jl: 3D Lorenz butterfly attractor built progressively
# Slide: "Theory: Lorenz-63 System"
#
# 3D butterfly built over 500 time units, camera slowly orbits,
# fixed points C+ and C- marked, equations labeled.

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 10 * FPS

# ── Pre-compute trajectory ────────────────────────────────────────────────────
T_ATT = 30.0  # 30 time units is plenty to show full butterfly
ts_att, traj_att = generate_lorenz(T_ATT; u0 = [1.0, 0.0, 0.0], warmup = 5.0)

xs_att = traj_att[:, 1]
ys_att = traj_att[:, 2]
zs_att = traj_att[:, 3]

# Fixed points C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)
c_val = sqrt(β_L * (ρ_L - 1.0))
Cplus  = ( c_val,  c_val, ρ_L - 1.0)
Cminus = (-c_val, -c_val, ρ_L - 1.0)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax3 = Axis3(fig[1, 1];
    title  = L"\text{Lorenz-63 strange attractor}",
    xlabel = L"x",
    ylabel = L"y",
    zlabel = L"z")

xlims!(ax3, -25.0, 25.0)
ylims!(ax3, -35.0, 35.0)
zlims!(ax3,   0.0, 55.0)

# Fixed points
scatter!(ax3, [Cplus[1], Cminus[1]], [Cplus[2], Cminus[2]], [Cplus[3], Cminus[3]];
    color = C_ATT, markersize = 14, marker = :star5)
text!(ax3, Cplus[1]  + 1.5, Cplus[2],  Cplus[3]  + 1.5;
    text = L"C^+", color = C_ATT, fontsize = 19)
text!(ax3, Cminus[1] - 4.0, Cminus[2], Cminus[3] + 1.5;
    text = L"C^-", color = C_ATT, fontsize = 19)

# Equation annotation (static text in figure)
Label(fig[2, 1],
    L"\dot{x}=\sigma(y-x),\;\dot{y}=x(\rho-z)-y,\;\dot{z}=xy-\beta z \qquad \sigma=10,\;\rho=28,\;\beta=8/3";
    color = C_DIM, fontsize = 19, tellwidth = false)

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)

lines!(ax3,
    @lift(xs_att[1:$n_show]),
    @lift(ys_att[1:$n_show]),
    @lift(zs_att[1:$n_show]);
    color = C_TRUE, linewidth = 1.2, alpha = 0.85)

# Camera orbit
N_pts = length(ts_att)
record(fig, out("s03_lorenz_attractor.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_pts))
    ax3.azimuth[] = 1.2 + frac * 1.0   # slow orbit ~57 degrees
end
println("✓  s03_lorenz_attractor.mp4")
