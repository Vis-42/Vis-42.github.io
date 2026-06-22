# s04_echo_state.jl: Echo State Property demonstration
# Slide: "Theory: Echo State Property"
#
# Left:  reservoir state traces h_1^A(t) and h_1^B(t): starting from different
#        random initial conditions, both driven by same Lorenz input: they converge
# Right: ‖h^A(t) - h^B(t)‖ on log scale showing exponential washout

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 8 * FPS

# ── Build ESN and drive with Lorenz input from two initial conditions ─────────
T_DRIVE = 6.0   # drive for 6 time units
WASHOUT_DEMO = 0   # no washout, we want to see the transient

_, traj_drive = generate_lorenz(T_DRIVE; u0 = [1.0, 0.0, 0.0], warmup = 5.0)
N_DRIVE = size(traj_drive, 1)

# Build ESN
esn_params = ESNParams(; N_res = 50, sparsity = 0.15, spectral_radius = 0.7,
                         input_scaling = 0.5, ridge_alpha = 1e-6, seed = 42)
esn = build_simple_esn(esn_params; n_input = 3)

# Drive from two different initial conditions
rng = MersenneTwister(123)
h0_A = randn(rng, 50) .* 0.5
h0_B = randn(rng, 50) .* 0.5

states_A = zeros(N_DRIVE, 50)
states_B = zeros(N_DRIVE, 50)

# Run trajectory A
esn_A = deepcopy(esn)
esn_A.state .= h0_A
for i in 1:N_DRIVE
    esn_step!(esn_A, traj_drive[i, :])
    states_A[i, :] .= esn_A.state
end

# Run trajectory B
esn_B = deepcopy(esn)
esn_B.state .= h0_B
for i in 1:N_DRIVE
    esn_step!(esn_B, traj_drive[i, :])
    states_B[i, :] .= esn_B.state
end

ts_drive = collect(range(0.0, T_DRIVE; length = N_DRIVE))
h1A = states_A[:, 1]
h1B = states_B[:, 1]
diff_norm = [norm(states_A[i, :] .- states_B[i, :]) for i in 1:N_DRIVE]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_h = Axis(fig[1, 1];
    title  = L"h_1(t)\text{ driven by same Lorenz input, two initial conditions}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"h_1(t)",
    limits = (0.0, T_DRIVE, -1.1, 1.1))

ax_d = Axis(fig[1, 2];
    title  = L"|h^A(t) - h^B(t)|\text{: echo state washout}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"|h^A - h^B|",
    yscale = log10,
    limits = (0.0, T_DRIVE, 1e-10, 10.0))

text!(ax_d, T_DRIVE * 0.05, 1e-8;
    text = L"\text{exponential washout} \Rightarrow \text{echo state property}",
    color = C_ATT, fontsize = 17)

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)
ts_obs  = @lift ts_drive[1:$n_show]

lines!(ax_h, ts_obs, @lift(h1A[1:$n_show]);
    color = C_TRUE, linewidth = 2.0, label = L"h_1^A(t)")
lines!(ax_h, ts_obs, @lift(h1B[1:$n_show]);
    color = C_PRED, linewidth = 2.0, linestyle = :dash, label = L"h_1^B(t)")
axislegend(ax_h; position = :rt, labelsize = 17)

# diff_norm may hit zero exactly, clamp to small positive for log scale
diff_safe = max.(diff_norm, 1e-14)
lines!(ax_d, ts_obs, @lift(diff_safe[1:$n_show]);
    color = C_RES, linewidth = 2.5)

t_label = @lift "t = $(round(ts_drive[$n_show]; digits=2)) time units"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

record(fig, out("s04_echo_state.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_DRIVE))
end
println("✓  s04_echo_state.mp4")
