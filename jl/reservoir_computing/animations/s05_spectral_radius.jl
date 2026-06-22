# s05_spectral_radius.jl: Spectral radius and memory depth
# Slide: "Theory: Spectral Radius"
#
# Left:  undriven reservoir state norms ‖h(t)‖ for ρ=0.5, 0.8, 0.95, 1.2
# Right: T_mem = -1/ln(ρ) vs ρ, vertical line at ρ=0.7 (best param from sweep)

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 7 * FPS

# ── Simulate undriven reservoirs ──────────────────────────────────────────────
T_UNDRIVEN = 60   # steps
rho_values = [0.5, 0.8, 0.95, 1.2]
rho_colors = [C_TRUE, C_ATT, C_PRED, C_DIV]
rho_labels = [L"\rho = 0.5", L"\rho = 0.8", L"\rho = 0.95", L"\rho = 1.2"]

steps_axis = 0:T_UNDRIVEN

function undriven_norms(rho_target)
    esn = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                       spectral_radius = rho_target, seed = 42))
    rng = MersenneTwister(7)
    esn.state .= randn(rng, 50) .* 0.1
    esn.state ./= norm(esn.state)   # unit initial state
    norms = zeros(T_UNDRIVEN + 1)
    norms[1] = norm(esn.state)
    u_zero = zeros(3)
    for t in 1:T_UNDRIVEN
        esn_step!(esn, u_zero)
        norms[t + 1] = norm(esn.state)
    end
    norms
end

norms_all = [undriven_norms(ρ) for ρ in rho_values]

# ── T_mem curve ───────────────────────────────────────────────────────────────
rho_range = range(0.5, 0.99; length = 200)
T_mem_curve = [-1.0 / log(ρ) for ρ in rho_range]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_n = Axis(fig[1, 1];
    title  = L"|h(t)|\text{: undriven reservoir (log scale)}",
    xlabel = L"\text{Step } t",
    ylabel = L"|h(t)|\;\text{(log scale)}",
    yscale = log10,
    limits = (0.0, Float64(T_UNDRIVEN), 1e-3, 20.0))

ax_m = Axis(fig[1, 2];
    title  = L"T_{mem} = -1/\ln\rho\text{: memory depth vs spectral radius}",
    xlabel = L"\rho",
    ylabel = L"T_{mem}\;\text{(steps)}",
    limits = (0.48, 1.01, 0.0, 120.0))

# Static T_mem curve
lines!(ax_m, collect(rho_range), T_mem_curve;
    color = C_TXT, linewidth = 2.0)

# ρ* where T_mem = 55 steps (Lorenz correlation time) → ρ* = exp(-1/55) ≈ 0.982
rho_match = exp(-1.0 / 55.0)
vlines!(ax_m, [rho_match];
    color = (C_PRED, 0.7), linewidth = 1.5, linestyle = :dash)
text!(ax_m, rho_match - 0.005, 105.0;
    text = L"\rho^* \approx 0.98\;\text{(matches Lorenz }T_{corr}\text{)}",
    color = C_PRED, fontsize = 17, align = (:right, :bottom))
# Also mark ρ=0.7 (best in N=50 sweep: suboptimal due to small reservoir)
vlines!(ax_m, [0.7];
    color = (C_DIM, 0.5), linewidth = 1.2, linestyle = :dot)
text!(ax_m, 0.72, 20.0;
    text = L"\rho=0.7\;\text{(N=50 sweep)}", color = C_DIM, fontsize = 19)

# Annotation for Lorenz correlation time ~55 steps
hlines!(ax_m, [55.0];
    color = (C_ATT, 0.5), linewidth = 1.5, linestyle = :dash)
text!(ax_m, 0.50, 57.0;
    text = L"\sim 55\;\text{steps (Lorenz corr.)}", color = C_ATT, fontsize = 17)

# ── Observables ───────────────────────────────────────────────────────────────
n_show = Observable(2)
steps_obs = @lift collect(steps_axis)[1:$n_show]

for (i, (nrm, col, lbl)) in enumerate(zip(norms_all, rho_colors, rho_labels))
    nrm_safe = max.(nrm, 1e-4)
    lines!(ax_n, steps_obs, @lift(nrm_safe[1:$n_show]);
        color = col, linewidth = 2.2, label = lbl)
end
axislegend(ax_n; position = :rt, labelsize = 17)

step_label = @lift "step $(collect(steps_axis)[$n_show])"
Label(fig[2, :], step_label; color = C_DIM, fontsize = 18)

record(fig, out("s05_spectral_radius.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * (T_UNDRIVEN + 1)))
end
println("✓  s05_spectral_radius.mp4")
