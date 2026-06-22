# s02_lyapunov.jl: Lyapunov exponent convergence + prediction ceiling bar chart
# Slide: "Theory: Lyapunov Exponents"
#
# Left:  running-average Lyapunov exponent converging to λ_max = 0.906
# Right: prediction ceiling t* = T_λ ln(σ_att/ε₀) for 3 initial precisions

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 8 * FPS

# ── Compute Benettin-style running Lyapunov estimate ─────────────────────────
# Use finite-difference perturbation renormalisation over 50 time units
T_LYA  = 50.0
DT_LYA = DT
N_LYA  = round(Int, T_LYA / DT_LYA)

function compute_running_lyapunov()
    u  = [1.0, 0.0, 0.0]
    # warmup
    for _ in 1:round(Int, 5.0 / DT_LYA)
        u = rk4_step_lorenz(u, DT_LYA)
    end
    delta_hat = [1.0, 0.0, 0.0]
    delta_hat ./= norm(delta_hat)
    eps_norm = 1e-8

    sum_log = 0.0
    lya_running = zeros(N_LYA)
    for i in 1:N_LYA
        v  = u .+ eps_norm .* delta_hat
        u2 = rk4_step_lorenz(u, DT_LYA)
        v2 = rk4_step_lorenz(v, DT_LYA)

        dv     = v2 .- u2
        dv_norm = norm(dv)
        sum_log += log(dv_norm / eps_norm)
        lya_running[i] = sum_log / (i * DT_LYA)

        delta_hat = dv ./ dv_norm
        u = u2
    end
    lya_running
end

lya_running = compute_running_lyapunov()
ts_lya = collect(range(DT_LYA, T_LYA; length = N_LYA))

# ── Bar chart data ────────────────────────────────────────────────────────────
σ_att = 12.0   # typical RMS amplitude of Lorenz attractor
precisions = [1e-2, 1e-4, 1e-8]
t_stars = [T_lyapunov * log(σ_att / ε) for ε in precisions]
bar_labels = [L"\varepsilon_0 = 10^{-2}", L"\varepsilon_0 = 10^{-4}", L"\varepsilon_0 = 10^{-8}"]
bar_colors = [C_TRUE, C_PRED, C_RES]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_lya = Axis(fig[1, 1];
    title  = L"\text{Running Lyapunov estimate (Benettin method)}",
    xlabel = L"t\;\text{(time units)}",
    ylabel = L"\hat\lambda_{\max}(t)",
    limits = (0.0, T_LYA, 0.0, 2.5))

t_star_max = maximum(t_stars ./ T_lyapunov)
ax_bar = Axis(fig[1, 2];
    title  = L"t^* = T_\lambda \ln(\sigma_{att}/\varepsilon_0)\;\text{prediction ceiling}",
    xlabel = L"\text{Initial precision } \varepsilon_0",
    ylabel = L"t^*\;\text{(Lyapunov times)}",
    limits = (0.0, 4.0, 0.0, t_star_max * 1.20),
    xticks = ([1, 2, 3], [L"10^{-2}", L"10^{-4}", L"10^{-8}"]))

# Converged λ_max reference line
hlines!(ax_lya, [λ_max];
    color = (C_DIV, 0.7), linewidth = 1.5, linestyle = :dash)
text!(ax_lya, T_LYA * 0.55, λ_max + 0.08;
    text = L"\lambda_{\max} = 0.906", color = C_DIV, fontsize = 18)

# Static bars (shown fully from start)
barplot!(ax_bar, [1, 2, 3], t_stars ./ T_lyapunov;
    color = bar_colors, width = 0.55, strokecolor = :white, strokewidth = 1)

# Value labels above bars
for (i, tv) in enumerate(t_stars ./ T_lyapunov)
    text!(ax_bar, Float64(i), tv + t_star_max * 0.03;
        text = "$(round(tv; digits=1)) LT",
        color = C_TXT, fontsize = 17, align = (:center, :bottom))
end

# ── Observables (animate convergence of Lyapunov estimate) ───────────────────
n_show = Observable(2)
ts_obs  = @lift ts_lya[1:$n_show]
lya_obs = @lift lya_running[1:$n_show]

lines!(ax_lya, ts_obs, lya_obs; color = C_TRUE, linewidth = 2.5)

t_label = @lift "t = $(round(ts_lya[$n_show]; digits=1)) time units"
Label(fig[2, :], t_label; color = C_DIM, fontsize = 18)

record(fig, out("s02_lyapunov.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = max(2, round(Int, frac * N_LYA))
end
println("✓  s02_lyapunov.mp4")
