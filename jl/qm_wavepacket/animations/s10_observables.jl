# s10 — Observable quantities: norm, T proxy, R proxy, ⟨x⟩ as time traces
# Slide: "Observable Quantities and Live Metrics"
#
# Uses a pre-computed analytical free Gaussian to build illustrative traces of all
# four observables, then reveals them progressively. The heavy real-time simulation
# is shown via a screen recording of the notebook on the actual slide.

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 7 * FPS
const N_PTS    = 200       # number of time samples to show

vg = ħ * k0_def / mₑ
t_end = L / vg            # time for packet to cross domain

# Pre-compute all four observables analytically for a free Gaussian approaching a
# barrier region (illustrative, not a full CN solve)
function observables_illustrative(n)
    ts = range(0.0, t_end; length=n)
    norms  = ones(n)                              # CN conserves norm exactly

    # ⟨x⟩: packet moves at v_g, eventually exits → flatten at L
    xcom   = clamp.(x0_def .+ vg .* ts, 0.0, L) .* 1e9

    # T proxy: probability in right half — rises as packet crosses x = L/2
    t_cross = (L/2 - x0_def) / vg
    T_vals  = @. 0.5 * (1.0 + tanh((ts - t_cross) / (t_end * 0.07))) * 0.38
    # R proxy: mirror of T with a smaller residual (some tunneling)
    R_vals  = @. 0.5 * (1.0 - tanh((ts - t_cross) / (t_end * 0.07))) * 0.60

    ts ./ t_end, norms, xcom, T_vals, R_vals
end

τ, norms, xcom, T_vals, R_vals = observables_illustrative(N_PTS)

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(size = RES)
Label(fig[0, :], "10 — Observable Quantities and Live Metrics";
    fontsize = 20, color = C_TXT, tellwidth = false)

ax_n = Axis(fig[1, 1];
    title  = L"\text{Norm}\ \int|\psi|^2\,dx",
    xlabel = "simulation progress",
    ylabel = L"\int|\psi|^2\,dx",
    limits = (0, 1, 0.0, 1.25))

ax_T = Axis(fig[1, 2];
    title  = L"\text{Right-half probability}\ (T\ \text{proxy})",
    xlabel = "simulation progress",
    ylabel = "probability",
    limits = (0, 1, 0, 0.55))

ax_R = Axis(fig[2, 1];
    title  = L"\text{Left-half probability}\ (R\ \text{proxy})",
    xlabel = "simulation progress",
    ylabel = "probability",
    limits = (0, 1, 0, 0.75))

ax_x = Axis(fig[2, 2];
    title  = L"\text{Centre of mass}\ \langle x \rangle",
    xlabel = "simulation progress",
    ylabel = L"\langle x \rangle\ \text{(nm)}",
    limits = (0, 1, 0, L*1e9))

rowsize!(fig.layout, 1, Relative(0.45))
rowsize!(fig.layout, 2, Relative(0.45))

# Reference lines
hlines!(ax_n, [1.0]; color = (C_DIM, 0.4), linestyle = :dash, linewidth = 1.2)
hlines!(ax_x, [L*1e9/2]; color = (C_DIM, 0.3), linestyle = :dash, linewidth = 1.0)

# Dim full curves as background guide
lines!(ax_n, τ, norms;  color = (C_IM, 0.15), linewidth = 1.5)
lines!(ax_T, τ, T_vals; color = (C_HX, 0.15), linewidth = 1.5)
lines!(ax_R, τ, R_vals; color = (C_RE, 0.15), linewidth = 1.5)
lines!(ax_x, τ, xcom;   color = (C_PR, 0.15), linewidth = 1.5)

# Progressive reveal
n_show = Observable(1)
lines!(ax_n, @lift(τ[1:$n_show]),     @lift(norms[1:$n_show]);  color = C_IM, linewidth = 2.5)
lines!(ax_T, @lift(τ[1:$n_show]),     @lift(T_vals[1:$n_show]); color = C_HX, linewidth = 2.5)
lines!(ax_R, @lift(τ[1:$n_show]),     @lift(R_vals[1:$n_show]); color = C_RE, linewidth = 2.5)
lines!(ax_x, @lift(τ[1:$n_show]),     @lift(xcom[1:$n_show]);   color = C_PR, linewidth = 2.5)

# Moving dot on each trace
for (ax, data, col) in [(ax_n, norms, C_IM), (ax_T, T_vals, C_HX),
                         (ax_R, R_vals, C_RE), (ax_x, xcom,  C_PR)]
    scatter!(ax, @lift([τ[$n_show]]), @lift([data[$n_show]]);
        color = col, markersize = 12)
end

# Annotation: norm stays flat until absorption — inside limits [0, 1.25]
text!(ax_n, 0.05, 1.18;
    text = L"\text{CN: norm = 1 exactly}",
    color = C_DIM, fontsize = 12)

record(fig, out("s10_observables.mp4"), 1:N_FRAMES; framerate = FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * N_PTS))
end
println("✓  s10_observables.mp4")
