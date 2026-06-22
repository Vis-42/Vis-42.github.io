# s04_conservation_phase.jl: Animated phase space + energy trace for 3 systems
# Slide: "6. Conservation Laws II"
#
# Animated: pendulum, Kepler, Duffing: phase portrait + H(t) flatness

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 8 * FPS
TRAIL    = 60

# --- data ---
ts_p, states_p, H_p = pendulum_data(; θ₀=1.2)
ts_k, states_k, H_k = kepler_data(; ecc=0.4)
ts_d, states_d, H_d = duffing_data(; x₀=0.7)

N = min(length(ts_p), length(ts_k), length(ts_d))
step = Observable(1)

fig = Figure(size = RES)
Label(fig[0,:], L"\text{Conservative Systems: Phase Portrait + Energy Conservation}";
    color=C_TXT, fontsize=22)

# Helper: make animated phase ax + energy ax
function make_panels!(fig, row, ts, states, H, xlbl, ylbl, title_str, color)
    ax_ps = Axis(fig[row,1]; title=title_str, xlabel=xlbl, ylabel=ylbl)
    lines!(ax_ps, states[:,1], states[:,2]; color=(C_DIM,0.25), linewidth=0.8)
    trl_x = @lift states[max(1,$step-TRAIL):min($step,size(states,1)), 1]
    trl_y = @lift states[max(1,$step-TRAIL):min($step,size(states,1)), 2]
    lines!(ax_ps, trl_x, trl_y; color=(color,0.85), linewidth=2.5)
    dot_x = @lift [states[min($step,size(states,1)),1]]
    dot_y = @lift [states[min($step,size(states,1)),2]]
    scatter!(ax_ps, dot_x, dot_y; color=C_LIB, markersize=14)

    ax_E = Axis(fig[row,2]; title=L"\text{Energy }H(t)", xlabel=L"t", ylabel=L"H")
    lines!(ax_E, ts, H; color=(color,0.5), linewidth=1.5)
    # drift from initial value
    H_drift = H .- H[1]
    ax_dE = Axis(fig[row,3];
        title = L"\Delta H(t) = H(t)-H(0)",
        xlabel = L"t",
        ylabel = L"\Delta H")
    lines!(ax_dE, ts, H_drift; color=C_ERR, linewidth=1.5)
    hlines!(ax_dE, [0.0]; color=C_THEORY, linestyle=:dash)
    cursor_x = @lift [ts[min($step,length(ts))]]
    cursor_y = @lift [H[min($step,length(H))]]
    scatter!(ax_E, cursor_x, cursor_y; color=C_LIB, markersize=12)
end

make_panels!(fig, 1, ts_p, states_p, H_p,
    L"\theta", L"\omega", L"\text{Pendulum (}\theta_0=1.2\text{)}", C_TRAJ)

make_panels!(fig, 2, ts_k, states_k, H_k,
    L"x", L"y", L"\text{Kepler (}e=0.4\text{)}", C_CONS)

make_panels!(fig, 3, ts_d, states_d, H_d,
    L"x", L"\dot{x}", L"\text{Duffing free (}x_0=0.7\text{)}", C_LIB)

record(fig, out("s04_conservation_phase.mp4"), 1:N_FRAMES; framerate=FPS) do frame
    step[] = round(Int, 1 + (N-1)*(frame-1)/(N_FRAMES-1))
end
println("✓  s04_conservation_phase.mp4")
