# s03_sindy_reconstruction.jl: Animated true vs reconstructed phase portrait
# Slide: "4. SINDy III: Reconstruction validation"
#
# Animated: shared time cursor on true (blue) vs SINDy-reconstructed (green)

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 8 * FPS
TRAIL    = 80

# --- Van der Pol (noise = 0) ---
ts_clean, states_clean, dt_clean = gen_trajectory("van_der_pol"; noise=0.0)
dX_c   = finite_diff(states_clean, dt_clean)
Θ_c, _ = build_library(states_clean, 3, false)
xi_c   = stlsq(Θ_c, dX_c, 0.1)
u0_c   = [2.0, 0.0]
recon_c = sindy_reconstruct(xi_c, u0_c, size(states_clean,1), dt_clean, 3, false)
nv_c    = sum(.!isnan.(recon_c[:,1]))

# --- Van der Pol (noise = 0.05) ---
ts_noisy, states_noisy, dt_n = gen_trajectory("van_der_pol"; noise=0.05)
dX_n   = finite_diff(states_noisy, dt_n)
Θ_n, _ = build_library(states_noisy, 3, false)
xi_n   = stlsq(Θ_n, dX_n, 0.15)
u0_n   = [2.0, 0.0]
recon_n = sindy_reconstruct(xi_n, u0_n, size(states_noisy,1), dt_n, 3, false)
nv_n    = sum(.!isnan.(recon_n[:,1]))

N = size(states_clean, 1)
step = Observable(1)

fig = Figure(size = RES)
Label(fig[0,:], L"\text{SINDy Trajectory Reconstruction: Noise }\sigma=0 \text{ vs } \sigma=0.05";
    color=C_TXT, fontsize=22)

# Panel 1: clean, true
ax1 = Axis(fig[1,1]; title=L"\text{True trajectory (}\sigma=0\text{)}",
    xlabel=L"x_1", ylabel=L"x_2")
lines!(ax1, states_clean[:,1], states_clean[:,2]; color=(C_DIM,0.3), linewidth=1.0)
trl_x1 = @lift states_clean[max(1,$step-TRAIL):$step, 1]
trl_y1 = @lift states_clean[max(1,$step-TRAIL):$step, 2]
lines!(ax1, trl_x1, trl_y1; color=C_TRAJ, linewidth=2.5)
dot_x1 = @lift [states_clean[$step,1]]; dot_y1 = @lift [states_clean[$step,2]]
scatter!(ax1, dot_x1, dot_y1; color=C_LIB, markersize=14)

# Panel 2: clean, SINDy
ax2 = Axis(fig[1,2]; title=L"\text{SINDy reconstruction (}\sigma=0\text{)}",
    xlabel=L"x_1", ylabel=L"x_2")
if nv_c > 10
    lines!(ax2, recon_c[1:nv_c,1], recon_c[1:nv_c,2]; color=(C_DIM,0.3), linewidth=1.0)
end
trl_rx1 = @lift begin
    t = min($step, nv_c); recon_c[max(1,t-TRAIL):max(1,t), 1]
end
trl_ry1 = @lift begin
    t = min($step, nv_c); recon_c[max(1,t-TRAIL):max(1,t), 2]
end
lines!(ax2, trl_rx1, trl_ry1; color=C_RECON, linewidth=2.5)
dot_rx1 = @lift [recon_c[min($step,nv_c),1]]
dot_ry1 = @lift [recon_c[min($step,nv_c),2]]
scatter!(ax2, dot_rx1, dot_ry1; color=C_LIB, markersize=14)

# Panel 3: noisy, SINDy
ax3 = Axis(fig[1,3]; title=L"\text{SINDy reconstruction (}\sigma=0.05\text{)}",
    xlabel=L"x_1", ylabel=L"x_2")
if nv_n > 10
    lines!(ax3, recon_n[1:nv_n,1], recon_n[1:nv_n,2]; color=(C_DIM,0.3), linewidth=1.0)
end
trl_rx2 = @lift begin
    t = min($step, nv_n); recon_n[max(1,t-TRAIL):max(1,t), 1]
end
trl_ry2 = @lift begin
    t = min($step, nv_n); recon_n[max(1,t-TRAIL):max(1,t), 2]
end
lines!(ax3, trl_rx2, trl_ry2; color=(C_ERR, 0.8), linewidth=2.5)
dot_rx2 = @lift [recon_n[min($step,nv_n),1]]
dot_ry2 = @lift [recon_n[min($step,nv_n),2]]
scatter!(ax3, dot_rx2, dot_ry2; color=C_LIB, markersize=14)

# Bottom: x₁(t) overlay
ax4 = Axis(fig[2,1:3];
    title  = L"x_1(t) \text{ comparison}",
    xlabel = L"t",
    ylabel = L"x_1")
lines!(ax4, ts_clean, states_clean[:,1]; color=(C_TRAJ,0.7), linewidth=1.5, label=L"\text{True}")
nv_c > 10 && lines!(ax4, ts_clean[1:nv_c], recon_c[1:nv_c,1]; color=C_RECON, linewidth=1.5, linestyle=:dash, label=L"\text{SINDy (}\sigma=0\text{)}")
nv_n > 10 && lines!(ax4, ts_noisy[1:nv_n], recon_n[1:nv_n,1]; color=C_ERR, linewidth=1.5, linestyle=:dot, label=L"\text{SINDy (}\sigma=0.05\text{)}")
cursor_x = @lift [ts_clean[$step]]; cursor_y = @lift [states_clean[$step,1]]
scatter!(ax4, cursor_x, cursor_y; color=C_LIB, markersize=12)
axislegend(ax4; position=:lt, labelsize=15)

record(fig, out("s03_sindy_reconstruction.mp4"), 1:N_FRAMES; framerate=FPS) do frame
    step[] = round(Int, 1 + (N-1)*(frame-1)/(N_FRAMES-1))
end
println("✓  s03_sindy_reconstruction.mp4")
