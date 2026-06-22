# s05_kernel_invariant.jl: Kernel-discovered invariant vs true energy
# Slide: "5. Conservation Laws I: Kernel method"
#
# Three panels: H_kernel(xᵢ) vs H_true(xᵢ), temporal trace, γ/α parameter sweep

using CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "discovery.jl"))

set_theme!(slide_theme())
N_FRAMES = 4 * FPS

println("Running kernel conservation law discovery ...")
ts_p, states_p, H_p = pendulum_data(; θ₀=1.2, n=500)
dt_p = ts_p[2] - ts_p[1]
H_k_p = discover_invariant(states_p, dt_p; γ=1.0, α=1e-5)

ts_k, states_k, H_k_energy = kepler_data(; ecc=0.3, n=500)
dt_k = ts_k[2] - ts_k[1]
H_k_k = discover_invariant(states_k, dt_k; γ=0.5, α=1e-5)

# Normalize for comparison
function normalize_corr(a, b)
    a_n = (a .- mean(a)) ./ (std(a) + 1e-12)
    b_n = (b .- mean(b)) ./ (std(b) + 1e-12)
    a_n, b_n, cor(a, b)
end
H_p_n, H_kp_n, ρ_p = normalize_corr(H_p, H_k_p)
H_k_n, H_kk_n, ρ_k = normalize_corr(H_k_energy, H_k_k)

# Parameter sweep: γ vs correlation (pendulum)
println("Sweeping γ ...")
γ_range = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
ρ_γ = Float64[]
for γ in γ_range
    H_tmp = discover_invariant(states_p, dt_p; γ=γ, α=1e-5)
    push!(ρ_γ, abs(cor(H_p, H_tmp)))
end

fig = Figure(size = RES)
Label(fig[0,:], L"\text{Kernel Conservation Law Discovery: RBF Generalized Eigenproblem}";
    color=C_TXT, fontsize=22)

# Pendulum: scatter H_true vs H_kernel
ax1 = Axis(fig[1,1];
    title  = L"\text{Pendulum: True }H \text{ vs Kernel}",
    xlabel = L"H_{\text{true}}",
    ylabel = L"H_{\text{kernel}} \text{ (normalized)}")
n_show = Observable(1)
N_pts = length(H_p_n)
scatter!(ax1, @lift(H_p_n[1:$n_show]), @lift(H_kp_n[1:$n_show]); color=C_CONS, markersize=4, alpha=0.5)
xlims = extrema(H_p_n)
lines!(ax1, collect(xlims), collect(xlims); color=C_THEORY, linestyle=:dash, linewidth=1.5)
text!(ax1, xlims[1] + 0.1*(xlims[2]-xlims[1]),
           xlims[2] - 0.2*(xlims[2]-xlims[1]);
    text="ρ = $(round(ρ_p; digits=3))", color=C_THEORY, fontsize=18)

# Kepler: scatter
ax2 = Axis(fig[1,2];
    title  = L"\text{Kepler: True }H \text{ vs Kernel}",
    xlabel = L"H_{\text{true}}",
    ylabel = L"H_{\text{kernel}} \text{ (normalized)}")
scatter!(ax2, @lift(H_k_n[1:$n_show]), @lift(H_kk_n[1:$n_show]); color=C_TRAJ, markersize=4, alpha=0.5)
xlims_k = extrema(H_k_n)
lines!(ax2, collect(xlims_k), collect(xlims_k); color=C_THEORY, linestyle=:dash, linewidth=1.5)
text!(ax2, xlims_k[1] + 0.1*(xlims_k[2]-xlims_k[1]),
           xlims_k[2] - 0.2*(xlims_k[2]-xlims_k[1]);
    text="ρ = $(round(ρ_k; digits=3))", color=C_THEORY, fontsize=18)

# γ sweep
ax3 = Axis(fig[1,3];
    title  = L"\text{Kernel width }\gamma\text{ vs discovery correlation}",
    xlabel = L"\gamma",
    ylabel = L"|\rho(H_{\text{kernel}}, H_{\text{true}})|",
    xscale = log10,
    limits = (nothing, (0.0, 1.05)))
n_γ = Observable(1)
lines!(ax3, @lift(γ_range[1:$n_γ]), @lift(ρ_γ[1:$n_γ]); color=C_CONS, linewidth=3.0)
scatter!(ax3, @lift(γ_range[1:$n_γ]), @lift(ρ_γ[1:$n_γ]); color=C_CONS, markersize=12)
idx_best = argmax(ρ_γ)
scatter!(ax3, [γ_range[idx_best]], [ρ_γ[idx_best]]; color=C_LIB, markersize=16, marker=:star5)
text!(ax3, γ_range[idx_best]*1.2, ρ_γ[idx_best]-0.06;
    text="γ* = $(γ_range[idx_best])", color=C_LIB, fontsize=15)

# Temporal trace
ax4 = Axis(fig[2,1:3];
    title  = L"\text{Temporal trace: }H_{\text{kernel}}(t) \text{ should be flat (pendulum)}",
    xlabel = L"t",
    ylabel = L"H")
lines!(ax4, @lift(ts_p[1:$n_show]), @lift(H_kp_n[1:$n_show]); color=C_CONS, linewidth=2.0, label=L"H_{\text{kernel}}")
lines!(ax4, @lift(ts_p[1:$n_show]), @lift(H_p_n[1:$n_show]);  color=C_THEORY, linewidth=1.5, linestyle=:dash, label=L"H_{\text{true}}")
axislegend(ax4; position=:rt, labelsize=15)

Label(fig[3,:],
    L"\text{RBF kernel finds the direction in function space with zero temporal drift.}\;\gamma^*\approx 1\text{ matches the median pairwise distance heuristic.}";
    color=C_DIM, fontsize=14)

record(fig, out("s05_kernel_invariant.mp4"), 1:N_FRAMES; framerate=FPS) do f
    n_show[] = max(1, round(Int, (f / N_FRAMES) * N_pts))
    n_γ[]    = max(1, round(Int, (f / N_FRAMES) * length(γ_range)))
end
println("✓  s05_kernel_invariant.mp4")
