# s06_reservoir_init.jl: Reservoir weight matrix and eigenvalue spectrum
# Slide: "Architecture: Reservoir Init"
#
# Left:  heatmap of 20×20 sub-block of W_res (sparse pattern)
# Right: eigenvalues in the complex plane, unit circle, spectral radius ring

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 6 * FPS

# ── Build ESN ─────────────────────────────────────────────────────────────────
esn = build_simple_esn(ESNParams(; N_res = 50, sparsity = 0.15,
                                   spectral_radius = 0.7, seed = 42))

W_sub = esn.W_res[1:20, 1:20]
evals  = eigvals(esn.W_res)
ev_re  = real.(evals)
ev_im  = imag.(evals)

# Unit circle
theta = range(0, 2π; length = 300)
uc_x  = cos.(theta)
uc_y  = sin.(theta)

# Spectral radius circle (ρ = 0.7)
sr_x = 0.7 .* cos.(theta)
sr_y = 0.7 .* sin.(theta)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax_h = Axis(fig[1, 1];
    title  = L"W_{res}\text{: 20}\times\text{20 sub-block (sparse pattern)}",
    xlabel = L"\text{column } j",
    ylabel = L"\text{row } i",
    limits = (0.5, 20.5, 0.5, 20.5))

ax_e = Axis(fig[1, 2];
    title  = L"\text{Eigenvalues of }W_{res}\text{ in the complex plane}",
    xlabel = L"\text{Re}(\lambda)",
    ylabel = L"\text{Im}(\lambda)",
    limits = (-1.2, 1.2, -1.2, 1.2),
    aspect = DataAspect())

# Heatmap (static)
heatmap!(ax_h, W_sub;
    colormap = :RdBu, colorrange = (-maximum(abs.(W_sub)), maximum(abs.(W_sub))))

# Unit circle (static)
lines!(ax_e, uc_x, uc_y;
    color = (C_DIM, 0.5), linewidth = 1.2, linestyle = :dash)
text!(ax_e, 0.72, 0.72;
    text = L"|\lambda|=1", color = C_DIM, fontsize = 17)

# Spectral radius ring
lines!(ax_e, sr_x, sr_y;
    color = (C_PRED, 0.7), linewidth = 1.5, linestyle = :dash)
text!(ax_e, 0.50, 0.54;
    text = L"\rho = 0.7", color = C_PRED, fontsize = 17)

# Origin cross-hairs
hlines!(ax_e, [0.0]; color = (C_DIM, 0.3), linewidth = 1.0)
vlines!(ax_e, [0.0]; color = (C_DIM, 0.3), linewidth = 1.0)

# ── Observables: eigenvalues fill in progressively ───────────────────────────
n_show = Observable(0)
ev_re_obs = @lift ev_re[1:$n_show]
ev_im_obs = @lift ev_im[1:$n_show]

scatter!(ax_e, ev_re_obs, ev_im_obs;
    color = C_RES, markersize = 10, marker = :circle)

n_evals = length(evals)
record(fig, out("s06_reservoir_init.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_show[] = round(Int, frac * n_evals)
end
println("✓  s06_reservoir_init.mp4")
