# s10_phase_diagram.jl: Phase diagram: prediction horizon over (sparsity, spectral_radius)
# Slide: "Results: Phase Diagram"
#
# Heatmap of prediction_horizon_lya over (sparsity, spectral_radius)
# Clipped to [0, 4] LT; cells fill progressively row by row

using CairoMakie
using DelimitedFiles
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "lorenz.jl"))

set_theme!(slide_theme())

N_FRAMES = 8 * FPS

# ── Load phase diagram data ───────────────────────────────────────────────────
csv_path = joinpath(@__DIR__, "..", "outputs", "phase_diagram.csv")
raw = readdlm(csv_path, ','; header = true)
data_mat = Float64.(raw[1])
# columns: sparsity, spectral_radius, prediction_horizon_lya, rmse

sparsities = sort(unique(data_mat[:, 1]))
spec_radii = sort(unique(data_mat[:, 2]))
n_s = length(sparsities)
n_r = length(spec_radii)

# Build PH matrix: rows = sparsity, cols = spectral_radius
PH_mat = zeros(n_s, n_r)
for row in eachrow(data_mat)
    si = findfirst(==(row[1]), sparsities)
    ri = findfirst(==(row[2]), spec_radii)
    (si === nothing || ri === nothing) && continue
    PH_mat[si, ri] = row[3]
end

# Clip outliers (numerical blow-up at some parameter combinations)
ph_finite = PH_mat[isfinite.(PH_mat) .& (PH_mat .< 100)]
ph_cap    = isempty(ph_finite) ? 4.0 : min(ceil(maximum(ph_finite) * 1.1; digits=1), 10.0)
PH_clipped = clamp.(PH_mat, 0.0, ph_cap)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = RES)

ax = Axis(fig[1, 1];
    title  = L"\text{Prediction Horizon (Lyapunov times): phase diagram}",
    xlabel = L"\text{Sparsity } p_{nz}",
    ylabel = L"\text{Spectral radius } \rho",
    xticks = (1:n_s, ["$(round(s; digits=2))" for s in sparsities]),
    yticks = (1:n_r, ["$(round(r; digits=2))" for r in spec_radii]))

# Animated: reveal cells row by row (each row = one sparsity value)
n_rows_show = Observable(0)

# We'll use a matrix Observable updated each frame
ph_anim = Observable(fill(NaN, n_s, n_r))

hm = heatmap!(ax, ph_anim;
    colormap = :viridis,
    colorrange = (0.0, ph_cap),
    nan_color = RGBf(0.08, 0.08, 0.10))

Colorbar(fig[1, 2], hm;
    label = L"\text{Prediction Horizon (LT)}",
    labelcolor = C_TXT,
    ticklabelcolor = C_DIM,
    width = 20)

# Best cell annotation
best_ph, best_idx = findmax(PH_clipped)
best_s = sparsities[best_idx[1]]
best_r = spec_radii[best_idx[2]]
text!(ax, Float64(best_idx[1]), Float64(best_idx[2]) + 0.35;
    text = "best\n$(round(best_ph; digits=2)) LT",
    color = C_TXT, fontsize = 17, align = (:center, :bottom))

record(fig, out("s10_phase_diagram.mp4"), 1:N_FRAMES; framerate = FPS) do f
    frac     = (f - 1) / (N_FRAMES - 1)
    n_r_show = round(Int, frac * n_r)   # reveal spectral_radius columns progressively
    mat      = fill(NaN, n_s, n_r)
    if n_r_show > 0
        mat[:, 1:n_r_show] .= PH_clipped[:, 1:n_r_show]
    end
    ph_anim[] = mat
end
println("✓  s10_phase_diagram.mp4")
