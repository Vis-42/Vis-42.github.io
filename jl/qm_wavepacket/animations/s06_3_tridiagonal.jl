# s06_3 — Tridiagonal matrix structure (static PNG)
# Slide: "Tridiagonal Structure and the Thomas Algorithm"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))

set_theme!(slide_theme())

n_show = 18   # show a small NxN block for visibility

fig = Figure(size = RES)

# Sparsity pattern of A
ax_A = Axis(fig[1, 1];
    title  = L"\text{Matrix }A\ \text{(tridiagonal: 3 nonzero diagonals)}",
    xlabel = "column j",
    ylabel = "row i",
    yreversed = true,
    limits = (0.5, n_show+0.5, 0.5, n_show+0.5),
    aspect = DataAspect())

ax_B = Axis(fig[1, 2];
    title  = L"\text{Matrix }B = A^\dagger\ \text{(tridiagonal)}",
    xlabel = "column j",
    ylabel = "row i",
    yreversed = true,
    limits = (0.5, n_show+0.5, 0.5, n_show+0.5),
    aspect = DataAspect())

# Build the pattern
rows_diag = 1:n_show
cols_diag = 1:n_show
rows_up   = 1:n_show-1; cols_up = 2:n_show
rows_lo   = 2:n_show;   cols_lo = 1:n_show-1

for ax in (ax_A, ax_B)
    # Main diagonal — filled rects
    scatter!(ax, cols_diag, rows_diag; color = C_HX, markersize = 18, marker = :rect)
    # Off-diagonal — smaller rects
    scatter!(ax, cols_up,   rows_up;   color = C_RE, markersize = 12, marker = :rect)
    scatter!(ax, cols_lo,   rows_lo;   color = C_IM, markersize = 12, marker = :rect)
    # Empty cells — dim dots
    for r in 1:n_show, c in 1:n_show
        abs(r-c) > 1 && scatter!(ax, [c], [r]; color = (C_DIM, 0.08), markersize = 6)
    end
end

Legend(fig[2, :], [
    MarkerElement(color=C_HX, marker=:rect, markersize=14),
    MarkerElement(color=C_RE, marker=:rect, markersize=10),
    MarkerElement(color=C_IM, marker=:rect, markersize=10),
    ],
    [L"A_{ii}\ \text{(main diagonal)}",
     L"A_{i,i+1}\ \text{(super-diagonal)}",
     L"A_{i+1,i}\ \text{(sub-diagonal)}"];
    orientation = :horizontal, tellwidth = false, labelsize = 13)

Label(fig[3, :],
    L"N = 500\ \text{unknowns} \rightarrow \mathcal{O}(N^3)\ \text{dense} \rightarrow \mathcal{O}(N)\ \text{via Thomas algorithm}";
    fontsize = 15, color = C_DIM, tellwidth = false)

save(out("s06_3_tridiagonal.png"), fig; px_per_unit = 2)
println("✓  s06_3_tridiagonal.png")
