#!/usr/bin/env julia
#=
export_notebooks.jl — batch-export all Pluto research notebooks to static HTML.

Run from the workspace root:
    julia jl/export_notebooks.jl

Each notebook is executed headlessly and its full output (all cell renders, PlutoUI
widgets, and WGLMakie plots) is saved to website/static/notebooks/<name>.html.
The resulting files are served at https://vis-42.github.io/notebooks/<name>.html
and are linked from project card headings on the website.

Requirements:
    • Julia ≥ 1.10
    • PlutoSliderServer (installed automatically if missing)

Note on WGLMakie: WGLMakie renders are captured at their initial state.
The 3D canvas will be frozen (not live WebGL) in the exported HTML, but
all PlutoUI controls, layout, and text cells export perfectly. If you want
the most complete snapshot, run the notebook fully in the Pluto browser UI
first (which caches outputs), then re-run this script while the notebook
is still loaded — Pluto will reuse the cached outputs.
=#

using Pkg

# ── Install PlutoSliderServer if not already in the global environment ─────────
if !haskey(Pkg.project().dependencies, "PlutoSliderServer")
    @info "Installing PlutoSliderServer…"
    Pkg.add("PlutoSliderServer")
end
using PlutoSliderServer

# ── Notebooks to export: (output_filename_stem, relative_path_from_workspace) ─
const NOTEBOOKS = [
    ("causal_emergence",       "jl/causal_emergence/app.pluto.jl"),
    ("conservation_laws",      "jl/conservation_laws/app.pluto.jl"),
    ("sindy",                  "jl/sindy/app.pluto.jl"),
    ("chimera_states",         "jl/chimera_states/app.pluto.jl"),
    ("kam_henon_heiles",       "jl/kam_henon_heiles/app.pluto.jl"),
    ("spin_glass_nn",          "jl/spin_glass_nn/app.pluto.jl"),
    ("cd_inversion",           "jl/cd_inversion/app.pluto.jl"),
    ("symbolic_regression",    "jl/symbolic_regression/app.pluto.jl"),
    ("transfer_entropy",       "jl/transfer_entropy/app.pluto.jl"),
    ("convergent_cross_mapping","jl/convergent_cross_mapping/app.pluto.jl"),
    ("percolation_simplicial", "jl/percolation_simplicial/app.pluto.jl"),
    ("reservoir_computing",    "jl/reservoir_computing/app.pluto.jl"),
    ("differentiable_pendulum","jl/differentiable_pendulum/app.pluto.jl"),
    ("protein_landscape",      "jl/protein_landscape/app.pluto.jl"),
    ("xrd_inverse",            "jl/xrd_inverse/app.pluto.jl"),
    ("qm_wavepacket",          "jl/qm_wavepacket/app.pluto.jl"),
]

const OUTPUT_DIR = joinpath(@__DIR__, "..", "website", "static", "notebooks")
mkpath(OUTPUT_DIR)

failed = String[]

for (name, relpath) in NOTEBOOKS
    nb_path = joinpath(@__DIR__, "..", relpath)
    out_path = joinpath(OUTPUT_DIR, name * ".html")

    if !isfile(nb_path)
        @warn "Notebook not found, skipping: $nb_path"
        push!(failed, name)
        continue
    end

    @info "Exporting $name…"
    try
        # PlutoSliderServer exports the notebook and writes the HTML file.
        # The exported file is placed in OUTPUT_DIR with the notebook's original
        # filename.  We rename it to our canonical <name>.html afterwards.
        PlutoSliderServer.export_notebook(nb_path;
            Export_output_dir         = OUTPUT_DIR,
            Export_create_index       = false,
            Export_baked_state        = true,
            Export_offer_binder       = false,
            Export_disable_ui         = false,
        )
        # PlutoSliderServer names the output file after the input file.
        auto_name = joinpath(OUTPUT_DIR, basename(nb_path) * ".html")
        # Some versions produce basename without extension + .html:
        alt_name  = joinpath(OUTPUT_DIR, splitext(basename(nb_path))[1] * ".html")
        src = isfile(auto_name) ? auto_name :
              isfile(alt_name)  ? alt_name  : nothing

        if !isnothing(src) && src != out_path
            mv(src, out_path; force=true)
        end

        if isfile(out_path)
            sz = round(stat(out_path).size / 1024; digits=1)
            @info "  ✓ $(name).html  ($(sz) KB)"
        else
            @warn "  Output file not found after export for $name"
            push!(failed, name)
        end
    catch e
        @warn "  ✗ Failed to export $name: $e"
        push!(failed, name)
    end
end

println("\n── Export summary ─────────────────────────────────")
n_ok = length(NOTEBOOKS) - length(failed)
println("  Succeeded : $n_ok / $(length(NOTEBOOKS))")
isempty(failed) || println("  Failed    : $(join(failed, ", "))")
println()
println("HTML files written to: $OUTPUT_DIR")
println()
println("Next steps:")
println("  git add website/static/notebooks/")
println("  git commit -m 'chore: export Pluto notebooks to static HTML'")
println("  git push")
println("  → deploy runs Hugo; files served at /notebooks/<name>.html")
