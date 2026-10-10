# shared/style.jl — colors, theme, resolution constants for all slides

using CairoMakie

# Project color palette
const C_RE = RGBf(0.35, 0.52, 0.90)   # blue-violet : Re(ψ)
const C_IM = RGBf(0.20, 0.75, 0.50)   # green       : Im(ψ)
const C_PR = RGBf(0.96, 0.60, 0.18)   # amber       : |ψ|²
const C_HX = RGBf(0.70, 0.30, 0.95)   # violet      : Argand helix
const C_V = RGBf(0.00, 0.85, 0.85)   # cyan        : V(x)
const C_BG = RGBf(0.00, 0.00, 0.00)   # background
const C_TXT = RGBf(0.88, 0.88, 0.93)   # primary text
const C_DIM = RGBf(0.42, 0.42, 0.48)   # axis ticks / dim labels
const C_GRD = (RGBf(1.0, 1.0, 1.0), 0.17)  # grid lines

const RES = (1280, 720)
const FPS = 30

function slide_theme()
    Theme(
        backgroundcolor=C_BG,
        textcolor=C_TXT,
        fontsize=17,
        Axis=(
            backgroundcolor=:transparent,
            spinecolor=RGBf(0.22, 0.22, 0.28),
            xgridcolor=C_GRD,
            ygridcolor=C_GRD,
            xticklabelcolor=C_DIM,
            yticklabelcolor=C_DIM,
            xlabelcolor=C_TXT,
            ylabelcolor=C_TXT,
            titlecolor=C_TXT,
            titlesize=19,
            titlegap=10,
        ),
        Axis3=(
            backgroundcolor=:transparent,
            xgridcolor=(RGBf(1, 1, 1), 0.12),
            ygridcolor=(RGBf(1, 1, 1), 0.12),
            zgridcolor=(RGBf(1, 1, 1), 0.12),
            xticklabelcolor=C_DIM,
            yticklabelcolor=C_DIM,
            zticklabelcolor=C_DIM,
            xlabelcolor=C_TXT,
            ylabelcolor=C_TXT,
            zlabelcolor=C_TXT,
            titlecolor=C_TXT,
            titlesize=19,
        ),
        Legend=(
            backgroundcolor=RGBf(0.08, 0.08, 0.10),
            framecolor=RGBf(0.22, 0.22, 0.28),
            labelcolor=C_TXT,
        ),
    )
end

# Scaled potential for 2D overlay (maps V peak to fraction of |ψ|² peak)
function scale_V(V, peak_density; frac=0.55)
    Vm = maximum(abs.(V))
    Vm < 1e-60 && return zeros(length(V))
    V ./ Vm .* peak_density .* frac
end

# Output path helper
out(name) = joinpath(@__DIR__, "..", "output", name)
