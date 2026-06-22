# shared/style.jl — colors, theme, resolution constants for all slides

using CairoMakie

# Project color palette
const C_TRUE  = RGBf(0.22, 0.55, 0.95)   # blue   — true Lorenz trajectory
const C_PRED  = RGBf(0.96, 0.60, 0.18)   # amber  — ESN predictions
const C_DIV   = RGBf(0.88, 0.28, 0.25)   # red    — diverging / unstable
const C_RES   = RGBf(0.68, 0.30, 0.92)   # violet — reservoir states
const C_ATT   = RGBf(0.15, 0.80, 0.72)   # teal   — attractor
const C_BG    = RGBf(0.00, 0.00, 0.00)
const C_TXT   = RGBf(0.88, 0.88, 0.93)
const C_DIM   = RGBf(0.42, 0.42, 0.48)
const C_GRD   = (RGBf(1.0, 1.0, 1.0), 0.12)

const RES = (1280, 720)
const FPS = 30

function slide_theme()
    Theme(
        backgroundcolor = C_BG,
        textcolor = C_TXT,
        fontsize = 22,
        Axis = (
            backgroundcolor = :transparent,
            spinecolor = RGBf(0.22, 0.22, 0.28),
            xgridcolor = C_GRD,
            ygridcolor = C_GRD,
            xticklabelcolor = C_DIM,
            yticklabelcolor = C_DIM,
            xlabelcolor = C_TXT,
            ylabelcolor = C_TXT,
            titlecolor = C_TXT,
            titlesize = 26,
            titlegap = 10,
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 17,
            yticklabelsize = 17,
        ),
        Axis3 = (
            backgroundcolor = :transparent,
            xgridcolor = (RGBf(1, 1, 1), 0.12),
            ygridcolor = (RGBf(1, 1, 1), 0.12),
            zgridcolor = (RGBf(1, 1, 1), 0.12),
            xticklabelcolor = C_DIM,
            yticklabelcolor = C_DIM,
            zticklabelcolor = C_DIM,
            xlabelcolor = C_TXT,
            ylabelcolor = C_TXT,
            zlabelcolor = C_TXT,
            titlecolor = C_TXT,
            titlesize = 26,
            xlabelsize = 20,
            ylabelsize = 20,
            zlabelsize = 20,
            xticklabelsize = 17,
            yticklabelsize = 17,
            zticklabelsize = 17,
        ),
        Legend = (
            backgroundcolor = RGBf(0.08, 0.08, 0.10),
            framecolor = RGBf(0.22, 0.22, 0.28),
            labelcolor = C_TXT,
            labelsize = 18,
        ),
    )
end

# Output path helper
out(name) = joinpath(@__DIR__, "..", "output", name)
