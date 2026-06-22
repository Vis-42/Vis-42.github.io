# shared/style.jl — colors, theme, resolution for physics_discovery slides

using CairoMakie

const C_TRAJ   = RGBf(0.22, 0.55, 0.95)   # blue   — true trajectory
const C_RECON  = RGBf(0.15, 0.85, 0.50)   # green  — SINDy reconstruction
const C_LIB    = RGBf(0.96, 0.60, 0.18)   # amber  — library terms / GP
const C_CONS   = RGBf(0.68, 0.30, 0.92)   # violet — conserved quantity
const C_ERR    = RGBf(0.88, 0.28, 0.25)   # red    — error / noise
const C_THEORY = RGBf(0.80, 0.80, 0.30)   # yellow — analytic reference
const C_BG     = RGBf(0.00, 0.00, 0.00)
const C_TXT    = RGBf(0.88, 0.88, 0.93)
const C_DIM    = RGBf(0.42, 0.42, 0.48)
const C_GRD    = (RGBf(1.0, 1.0, 1.0), 0.12)

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
        Legend = (
            backgroundcolor = RGBf(0.08, 0.08, 0.10),
            framecolor = RGBf(0.22, 0.22, 0.28),
            labelcolor = C_TXT,
            labelsize = 18,
        ),
    )
end

out(name) = joinpath(@__DIR__, "..", "output", name)
