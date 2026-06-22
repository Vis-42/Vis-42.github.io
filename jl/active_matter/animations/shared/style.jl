# shared/style.jl — colors, theme, resolution constants for all slides

using CairoMakie

const C_FLOCK = RGBf(0.22, 0.55, 0.95)   # blue   — ordered flock
const C_DIS   = RGBf(0.88, 0.28, 0.25)   # red    — disordered gas
const C_PHI   = RGBf(0.15, 0.80, 0.72)   # teal   — order parameter φ
const C_MSD   = RGBf(0.96, 0.60, 0.18)   # amber  — MSD
const C_THEO  = RGBf(0.68, 0.30, 0.92)   # violet — theory curve
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
        Legend = (
            backgroundcolor = RGBf(0.08, 0.08, 0.10),
            framecolor = RGBf(0.22, 0.22, 0.28),
            labelcolor = C_TXT,
            labelsize = 18,
        ),
    )
end

out(name) = joinpath(@__DIR__, "..", "output", name)

# ── Staging helpers (3B1B-style timeline reveals) ─────────────────────────────
# smoothstep easing: slow-in, slow-out
ease(t) = (u = clamp(t, 0.0, 1.0); u * u * (3.0 - 2.0u))

# phase(f, a, b): eased 0→1 as global progress f sweeps the window [a, b];
# 0 before a, 1 after b. Stages entrances, draw-ons, and fades.
phase(f, a, b) = ease((f - a) / (b - a))

# reveal k of n points, eased, over window [a, b]
reveal_n(f, a, b, n) = max(1, round(Int, phase(f, a, b) * n))

# colour with animated alpha over [a, b] — for fade-ins
fadein(c, f, a, b) = (c, phase(f, a, b))
