# s04_4 — Helix dictionary: pitch/collapse in four potential scenarios
# Slide: "The Helix as a Local Dictionary"

using CairoMakie
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "physics.jl"))

set_theme!(slide_theme())

const N_FRAMES = 5 * FPS   # Static 4-panel, camera slowly rotates

# x domain — short stretch showing a few wavelengths
λ_dB  = 2π / k0_def
x_sh  = range(0.0, 5λ_dB; length=400)
xsh_nm = x_sh .* 1e9
z0    = zeros(length(x_sh))
Ek_v  = ħ^2 * k0_def^2 / (2mₑ)

# Case helpers
function helix_free(x)
    re = cos.(k0_def .* x)
    im = sin.(k0_def .* x)
    re, im
end
function helix_well(x)   # V = -V₀ → higher k
    k1 = k0_def * 1.6
    re = cos.(k1 .* x); im = sin.(k1 .* x); re, im
end
function helix_step(x)   # V = 0.5 Ek → lower k
    k2 = k0_def * sqrt(0.5)
    re = cos.(k2 .* x); im = sin.(k2 .* x); re, im
end
function helix_forbid(x)  # V > E → exponential collapse
    κ   = sqrt(2mₑ * 0.5 * Ek_v) / ħ
    env = exp.(-κ .* x)
    re  = env .* cos.(k0_def .* x[1] .* ones(length(x)))
    im  = env .* sin.(k0_def .* x[1] .* ones(length(x)))
    re, im
end

cases = [
    (helix_free,   L"\text{Free particle} \\ \text{pitch} = \lambda_\text{dB}",         C_RE),
    (helix_well,   L"V < 0\ \text{(well)} \\ \text{pitch shorter, higher }k",           C_IM),
    (helix_step,   L"0 < V < E\ \text{(step)} \\ \text{pitch longer, lower }k",         C_HX),
    (helix_forbid, L"V > E\ \text{(forbidden)} \\ \text{exponential collapse}",          C_PR),
]

fig = Figure(size = RES)
axes3 = []
for (idx, (fn, title, col)) in enumerate(cases)
    row, col_idx = fldmod(idx - 1, 2)
    ax = Axis3(fig[row+1, col_idx+1];
        title     = title,
        titlesize = 14,
        xlabel    = L"x",
        ylabel    = L"\mathrm{Im}",
        zlabel    = L"\mathrm{Re}",
        azimuth   = π/6,
        elevation = π/8)

    xlims!(ax, 0, 5λ_dB*1e9)
    ylims!(ax, -1.4, 1.4)
    zlims!(ax, -1.4, 1.4)

    re_h, im_h = fn(x_sh)
    lines!(ax, xsh_nm, z0, re_h;    color = (col, 0.45), linewidth = 1.5)
    lines!(ax, xsh_nm, im_h, z0;    color = (col, 0.45), linewidth = 1.5)
    lines!(ax, xsh_nm, im_h, re_h;  color = col,         linewidth = 2.5)
    lines!(ax, xsh_nm, z0,   z0;    color = (C_DIM, 0.2), linewidth = 0.7)
    push!(axes3, ax)
end

# Rotate all cameras together
record(fig, out("s04_4_helix_dict.mp4"), 1:N_FRAMES; framerate = FPS) do f
    az = π/6 + (f-1)/N_FRAMES * π/2
    for ax in axes3
        ax.azimuth[] = az
    end
end
println("✓  s04_4_helix_dict.mp4")
