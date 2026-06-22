# s01_motivation.jl, Active vs passive matter
# Slide: "Active Matter vs Passive Matter"
#
# 3B1B-style: two live ensembles side by side. Passive (thermal) particles
# jiggle in place with no net current; active (self-propelled) particles stream.
# The difference is energy injected at the single-particle level.

using CairoMakie, Random
include(joinpath(@__DIR__, "shared", "style.jl"))

set_theme!(slide_theme())

N, L = 70, 8.0
N_FRAMES = 6 * FPS
rng = MersenneTwister(1)

px = rand(rng, N).*L; py = rand(rng, N).*L
axx = rand(rng, N).*L; ayy = rand(rng, N).*L; aθ = rand(rng, N).*2π
Dp, v0 = 0.10, 0.10

pxo, pyo = Observable(px), Observable(py)
axo, ayo, aθo = Observable(axx), Observable(ayy), Observable(aθ)
acol(θ) = [RGBAf(0.5+0.45cos(t), 0.5+0.45sin(t), 0.5+0.45cos(t+π), 1.0) for t in θ]

fig = Figure(size = RES)
Label(fig[0, :], L"\text{What makes matter active? Energy injected at every particle}";
      color=C_TXT, fontsize=24)

axP = Axis(fig[1,1]; title=L"\text{Passive: thermal equilibrium}", titlecolor=C_DIM,
    aspect=DataAspect(), limits=(0,L,0,L)); hidedecorations!(axP)
axA = Axis(fig[1,2]; title=L"\text{Active: self-propelled}", titlecolor=C_FLOCK,
    aspect=DataAspect(), limits=(0,L,0,L)); hidedecorations!(axA)

scatter!(axP, pxo, pyo; color=C_DIM, markersize=12)
arrows2d!(axA, axo, ayo, @lift(cos.($aθo).*0.45), @lift(sin.($aθo).*0.45);
    color=@lift(acol($aθo)), tipwidth=9, shaftwidth=1.8)

text!(axP, L*0.5, L*0.04; text=L"\langle\mathbf{v}\rangle = 0\text{: no net current}",
      color=C_DIM, fontsize=18, align=(:center,:bottom))
text!(axA, L*0.5, L*0.04; text=L"\text{directed motion, clustering, flow}",
      color=C_FLOCK, fontsize=18, align=(:center,:bottom))

Label(fig[2, :],
    L"\text{Passive obeys detailed balance and fluctuation-dissipation; active breaks both, with no equilibrium analogue}";
    color=C_DIM, fontsize=17)

record(fig, out("s01_motivation.mp4"), 1:N_FRAMES; framerate=FPS) do _
    pxo[] = mod.(pxo[] .+ randn(rng,N).*Dp, L)
    pyo[] = mod.(pyo[] .+ randn(rng,N).*Dp, L)
    aθo[] = aθo[] .+ randn(rng,N).*0.06
    axo[] = mod.(axo[] .+ v0.*cos.(aθo[]), L)
    ayo[] = mod.(ayo[] .+ v0.*sin.(aθo[]), L)
end
println("✓  s01_motivation.mp4")
