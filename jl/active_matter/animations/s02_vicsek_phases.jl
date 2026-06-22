# s02_vicsek_phases.jl, Hook: ordered flock vs disordered gas (live)
# Slide: "Hook: Order from Self-Propulsion"
#
# 3B1B-style: the disordered gas appears first (milling, φ≈0); then the SAME
# model at lower noise fades in beside it and locks into a flock (φ→1). One
# banner drives the point home: identical rule, only η differs.

using CairoMakie, Random
include(joinpath(@__DIR__, "shared", "style.jl"))
include(joinpath(@__DIR__, "shared", "vicsek.jl"))

set_theme!(slide_theme())

N_VIC, RHO_VIC, V0, R_INT = 120, 1.5, 0.03, 1.0
N_FRAMES = 7 * FPS
frac = Observable(0.0)

println("Pre-warming both phases ...")
state_dis = init_vicsek(N_VIC, RHO_VIC; seed=13)
state_ord = init_vicsek(N_VIC, RHO_VIC; seed=7)
for s in 1:200
    vicsek_step!(state_dis, 3.0; v0=V0, r=R_INT, seed_offset=s)
    vicsek_step!(state_ord, 0.5; v0=V0, r=R_INT, seed_offset=s)
end
L = state_dis.L

xd, yd, θd = Observable(copy(state_dis.x)), Observable(copy(state_dis.y)), Observable(copy(state_dis.θ))
xo, yo, θo = Observable(copy(state_ord.x)), Observable(copy(state_ord.y)), Observable(copy(state_ord.θ))
φd = Observable(order_parameter(state_dis.θ)); φo = Observable(order_parameter(state_ord.θ))

# heading → colour with animated alpha
hcols(θ, α) = [RGBAf(0.5+0.45cos(t), 0.5+0.45sin(t), 0.5+0.45cos(t+π), α) for t in θ]

fig = Figure(size = RES)
Label(fig[0, :], L"\text{Same rule for every particle; only the noise changes}";
      color=@lift(fadein(C_TXT,$frac,0.0,0.06)), fontsize=25)

ax_r = Axis(fig[1,2]; title=L"\eta=3.0,\;\text{disordered gas}", titlecolor=C_DIS,
    aspect=DataAspect(), limits=(0,L,0,L)); hidedecorations!(ax_r)
ax_l = Axis(fig[1,1]; title=L"\eta=0.5,\;\text{ordered flock}", titlecolor=C_FLOCK,
    aspect=DataAspect(), limits=(0,L,0,L)); hidedecorations!(ax_l)

uv = L * 0.07
arrows2d!(ax_r, xd, yd, @lift(cos.($θd).*uv), @lift(sin.($θd).*uv);
    color=@lift(hcols($θd, phase($frac,0.04,0.16))), tipwidth=8, shaftwidth=1.6)
arrows2d!(ax_l, xo, yo, @lift(cos.($θo).*uv), @lift(sin.($θo).*uv);
    color=@lift(hcols($θo, phase($frac,0.24,0.40))), tipwidth=8, shaftwidth=1.6)

text!(ax_r, L*0.02, L*0.97; text=@lift("φ = $(round($φd;digits=3))"),
      color=@lift(fadein(C_DIS,$frac,0.10,0.18)), fontsize=22, align=(:left,:top))
text!(ax_l, L*0.02, L*0.97; text=@lift("φ = $(round($φo;digits=3))"),
      color=@lift(fadein(C_FLOCK,$frac,0.30,0.40)), fontsize=22, align=(:left,:top))

Label(fig[2, :], L"\text{Local rule: align to neighbours within radius } r,\;\text{then add a noise kick } \xi\sim\eta";
      color=@lift(fadein(C_DIM,$frac,0.55,0.70)), fontsize=18)

step = Ref(200)
record(fig, out("s02_vicsek_phases.mp4"), 1:N_FRAMES; framerate=FPS) do f
    frac[] = f / N_FRAMES
    step[] += 1; s = step[]
    vicsek_step!(state_dis, 3.0; v0=V0, r=R_INT, seed_offset=s+10000)
    vicsek_step!(state_ord, 0.5; v0=V0, r=R_INT, seed_offset=s)
    xd[]=copy(state_dis.x); yd[]=copy(state_dis.y); θd[]=copy(state_dis.θ); φd[]=order_parameter(state_dis.θ)
    xo[]=copy(state_ord.x); yo[]=copy(state_ord.y); θo[]=copy(state_ord.θ); φo[]=order_parameter(state_ord.θ)
end
println("✓  s02_vicsek_phases.mp4")
