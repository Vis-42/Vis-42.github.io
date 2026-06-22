using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Statistics, Random, LinearAlgebra

include(joinpath(@__DIR__, "src", "Network.jl"))
include(joinpath(@__DIR__, "src", "DataGeneration.jl"))
include(joinpath(@__DIR__, "src", "HessianSpectrum.jl"))
include(joinpath(@__DIR__, "src", "Analysis.jl"))

using .Network, .DataGeneration, .HessianSpectrum, .Analysis

mkpath(joinpath(@__DIR__, "outputs"))
outdir = joinpath(@__DIR__, "outputs")

println("Spin Glass NN — Hessian Spectrum During Training")
println("=" ^ 60)

# Dataset
X_data, y_data = make_spiral(80; noise=0.15, seed=42)
println("Dataset: spiral, N=$(size(X_data,1))")

# MLP config
cfg = MLPConfig(n_in=2, n_hidden=20, n_out=2, lr=0.02)
mlp = MLP(cfg; seed=7)

n_epochs      = 500
snapshot_every = 50
snapshots = []   # (epoch, diag_H, loss_val)

println("Training ...")
for epoch in 1:n_epochs
    gradient_step!(mlp, X_data, y_data)
    if epoch % snapshot_every == 0
        l = loss(mlp, X_data, y_data)
        diag_H = hessian_diagonal(mlp, X_data, y_data; eps=5e-4)
        push!(snapshots, (epoch=epoch, diag_H=diag_H, loss=l))
        ip = ipr(diag_H)
        pr = participation_ratio(diag_H)
        n_near_zero = sum(abs.(diag_H) .< 0.01)
        println("  epoch $epoch: loss=$(round(l;digits=4))  IPR=$(round(ip;digits=4))  PR=$(round(pr;digits=1))  |Hₖₖ|<0.01: $n_near_zero")
    end
end

losses  = [s.loss for s in snapshots]
iprs    = [ipr(s.diag_H) for s in snapshots]

save_summary(losses, iprs, joinpath(outdir, "summary.csv"); snapshot_every=snapshot_every)
println("Saved: outputs/summary.csv")

open(joinpath(outdir, "summary.txt"), "w") do f
    println(f, "Spin Glass NN — Spiral Dataset (N=160, 2 hidden layers of 20 units)")
    println(f, "Final loss: $(round(losses[end];digits=4))")
    println(f, "Final IPR: $(round(iprs[end];digits=4))")
    println(f, "Initial IPR: $(round(iprs[1];digits=4))")
    println(f, "IPR decreased from $(round(iprs[1];digits=3)) to $(round(iprs[end];digits=3)) during training")
end
println("Saved: outputs/summary.txt")

try
    using Plots
    epochs_x = [s.epoch for s in snapshots]
    p1 = Plots.plot(epochs_x, losses;
        label="Loss", lw=2, color=:violet,
        xlabel="Epoch", ylabel="Loss",
        title="Training Loss")
    Plots.savefig(p1, joinpath(outdir, "loss_curve.png"))

    p2 = Plots.plot(epochs_x, iprs;
        label="IPR", lw=2, color=:green,
        xlabel="Epoch", ylabel="IPR",
        title="Hessian IPR Evolution")
    Plots.savefig(p2, joinpath(outdir, "ipr_evolution.png"))
    println("Saved: outputs/loss_curve.png, ipr_evolution.png")
catch e
    println("Plots not available: $e")
end

println("\nDone.")
