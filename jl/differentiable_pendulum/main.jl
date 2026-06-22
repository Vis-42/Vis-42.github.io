using Pkg
Pkg.activate(@__DIR__)
import Random

include(joinpath(@__DIR__, "src", "PendulumSim.jl"))
include(joinpath(@__DIR__, "src", "GradientCalc.jl"))
include(joinpath(@__DIR__, "src", "ParameterInference.jl"))
include(joinpath(@__DIR__, "src", "Analysis.jl"))
using .PendulumSim, .GradientCalc, .ParameterInference, .Analysis

mkpath(joinpath(@__DIR__, "outputs"))

println("Differentiable Pendulum — Parameter Inference")
println("="^60)

true_params = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81)
u0          = [π/4, 0.0, π/6, 0.0]   # θ1, ω1, θ2, ω2
tspan       = (0.0, 2.0)   # shorter window → smoother loss landscape
dt          = 0.05

noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2]
results = []

for σ in noise_levels
    println("\nNoise σ = $σ")
    times, obs, clean = PendulumSim.simulate_noisy(true_params, u0, tspan, dt; σ=σ)
    cfg = InferenceConfig(lr=0.02, n_steps=120, verbose=false)
    res = ParameterInference.infer_parameters(obs, u0, true_params, tspan, dt; cfg=cfg)
    push!(results, res)
    err = res.param_error_pct
    inf = res.inferred_params
    println("  Inferred: m1=$(round(inf.m1;digits=3)) m2=$(round(inf.m2;digits=3)) L1=$(round(inf.L1;digits=3)) L2=$(round(inf.L2;digits=3))")
    println("  Error: m1=$(round(err[1];digits=1))%  m2=$(round(err[2];digits=1))%  L1=$(round(err[3];digits=1))%  L2=$(round(err[4];digits=1))%")
    println("  Final loss: $(round(res.final_loss;sigdigits=4))")
end

Analysis.save_summary(results, noise_levels, joinpath(@__DIR__, "outputs", "inference_summary.csv"))
println("\nSaved to outputs/inference_summary.csv")

# Save convergence data for best result (low noise)
res0 = results[1]
open(joinpath(@__DIR__, "outputs", "convergence.csv"), "w") do f
    write(f, "step,loss,m1,m2,L1,L2\n")
    for i in 1:length(res0.loss_history)
        ph = res0.param_history[i,:]
        write(f, "$i,$(res0.loss_history[i]),$(ph[1]),$(ph[2]),$(ph[3]),$(ph[4])\n")
    end
end
println("Saved convergence to outputs/convergence.csv")
println("\nDone.")
