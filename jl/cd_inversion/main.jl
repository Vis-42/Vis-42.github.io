using Pkg
Pkg.activate(@__DIR__)
import Random

include(joinpath(@__DIR__, "src", "CDForwardModel.jl"))
include(joinpath(@__DIR__, "src", "TrainingData.jl"))
include(joinpath(@__DIR__, "src", "InversionNetwork.jl"))
include(joinpath(@__DIR__, "src", "Analysis.jl"))
using .CDForwardModel, .TrainingData, .InversionNetwork, .Analysis
using Statistics

mkpath(joinpath(@__DIR__, "outputs"))

println("CD Inversion Network")
println("="^60)

# Generate dataset
println("Generating 1000 synthetic CD spectra...")
wl, spectra, compositions = TrainingData.generate_cd_dataset(1000; noise_σ=0.5, seed=42)
n_train = 800
train_spec = spectra[1:n_train, :]; train_comp = compositions[1:n_train, :]
test_spec  = spectra[n_train+1:end, :]; test_comp = compositions[n_train+1:end, :]
println("  Train: $n_train, Test: $(size(test_spec,1))")

# Train WITH physics constraint
println("\nTraining with physics loss (λ_phys=0.3)...")
net_with = InversionNetwork.build_net(; input_dim=length(wl), hidden_dim=64, seed=42)
losses_with = InversionNetwork.train_net!(net_with, train_spec, train_comp;
    epochs=30, lr=1e-3, lambda_physics=0.3, batch_size=32, seed=0)
println("  Final training loss: $(round(losses_with[end];sigdigits=4))")

# Train WITHOUT physics constraint
println("Training without physics loss...")
net_without = InversionNetwork.build_net(; input_dim=length(wl), hidden_dim=64, seed=42)
losses_without = InversionNetwork.train_net!(net_without, train_spec, train_comp;
    epochs=30, lr=1e-3, lambda_physics=0.0, batch_size=32, seed=0)
println("  Final training loss: $(round(losses_without[end];sigdigits=4))")

# Evaluate
function evaluate_net(net, spectra, compositions)
    N = size(spectra, 1)
    preds = zeros(N, 3)
    for i in 1:N
        preds[i,:] = InversionNetwork.predict_composition(net, spectra[i,:])
    end
    rmse = sqrt(mean((preds .- compositions).^2))
    # Physics satisfaction: how close do predicted fractions sum to 1 (they do by softmax)
    sum_err = mean(abs.(sum(preds;dims=2) .- 1.0))
    # Spectral R²: reconstruct spectrum from predictions
    r2_vals = Float64[]
    for i in 1:N
        rec = InversionNetwork._reconstruct(preds[i,1], preds[i,2], preds[i,3])
        ȳ   = mean(spectra[i,:])
        ss_tot = sum((spectra[i,:] .- ȳ).^2)
        ss_res = sum((spectra[i,:] .- rec).^2)
        push!(r2_vals, max(0.0, 1.0 - ss_res/(ss_tot+1e-6)))
    end
    (test_rmse=rmse, physics_sat=1.0-sum_err, spectral_r2=mean(r2_vals))
end

res_with    = evaluate_net(net_with, test_spec, test_comp)
res_without = evaluate_net(net_without, test_spec, test_comp)

println("\nTest Results:")
println("  With physics:    RMSE=$(round(res_with.test_rmse;sigdigits=3))  spectral_R²=$(round(res_with.spectral_r2;sigdigits=3))")
println("  Without physics: RMSE=$(round(res_without.test_rmse;sigdigits=3))  spectral_R²=$(round(res_without.spectral_r2;sigdigits=3))")

Analysis.save_summary(Dict(pairs(res_with)), Dict(pairs(res_without)),
    joinpath(@__DIR__, "outputs", "cd_summary.csv"))

# Save training curves
open(joinpath(@__DIR__, "outputs", "training_losses.csv"), "w") do f
    write(f, "epoch,loss_with,loss_without\n")
    for i in 1:length(losses_with)
        write(f, "$i,$(losses_with[i]),$(losses_without[i])\n")
    end
end

println("Outputs saved to outputs/")
println("Done.")
