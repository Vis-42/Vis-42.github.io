using Pkg
Pkg.activate(@__DIR__)
import Random

include(joinpath(@__DIR__, "src", "GoModel.jl"))
include(joinpath(@__DIR__, "src", "MonteCarlo.jl"))
include(joinpath(@__DIR__, "src", "FreeEnergy.jl"))
include(joinpath(@__DIR__, "src", "Analysis.jl"))
using .GoModel, .MonteCarlo, .FreeEnergy, .Analysis
using Statistics

mkpath(joinpath(@__DIR__, "outputs"))

println("Protein Landscape — Gō Model + Replica Exchange MC")
println("="^60)

cfg = GoModel.TrpCageConfig(; epsilon=1.0)
println("Trp-cage model: $(cfg.n_residues) residues, $(length(cfg.native_contacts)) native contacts")

T_ladder = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
println("\nRunning replica exchange MC at T = $(T_ladder)")
println("This may take a minute...")

results = MonteCarlo.replica_exchange_mc(cfg, T_ladder; n_steps=8000, swap_freq=100, seed=42)

println("\nResults:")
println("T      mean_Q   std_Q   mean_E   std_E")
for r in results
    mQ = isempty(r.Q_traj) ? 0.0 : mean(r.Q_traj)
    sQ = isempty(r.Q_traj) ? 0.0 : std(r.Q_traj)
    mE = isempty(r.E_traj) ? 0.0 : mean(r.E_traj)
    sE = isempty(r.E_traj) ? 0.0 : std(r.E_traj)
    println("$(lpad(r.T,5))  $(lpad(round(mQ;digits=3),7))  $(lpad(round(sQ;digits=3),6))  $(lpad(round(mE;digits=1),8))  $(lpad(round(sE;digits=1),6))")
end

# Free energy profile at lowest T
centers, F = FreeEnergy.free_energy_profile(results[1].Q_traj, results[1].T; n_bins=20)
println("\nFree energy profile F(Q) at T=$(results[1].T):")
println("Q      F(Q)")
for (q,f) in zip(centers, F)
    println("$(round(q;digits=3))   $(round(f;digits=3))")
end

# Find folding temperature: where mean Q transitions
mean_Qs = [isempty(r.Q_traj) ? 0.0 : mean(r.Q_traj) for r in results]
T_fold_approx = T_ladder[argmin(abs.(mean_Qs .- 0.5))]
println("\nApproximate folding temperature (where ⟨Q⟩≈0.5): T_f ≈ $(T_fold_approx)")

Analysis.save_summary(results, T_ladder, joinpath(@__DIR__, "outputs", "landscape_summary.csv"))

# Save free energy data
open(joinpath(@__DIR__, "outputs", "free_energy.csv"), "w") do f
    write(f, "Q,F\n")
    for (q,fv) in zip(centers, F)
        write(f, "$(q),$(fv)\n")
    end
end

println("\nSaved outputs to outputs/")
println("Done.")
