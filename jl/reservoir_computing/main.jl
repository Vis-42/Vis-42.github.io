#!/usr/bin/env julia

using LinearAlgebra
using Statistics
using Random
using Printf

include(joinpath(@__DIR__, "src", "ChaoticSystems.jl"))
include(joinpath(@__DIR__, "src", "ESN.jl"))

using .ChaoticSystems
using .ESN

function ensure_outdir(dir::String)
    isdir(dir) || mkpath(dir)
end

function prediction_horizon_val(true_traj, pred_traj; tol=0.5, dt=0.02, lya_time=1.0)
    n = min(size(true_traj,1), size(pred_traj,1))
    ns = sqrt(Statistics.mean(true_traj.^2)) + 1e-12
    for t in 1:n
        err = sqrt(sum((true_traj[t,:] .- pred_traj[t,:]).^2)) / ns
        err > tol && return t * dt / lya_time
    end
    n * dt / lya_time
end

function run_pipeline(; output_dir::String = joinpath(@__DIR__, "outputs"))
    ensure_outdir(output_dir)

    system = "lorenz63"
    params = Dict{Symbol,Float64}()
    dt     = 0.02
    T_train = 40.0
    T_test  = 8.0
    washout = 200
    lya     = 1.0/0.906  # Lorenz Lyapunov time

    println("Generating Lorenz63 trajectory...")
    total_steps = round(Int, (T_train + T_test)/dt) + washout + 100
    times, traj_full = generate_trajectory(system, params, T_train + T_test + dt*washout, dt)
    train_len = round(Int, T_train/dt)
    test_len  = round(Int, T_test/dt)
    println("  Total steps: $(size(traj_full,1)), train: $train_len, test: $test_len")

    # Grid sweep
    sparsity_range    = 0.05:0.05:0.4
    spectral_range    = 0.7:0.05:1.0
    N_reservoir       = 50

    println("\nPhase diagram sweep ($(length(sparsity_range)) × $(length(spectral_range)) = $(length(sparsity_range)*length(spectral_range)) configs)...")

    results = []
    for sp in sparsity_range
        for sr in spectral_range
            cfg = ESNConfig(N_reservoir=N_reservoir, sparsity=sp, spectral_radius=sr,
                            input_scaling=0.5, ridge_alpha=1e-6, seed=42)
            esn = build_esn(cfg; n_input=3)
            train_data = traj_full[1:train_len+washout, :]
            train_esn!(esn, train_data, washout)
            pred = predict_esn(esn, test_len)
            true_test = traj_full[train_len+washout+1:min(train_len+washout+test_len, size(traj_full,1)), :]
            n_cmp = min(size(pred,1), size(true_test,1))
            rmse = sqrt(Statistics.mean((pred[1:n_cmp,:] .- true_test[1:n_cmp,:]).^2))
            ph   = prediction_horizon_val(true_test[1:n_cmp,:], pred[1:n_cmp,:]; tol=0.5, dt=dt, lya_time=lya)
            push!(results, (sp=sp, sr=sr, ph=ph, rmse=rmse))
        end
    end

    # Save phase diagram CSV
    open(joinpath(output_dir, "phase_diagram.csv"), "w") do io
        println(io, "sparsity,spectral_radius,prediction_horizon_lya,rmse")
        for r in results
            @printf(io, "%.3f,%.3f,%.4f,%.6f\n", r.sp, r.sr, r.ph, r.rmse)
        end
    end

    # Best configuration
    best = results[argmax([r.ph for r in results])]
    println("\nBest config: sparsity=$(best.sp), sr=$(best.sr), PH=$(round(best.ph;sigdigits=3)) Lyapunov times")

    # Predict with best config and save
    cfg_best = ESNConfig(N_reservoir=N_reservoir, sparsity=best.sp, spectral_radius=best.sr,
                         input_scaling=0.5, ridge_alpha=1e-6, seed=42)
    esn_best = build_esn(cfg_best; n_input=3)
    train_esn!(esn_best, traj_full[1:train_len+washout, :], washout)
    pred_best = predict_esn(esn_best, test_len)
    true_test  = traj_full[train_len+washout+1:min(train_len+washout+test_len, size(traj_full,1)), :]
    n_cmp = min(size(pred_best,1), size(true_test,1))

    open(joinpath(output_dir, "best_prediction.csv"), "w") do io
        println(io, "t,x_true,y_true,z_true,x_pred,y_pred,z_pred")
        for i in 1:n_cmp
            @printf(io, "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                i*dt, true_test[i,1], true_test[i,2], true_test[i,3],
                pred_best[i,1], pred_best[i,2], pred_best[i,3])
        end
    end

    # Summary
    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "Reservoir Computing — Lorenz63 ESN Phase Diagram")
        println(io, "N_reservoir = $N_reservoir")
        println(io, "Train steps = $train_len, Test steps = $test_len")
        println(io, "Lyapunov time = $(round(lya; sigdigits=4))")
        println(io, "Best sparsity = $(best.sp), spectral_radius = $(best.sr)")
        println(io, "Best prediction horizon = $(round(best.ph; sigdigits=3)) Lyapunov times")
        println(io, "Best RMSE = $(round(best.rmse; sigdigits=4))")
    end

    println("✓ Outputs saved to $output_dir")
    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
    println("Completed reservoir_computing Julia pipeline.")
end
