"""Differentiable pendulum parameter inference — main runner."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

from pendulum_sim import PendulumParams, simulate_noisy, simulate
from parameter_inference import InferenceConfig, infer_parameters
from analysis import plot_inference_convergence, plot_trajectory_comparison, save_summary
import numpy as np

def main():
    print("Differentiable Pendulum — Parameter Inference")
    print("=" * 60)

    true_params = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
    u0     = np.array([np.pi/4, 0.0, np.pi/6, 0.0])
    tspan  = (0.0, 2.0); dt = 0.05  # shorter window for stability
    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2]
    results = []

    for σ in noise_levels:
        print(f"\nNoise σ = {σ}")
        times, obs, clean = simulate_noisy(true_params, u0, tspan, dt, sigma=σ, seed=42)
        cfg = InferenceConfig(lr=0.02, n_steps=120)
        res = infer_parameters(obs, u0, true_params, tspan, dt, cfg=cfg)
        results.append(res)
        inf = res.inferred_params; err = res.param_error_pct
        print(f"  Inferred: m1={inf.m1:.3f} m2={inf.m2:.3f} L1={inf.L1:.3f} L2={inf.L2:.3f}")
        print(f"  Errors: {err[0]:.1f}% {err[1]:.1f}% {err[2]:.1f}% {err[3]:.1f}%")

    save_summary(results, noise_levels, "outputs/inference_summary.csv")
    plot_inference_convergence(results[0], "outputs/convergence.png")

    # Plot trajectory comparison for noise=0
    _, obs0, clean0 = simulate_noisy(true_params, u0, tspan, dt, sigma=0.0)
    _, inf_traj0 = simulate(results[0].inferred_params, u0, tspan, dt)
    plot_trajectory_comparison(clean0, inf_traj0, obs0, "outputs/trajectory_comparison.png")
    print("\nOutputs saved to outputs/"); print("Done.")


if __name__ == "__main__":
    main()
