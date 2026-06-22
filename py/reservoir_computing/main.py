"""End-to-end reservoir computing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chaotic_systems import generate_trajectory, lyapunov_time
from esn import ESNConfig, EchoStateNetwork
from phase_diagram import sweep_phase_diagram
from analysis import plot_phase_diagram, plot_prediction, save_summary


def run_pipeline(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    system = "lorenz63"
    dt = 0.02
    T_train = 40.0
    T_test  = 8.0
    washout = 200
    lya = lyapunov_time(system, {}, dt=dt)

    print(f"Generating {system} trajectory...")
    _, traj_full = generate_trajectory(system, {}, T_train + T_test + dt*washout, dt)
    train_len = round(T_train / dt)
    test_len  = round(T_test / dt)
    train_data = traj_full[:train_len + washout]
    true_test  = traj_full[train_len + washout:train_len + washout + test_len]

    print("Sweeping phase diagram...")
    sparsity_range = np.arange(0.05, 0.45, 0.05)
    spectral_range = np.arange(0.70, 1.05, 0.05)
    results = sweep_phase_diagram(system, sparsity_range, spectral_range,
                                   N_reservoir=50, T_train=T_train, T_test=T_test,
                                   dt=dt, washout=100)
    plot_phase_diagram(results, output_dir / "phase_diagram.png")
    save_summary(results, output_dir / "summary.csv")

    best = max(results, key=lambda r: r.prediction_horizon)
    print(f"Best: sparsity={best.sparsity:.2f}, sr={best.spectral_radius:.2f}, "
          f"PH={best.prediction_horizon:.3f} Lya")

    cfg = ESNConfig(N_reservoir=50, sparsity=best.sparsity,
                    spectral_radius=best.spectral_radius, seed=42)
    esn = EchoStateNetwork(cfg, n_input=3)
    esn.train(train_data, washout=200)
    pred = esn.predict(test_len)
    n_cmp = min(len(pred), len(true_test))
    plot_prediction(true_test[:n_cmp], pred[:n_cmp], output_dir / "best_prediction.png")

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"System: {system}\nLyapunov time: {lya:.4f}\n")
        f.write(f"Best sparsity={best.sparsity:.2f}, sr={best.spectral_radius:.2f}\n")
        f.write(f"Best prediction horizon: {best.prediction_horizon:.3f} Lyapunov times\n")
        f.write(f"RMSE: {best.rmse:.4f}\n")

    print(f"✓ Outputs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_pipeline(args.output_dir)
    print("Completed reservoir_computing Python pipeline.")
