"""Protein landscape main runner."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

from go_model import TrpCageConfig
from monte_carlo import replica_exchange_mc
from free_energy import free_energy_profile
from analysis import plot_free_energy, plot_folding_trajectory, save_summary
import numpy as np

print("Protein Landscape — Gō Model + Replica Exchange MC")
print("=" * 60)

cfg = TrpCageConfig(epsilon=1.0)
print(f"Trp-cage: {cfg.n_residues} residues, {len(cfg.native_contacts)} native contacts")

T_ladder = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
print(f"\nRunning replica exchange MC at T={T_ladder}")

results = replica_exchange_mc(cfg, T_ladder, n_steps=8000, swap_freq=100, seed=42)

print("\nT      mean_Q   std_Q    mean_E    n_samples")
for r in results:
    mQ = float(np.mean(r.Q_traj)) if r.Q_traj else 0.0
    sQ = float(np.std(r.Q_traj)) if len(r.Q_traj) > 1 else 0.0
    mE = float(np.mean(r.E_traj)) if r.E_traj else 0.0
    print(f"  {r.T:.1f}    {mQ:.3f}    {sQ:.3f}    {mE:.2f}    {len(r.Q_traj)}")

centers, F = free_energy_profile(results[0].Q_traj, results[0].T, n_bins=20)
plot_free_energy(centers, F, results[0].T, "outputs/free_energy.png")
plot_folding_trajectory(results[0].Q_traj, results[0].E_traj, results[0].T, "outputs/folding_traj.png")
save_summary(results, "outputs/landscape_summary.csv")

print("\nOutputs saved to outputs/"); print("Done.")
