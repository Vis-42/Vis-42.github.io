"""Analysis and plotting for differentiable pendulum."""
import os
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")


def plot_inference_convergence(result, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0c0c0e")
    labels = ["m1", "m2", "L1", "L2"]; colors = ["#22c55e","#a78bfa","#f97316","#24a3ff"]
    for ax in axes:
        ax.set_facecolor("#111113")
        for sp in ax.spines.values(): sp.set_color("#222226")
        ax.tick_params(colors="#71717a")
    axes[0].semilogy(result.loss_history, color="#24a3ff", linewidth=2)
    axes[0].set_title("Loss vs Gradient Step", color="#a1a1aa"); axes[0].set_xlabel("Step", color="#71717a")
    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        axes[1].plot(result.param_history[:, i], color=clr, linewidth=1.5, label=lbl)
    axes[1].set_title("Parameter Convergence", color="#a1a1aa"); axes[1].legend(facecolor="#111113", labelcolor="#a1a1aa")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def plot_trajectory_comparison(true_traj, inf_traj, observed, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0c0c0e")
    labels = ["θ₁","ω₁","θ₂","ω₂"]
    for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
        ax.set_facecolor("#111113")
        for sp in ax.spines.values(): sp.set_color("#222226")
        ax.tick_params(colors="#71717a")
        ax.plot(true_traj[:, i], color="#24a3ff", linewidth=2, label="True", linestyle="--")
        ax.plot(inf_traj[:, i],  color="#22c55e", linewidth=2, label="Inferred")
        ax.scatter(range(len(observed)), observed[:, i], s=4, color="#f97316", alpha=0.5, label="Observed")
        ax.set_title(lbl, color="#a1a1aa"); ax.legend(facecolor="#111113", labelcolor="#a1a1aa", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0c0c0e"); plt.close()


def save_summary(results, noise_levels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("noise_sigma,m1_err%,m2_err%,L1_err%,L2_err%,final_loss\n")
        for σ, res in zip(noise_levels, results):
            e = res.param_error_pct
            f.write(f"{σ},{e[0]:.2f},{e[1]:.2f},{e[2]:.2f},{e[3]:.2f},{res.loss_history[-1]:.6g}\n")
