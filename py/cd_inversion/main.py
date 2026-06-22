"""CD inversion main runner."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

from training_data import generate_cd_dataset
from inversion_network import CDInvNet
from cd_forward_model import predict_spectrum
from analysis import plot_forward_model, plot_training_curve, save_summary

def eval_net(net, specs, comps):
    preds = np.array([net.predict(specs[i]) for i in range(len(specs))])
    rmse = float(np.sqrt(np.mean((preds - comps)**2)))
    r2s = []
    for i in range(len(specs)):
        rec = predict_spectrum(preds[i,0], preds[i,1], preds[i,2])
        ss_res = np.sum((specs[i]-rec)**2); ss_tot = np.sum((specs[i]-specs[i].mean())**2)
        r2s.append(max(0, 1 - ss_res/(ss_tot+1e-6)))
    return {"rmse": rmse, "r2": float(np.mean(r2s))}


def main():
    print("CD Inversion Network"); print("=" * 60)

    wl, specs, comps = generate_cd_dataset(1000, noise_sigma=0.5, seed=42)
    n_train = 800
    train_s, train_c = specs[:n_train], comps[:n_train]
    test_s,  test_c  = specs[n_train:], comps[n_train:]

    print("Training with physics loss...")
    net_w = CDInvNet(seed=42)
    losses_w = net_w.train(train_s, train_c, epochs=30, lr=1e-3, lambda_phys=0.3)
    print(f"  Final loss: {losses_w[-1]:.4f}")

    print("Training without physics loss...")
    net_wo = CDInvNet(seed=42)
    losses_wo = net_wo.train(train_s, train_c, epochs=30, lr=1e-3, lambda_phys=0.0)

    res_w = eval_net(net_w, test_s, test_c)
    res_wo = eval_net(net_wo, test_s, test_c)
    print(f"\nWith physics:    RMSE={res_w['rmse']:.4f}  spectral_R²={res_w['r2']:.4f}")
    print(f"Without physics: RMSE={res_wo['rmse']:.4f}  spectral_R²={res_wo['r2']:.4f}")

    plot_forward_model("outputs/reference_spectra.png")
    plot_training_curve(losses_w, losses_wo, "outputs/training_curves.png")
    save_summary(res_w, res_wo, "outputs/cd_summary.csv")
    print("\nOutputs saved."); print("Done.")


if __name__ == "__main__":
    main()
