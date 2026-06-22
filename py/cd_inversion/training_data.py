"""Generate synthetic CD spectroscopy dataset."""
import numpy as np
from cd_forward_model import WAVELENGTHS, predict_spectrum


def generate_cd_dataset(N: int, noise_sigma: float = 0.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    spectra = np.zeros((N, len(WAVELENGTHS)))
    compositions = np.zeros((N, 3))
    for i in range(N):
        x1, x2 = np.sort(rng.random(2))
        h, s, c = x1, x2 - x1, 1 - x2
        compositions[i] = [h, s, c]
        clean = predict_spectrum(h, s, c)
        spectra[i] = clean + noise_sigma * rng.standard_normal(len(WAVELENGTHS))
    return WAVELENGTHS, spectra, compositions
