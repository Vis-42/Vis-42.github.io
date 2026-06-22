"""CD spectroscopy forward model."""
import numpy as np

WAVELENGTHS = np.arange(190.0, 251.0, 1.0)  # 61 points
N_WL = len(WAVELENGTHS)

def _build_basis():
    wl = WAVELENGTHS
    helix = (-30*np.exp(-0.5*((wl-208)/5)**2) - 25*np.exp(-0.5*((wl-222)/4)**2)
             + 50*np.exp(-0.5*((wl-193)/8)**2))
    sheet = (-15*np.exp(-0.5*((wl-216)/7)**2) + 20*np.exp(-0.5*((wl-195)/6)**2))
    coil  = (-8*np.exp(-0.5*((wl-200)/6)**2) + 5*np.exp(-0.5*((wl-218)/8)**2))
    return helix, sheet, coil

HELIX_BASIS, SHEET_BASIS, COIL_BASIS = _build_basis()
REFERENCE_SPECTRA = {"helix": HELIX_BASIS, "sheet": SHEET_BASIS, "coil": COIL_BASIS}


def predict_spectrum(h: float, s: float, c: float) -> np.ndarray:
    tot = h + s + c
    if tot < 1e-6:
        h = s = c = 1/3; tot = 1.0
    return (h/tot)*HELIX_BASIS + (s/tot)*SHEET_BASIS + (c/tot)*COIL_BASIS


def add_noise(spectrum: np.ndarray, sigma: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return spectrum + sigma * rng.standard_normal(len(spectrum))
