"""Off-lattice Gō model for protein folding."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

Array = np.ndarray


@dataclass
class ProteinConfig:
    n_residues: int
    native_coords: Array
    native_contacts: List[Tuple[int, int]]
    epsilon: float = 1.0
    sigma: float = 0.4
    k_bond: float = 100.0
    r0: float = 0.38
    contact_cutoff: float = 0.8


def TrpCageConfig(epsilon: float = 1.0, seed: int = 42) -> ProteinConfig:
    n = 25
    coords = np.zeros((n, 3))
    for i in range(n):
        angle = 2 * np.pi * i / 3.6
        coords[i, 0] = 0.23 * np.cos(angle)
        coords[i, 1] = 0.23 * np.sin(angle)
        coords[i, 2] = 0.15 * i
    for i in range(14, n):
        coords[i, 0] += 0.08 * np.sin(i)
        coords[i, 1] += 0.08 * np.cos(i)
        coords[i, 2] *= 0.75

    cutoff = 0.8
    contacts = []
    for i in range(n):
        for j in range(i + 3, n):
            if np.linalg.norm(coords[i] - coords[j]) < cutoff:
                contacts.append((i, j))

    return ProteinConfig(n, coords.copy(), contacts, epsilon=epsilon)


def go_energy(coords: Array, cfg: ProteinConfig) -> float:
    return bond_energy(coords, cfg) + contact_energy(coords, cfg) + repulsion_energy(coords, cfg)


def bond_energy(coords: Array, cfg: ProteinConfig) -> float:
    diffs = coords[1:] - coords[:-1]
    r = np.linalg.norm(diffs, axis=1)
    return 0.5 * cfg.k_bond * float(np.sum((r - cfg.r0) ** 2))


def contact_energy(coords: Array, cfg: ProteinConfig) -> float:
    E = 0.0
    for i, j in cfg.native_contacts:
        r_ij  = np.linalg.norm(coords[i] - coords[j])
        r_nat = max(np.linalg.norm(cfg.native_coords[i] - cfg.native_coords[j]), 0.2)
        x = r_nat / max(r_ij, 1e-6)
        E += cfg.epsilon * (5 * x**12 - 6 * x**10)
    return E


def repulsion_energy(coords: Array, cfg: ProteinConfig) -> float:
    contact_set = set(cfg.native_contacts)
    E = 0.0
    for i in range(cfg.n_residues):
        for j in range(i + 3, cfg.n_residues):
            if (i, j) in contact_set:
                continue
            r = max(np.linalg.norm(coords[i] - coords[j]), 0.1)
            E += cfg.epsilon * (cfg.sigma / r) ** 12
    return E


def radius_of_gyration(coords: Array) -> float:
    com = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - com)**2, axis=1))))


def fraction_native_contacts(coords: Array, cfg: ProteinConfig) -> float:
    if not cfg.native_contacts:
        return 0.0
    n_formed = sum(
        1 for i, j in cfg.native_contacts
        if np.linalg.norm(coords[i] - coords[j]) <
           1.2 * max(np.linalg.norm(cfg.native_coords[i] - cfg.native_coords[j]), 0.2)
    )
    return n_formed / len(cfg.native_contacts)
