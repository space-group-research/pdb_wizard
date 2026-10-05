"""Radial distribution function (RDF) calculation with PBC support."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def compute_rdf(
    mol: Molecule,
    type1: str,
    type2: str,
    n_bins: int = 200,
    r_max: float | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RDF g(r) between two atom types for a single frame.

    Args:
        mol: Molecule with PBC.
        type1: Element symbol for first atom type.
        type2: Element symbol for second atom type.
        n_bins: Number of histogram bins.
        r_max: Maximum distance (default: half the shortest cell dimension).
        progress_callback: Optional callable(frac: float).

    Returns:
        (r_values, g_r) arrays.
    """
    if mol.pbc is None:
        return np.array([]), np.array([])

    pbc = mol.pbc
    if r_max is None:
        r_max = min(pbc.a, pbc.b, pbc.c) / 2.0

    idx1 = [i for i, a in enumerate(mol.atoms) if a.element.symbol == type1]
    idx2 = [i for i, a in enumerate(mol.atoms) if a.element.symbol == type2]

    if not idx1 or not idx2:
        return np.array([]), np.array([])

    coords = np.array([a.x for a in mol.atoms])
    dr = r_max / n_bins
    hist = np.zeros(n_bins)

    same_type = type1 == type2
    total = len(idx1)

    for ci, i in enumerate(idx1):
        targets = idx2 if not same_type else [j for j in idx2 if j > i]
        if not targets:
            continue
        dx = coords[targets] - coords[i]
        frac = dx @ pbc.reciprocal_basis_matrix
        wrapped = dx - np.round(frac) @ pbc.basis_matrix
        dists = np.linalg.norm(wrapped, axis=1)
        bins = (dists / dr).astype(int)
        valid = (bins >= 0) & (bins < n_bins) & (dists < r_max)
        for b in bins[valid]:
            hist[b] += 1

        if progress_callback and ci % max(1, total // 20) == 0:
            progress_callback(ci / total)

    if progress_callback:
        progress_callback(1.0)

    # Normalize
    r_values = (np.arange(n_bins) + 0.5) * dr
    n1 = len(idx1)
    n2 = len(idx2)
    volume = pbc.volume

    shell_vol = 4.0 * np.pi * r_values**2 * dr
    if same_type:
        # Counted each pair once (j > i). Normalize by N*(N-1)/2 pairs
        # and ideal gas density.
        rho = n1 / volume
        ideal = rho * shell_vol
        g_r = np.zeros(n_bins)
        nonzero = ideal > 0
        # Factor 2: we counted j>i only, but ideal counts full shell
        g_r[nonzero] = 2.0 * hist[nonzero] / (n1 * ideal[nonzero])
    else:
        rho = n2 / volume
        ideal = rho * shell_vol
        g_r = np.zeros(n_bins)
        nonzero = ideal > 0
        g_r[nonzero] = hist[nonzero] / (n1 * ideal[nonzero])

    return r_values, g_r


def compute_rdf_trajectory(
    frames: list[Molecule],
    type1: str,
    type2: str,
    n_bins: int = 200,
    r_max: float | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RDF averaged over multiple trajectory frames."""
    if not frames:
        return np.array([]), np.array([])

    g_sum = None
    r_values = None
    n_frames = len(frames)

    for fi, mol in enumerate(frames):
        r, g = compute_rdf(mol, type1, type2, n_bins=n_bins, r_max=r_max)
        if len(r) == 0:
            continue
        if g_sum is None:
            g_sum = g.copy()
            r_values = r
        else:
            g_sum += g
        if progress_callback:
            progress_callback((fi + 1) / n_frames)

    if g_sum is None:
        return np.array([]), np.array([])

    return r_values, g_sum / n_frames
