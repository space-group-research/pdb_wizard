"""Radial distribution function (RDF) calculation with PBC support."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def _select(mol: Molecule, element: str, name: str | None) -> np.ndarray:
    return np.array([i for i, a in enumerate(mol.atoms)
                     if a.element.symbol == element
                     and (name is None or a.name.strip() == name)], dtype=int)


def half_cell_width(pbc) -> float:
    """Half the smallest distance between opposite cell faces: the largest distance for
    which the minimum image finds every pair. min(a, b, c) / 2 for an orthogonal cell."""
    half = float(pbc.perpendicular_widths().min()) / 2.0
    edge = min(pbc.a, pbc.b, pbc.c) / 2.0
    return edge if abs(half - edge) <= 1e-9 * edge else half


def image_shifts(pbc, r_max: float) -> np.ndarray:
    """Lattice translations (in cell units, integer rows) needed besides the minimum image
    to reach every pair within ``r_max``; the zero shift is not included. Empty when
    ``r_max`` is within half the smallest cell width."""
    if r_max <= half_cell_width(pbc) * (1.0 + 1e-12):
        return np.zeros((0, 3))
    reach = np.ceil(r_max / pbc.perpendicular_widths() + 0.5).astype(int)
    grid = np.mgrid[-reach[0]:reach[0] + 1, -reach[1]:reach[1] + 1,
                    -reach[2]:reach[2] + 1].reshape(3, -1).T.astype(float)
    return grid[np.any(grid != 0, axis=1)]


def compute_rdf(
    mol: Molecule,
    type1: str,
    type2: str,
    n_bins: int = 200,
    r_max: float | None = None,
    progress_callback=None,
    name1: str | None = None,
    name2: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RDF g(r) between two atom types for a single frame.

    Args:
        mol: Molecule with PBC.
        type1: Element symbol for first atom type.
        type2: Element symbol for second atom type.
        n_bins: Number of histogram bins.
        r_max: Maximum distance (default: half the smallest cell width). Beyond that
            the pairs are summed over periodic images.
        progress_callback: Optional callable(frac: float).
        name1, name2: Optional atom names (e.g. "OW") narrowing each type.

    Returns:
        (r_values, g_r) arrays. g(r) tends to 1 at long range for an ideal gas, both for
        a type with itself (each pair counted once, hence the factor 2) and for two types.
    """
    if mol.pbc is None:
        return np.array([]), np.array([])

    pbc = mol.pbc
    if r_max is None:
        r_max = half_cell_width(pbc)

    idx1 = _select(mol, type1, name1)
    idx2 = _select(mol, type2, name2)

    if not len(idx1) or not len(idx2):
        return np.array([]), np.array([])

    coords = np.array([a.x for a in mol.atoms])
    dr = r_max / n_bins
    hist = np.zeros(n_bins)

    same_type = len(idx1) == len(idx2) and bool(np.array_equal(idx1, idx2))
    total = len(idx1)
    targets = coords[idx2]
    # rows of the i x j distance matrix per block, keeping a block near 2M pairs
    block = max(1, 2_000_000 // len(idx2))
    # beyond half the cell width one image per pair is not enough
    shifts = image_shifts(pbc, r_max) @ pbc.basis_matrix

    for b0 in range(0, total, block):
        rows = np.arange(b0, min(b0 + block, total))
        dx = targets[None, :, :] - coords[idx1[rows]][:, None, :]
        frac = dx @ pbc.reciprocal_basis_matrix
        wrapped = dx - np.round(frac) @ pbc.basis_matrix
        dists = np.linalg.norm(wrapped, axis=2)
        if same_type:  # each pair once: j after i
            keep = np.arange(len(idx2))[None, :] > rows[:, None]
        else:  # overlapping selections (e.g. all O vs OW): never an atom with itself
            keep = idx2[None, :] != idx1[rows][:, None]
        d = dists[keep]
        bins = (d / dr).astype(int)
        valid = (bins >= 0) & (bins < n_bins) & (d < r_max)
        hist += np.bincount(bins[valid], minlength=n_bins)[:n_bins]
        for shift in shifts:
            # further images of every atom, its own included. For one type with itself
            # (i, j, +L) and (j, i, -L) are the same pair, so each counts a half.
            d = np.linalg.norm(wrapped + shift, axis=2).ravel()
            d = d[d < r_max]
            hist += np.bincount((d / dr).astype(int), minlength=n_bins)[:n_bins] * (
                0.5 if same_type else 1.0)

        if progress_callback:
            progress_callback(rows[-1] / total)

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
    name1: str | None = None,
    name2: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RDF averaged over multiple trajectory frames."""
    if not frames:
        return np.array([]), np.array([])

    g_sum = None
    r_values = None
    n_frames = len(frames)

    for fi, mol in enumerate(frames):
        def _frame_progress(frac, _fi=fi):
            if progress_callback:
                progress_callback((_fi + frac) / n_frames)

        r, g = compute_rdf(mol, type1, type2, n_bins=n_bins, r_max=r_max,
                           progress_callback=_frame_progress, name1=name1, name2=name2)
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
