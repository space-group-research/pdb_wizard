"""Coordination number analysis."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule
from pdb_wizard.rdf import image_shifts


def compute_coordination(
    mol: Molecule,
    center_element: str,
    neighbor_element: str,
    r_max: float = 5.0,
    n_bins: int = 200,
    center_name: str | None = None,
    neighbor_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute coordination number as a function of distance.

    Returns the running coordination number N(r) — the average number of
    neighbor atoms within distance r of each center atom.

    Args:
        mol: Molecule (with or without PBC).
        center_element: Element symbol for the central atoms.
        neighbor_element: Element symbol for the neighbor atoms.
        r_max: Maximum distance in Angstroms.
        n_bins: Number of distance bins.
        center_name: Optional atom name filter for center.
        neighbor_name: Optional atom name filter for neighbor.

    Returns:
        (r_values, coordination_number, avg_at_cutoff):
        r_values and running N(r), plus the average coordination at r_max.
    """
    center_idx = []
    for i, a in enumerate(mol.atoms):
        if a.element.symbol != center_element:
            continue
        if center_name and center_name != "(all)" and a.name.strip() != center_name:
            continue
        center_idx.append(i)

    neighbor_idx = []
    for i, a in enumerate(mol.atoms):
        if a.element.symbol != neighbor_element:
            continue
        if neighbor_name and neighbor_name != "(all)" and a.name.strip() != neighbor_name:
            continue
        neighbor_idx.append(i)

    if not center_idx or not neighbor_idx:
        return np.array([]), np.array([]), 0.0

    center_pos = np.array([mol.atoms[i].x for i in center_idx])
    neighbor_pos = np.array([mol.atoms[i].x for i in neighbor_idx])

    r_values = np.linspace(0, r_max, n_bins)
    dr = r_values[1] - r_values[0]
    histogram = np.zeros(n_bins)

    use_pbc = mol.pbc is not None
    # beyond half the cell width one image per pair is not enough
    shifts = (image_shifts(mol.pbc, r_max, len(center_idx) * len(neighbor_idx))
              @ mol.pbc.basis_matrix) if use_pbc else ()

    for ci in range(len(center_pos)):
        diffs = neighbor_pos - center_pos[ci]
        if use_pbc:
            # Minimum image convention
            frac = diffs @ mol.pbc.reciprocal_basis_matrix
            frac -= np.round(frac)
            diffs = frac @ mol.pbc.basis_matrix
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        if len(shifts):
            dists = np.concatenate([dists] + [np.sqrt(np.sum((diffs + shift) ** 2, axis=1))
                                              for shift in shifts])
        # Exclude self (distance ~0) and clamp to r_max
        mask = (dists > 0.1) & (dists < r_max)
        if not mask.any():
            continue
        # entry i of the running sum is the number closer than r_values[i]
        bins = np.minimum((dists[mask] / dr).astype(int) + 1, n_bins - 1)
        histogram += np.bincount(bins, minlength=n_bins)[:n_bins]

    # Average per center atom
    n_center = len(center_idx)
    histogram /= n_center

    # Running coordination number = cumulative sum
    coord_number = np.cumsum(histogram)

    avg_at_cutoff = float(coord_number[-1]) if len(coord_number) > 0 else 0.0

    return r_values, coord_number, avg_at_cutoff


def compute_coordination_trajectory(
    frames: list[Molecule],
    center_element: str,
    neighbor_element: str,
    r_max: float = 5.0,
    n_bins: int = 200,
    center_name: str | None = None,
    neighbor_name: str | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average coordination number over trajectory frames."""
    if not frames:
        return np.array([]), np.array([]), 0.0

    r_values = np.linspace(0, r_max, n_bins)
    coord_sum = np.zeros(n_bins)

    for fi, mol in enumerate(frames):
        _, cn, _ = compute_coordination(
            mol, center_element, neighbor_element, r_max, n_bins,
            center_name, neighbor_name,
        )
        if len(cn) == n_bins:
            coord_sum += cn
        if progress_callback and fi % max(1, len(frames) // 20) == 0:
            progress_callback(fi / len(frames))

    coord_avg = coord_sum / len(frames)
    avg_at_cutoff = float(coord_avg[-1]) if len(coord_avg) > 0 else 0.0

    if progress_callback:
        progress_callback(1.0)

    return r_values, coord_avg, avg_at_cutoff
