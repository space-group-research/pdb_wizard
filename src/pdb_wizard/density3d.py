"""3D spatial density histogram for sorbate position analysis."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC


def compute_density_3d(
    frames: list[Molecule],
    element: str,
    n_bins: int = 20,
    name: str | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, PBC]:
    """Compute 3D spatial density histogram for an element over trajectory frames.

    Bins atom positions into a 3D grid in fractional coordinates, then
    converts bin centers back to Cartesian for rendering.

    Args:
        frames: Trajectory frames.
        element: Element symbol to histogram.
        n_bins: Number of bins per axis.
        name: Optional atom name filter.
        progress_callback: Optional callable(frac).

    Returns:
        (positions, densities, pbc):
        positions is (N, 3) Cartesian coords of non-zero bins,
        densities is (N,) normalized density values (0-1),
        pbc is the PBC from the first frame.
    """
    if not frames or frames[0].pbc is None:
        return np.zeros((0, 3)), np.zeros(0), None

    pbc = frames[0].pbc
    histogram = np.zeros((n_bins, n_bins, n_bins))

    for fi, mol in enumerate(frames):
        if mol.pbc is None:
            continue
        for atom in mol.atoms:
            if atom.element.symbol != element:
                continue
            if name and name != "(all)" and atom.name.strip() != name:
                continue
            # Convert to fractional coordinates
            frac = atom.x @ mol.pbc.reciprocal_basis_matrix
            frac -= np.floor(frac)  # wrap to [0, 1)
            ix = min(int(frac[0] * n_bins), n_bins - 1)
            iy = min(int(frac[1] * n_bins), n_bins - 1)
            iz = min(int(frac[2] * n_bins), n_bins - 1)
            histogram[ix, iy, iz] += 1

        if progress_callback and fi % max(1, len(frames) // 20) == 0:
            progress_callback(fi / len(frames))

    if progress_callback:
        progress_callback(1.0)

    # Extract non-zero bins
    nonzero = histogram > 0
    if not nonzero.any():
        return np.zeros((0, 3)), np.zeros(0), pbc

    indices = np.argwhere(nonzero)  # (N, 3) array of bin indices
    densities = histogram[nonzero]

    # Normalize to 0-1
    max_dens = densities.max()
    if max_dens > 0:
        densities = densities / max_dens

    # Convert bin centers to Cartesian
    frac_centers = (indices + 0.5) / n_bins  # fractional coordinates
    positions = frac_centers @ pbc.basis_matrix

    return positions, densities, pbc
