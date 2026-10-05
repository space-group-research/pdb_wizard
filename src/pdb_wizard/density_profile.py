"""1D density profile along a cell axis."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def compute_density_profile(
    mol: Molecule,
    axis: int = 2,
    n_bins: int = 100,
    element: str | None = None,
    name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D number density profile along a cell axis.

    Args:
        mol: Molecule with PBC.
        axis: 0=a, 1=b, 2=c.
        n_bins: Number of bins along the axis.
        element: Filter to this element (None = all).
        name: Filter to this atom name (None = all).

    Returns:
        (positions_along_axis, density) arrays.
    """
    if mol.pbc is None:
        return np.array([]), np.array([])

    pbc = mol.pbc
    atoms = mol.atoms
    if element:
        if name and name != "(all)":
            atoms = [a for a in atoms if a.element.symbol == element and a.name.strip() == name]
        else:
            atoms = [a for a in atoms if a.element.symbol == element]

    if not atoms:
        return np.array([]), np.array([])

    coords = np.array([a.x for a in atoms])
    # Project onto fractional coordinates along the chosen axis
    frac = coords @ pbc.reciprocal_basis_matrix
    frac_along = frac[:, axis] % 1.0

    # Histogram
    cell_length = [pbc.a, pbc.b, pbc.c][axis]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(frac_along, bins=bin_edges)

    # Convert to number density (atoms per Angstrom)
    bin_width = cell_length / n_bins
    density = hist / bin_width

    positions = (bin_edges[:-1] + bin_edges[1:]) / 2 * cell_length

    return positions, density


def compute_density_profile_trajectory(
    frames: list[Molecule],
    axis: int = 2,
    n_bins: int = 100,
    element: str | None = None,
    name: str | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average density profile over trajectory frames."""
    d_sum = None
    positions = None
    for fi, mol in enumerate(frames):
        p, d = compute_density_profile(mol, axis, n_bins, element, name)
        if len(p) == 0:
            continue
        if d_sum is None:
            d_sum = d.copy()
            positions = p
        else:
            d_sum += d
        if progress_callback:
            progress_callback((fi + 1) / len(frames))
    if d_sum is None:
        return np.array([]), np.array([])
    return positions, d_sum / len(frames)
