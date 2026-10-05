"""Radius of gyration calculation."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def compute_gyration_radius(
    mol: Molecule,
    element: str | None = None,
    name: str | None = None,
) -> float:
    """Compute radius of gyration for selected atoms.

    Rg = sqrt( (1/N) * sum_i |r_i - r_com|^2 )
    """
    if element:
        if name and name != "(all)":
            atoms = [a for a in mol.atoms if a.element.symbol == element and a.name.strip() == name]
        else:
            atoms = [a for a in mol.atoms if a.element.symbol == element]
    else:
        atoms = list(mol.atoms)

    if len(atoms) < 2:
        return 0.0

    positions = np.array([a.x for a in atoms])
    com = positions.mean(axis=0)
    sq_dist = np.sum((positions - com) ** 2, axis=1)
    return float(np.sqrt(sq_dist.mean()))


def compute_gyration_trajectory(
    frames: list[Molecule],
    element: str | None = None,
    name: str | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radius of gyration vs frame number.

    Returns:
        (frame_indices, rg_values) arrays.
    """
    if len(frames) < 1:
        return np.array([]), np.array([])

    n = len(frames)
    frame_idx = np.arange(n, dtype=float)
    rg = np.zeros(n)

    for fi, mol in enumerate(frames):
        rg[fi] = compute_gyration_radius(mol, element, name)
        if progress_callback and fi % max(1, n // 20) == 0:
            progress_callback(fi / n)

    if progress_callback:
        progress_callback(1.0)

    return frame_idx, rg
