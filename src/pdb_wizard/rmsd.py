"""Root Mean Square Deviation (RMSD) calculation over trajectories."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def compute_rmsd(
    frames: list[Molecule],
    element: str | None = None,
    name: str | None = None,
    reference_frame: int = 0,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMSD vs frame number relative to a reference frame.

    RMSD(t) = sqrt( (1/N) * sum_i |r_i(t) - r_i(ref)|^2 )

    Args:
        frames: List of Molecule objects (trajectory).
        element: Filter to this element (None = all atoms).
        name: Filter to this atom name (None = all matching element).
        reference_frame: Frame index to use as reference (default 0).
        progress_callback: Optional callable(frac).

    Returns:
        (frame_indices, rmsd_values) arrays.
    """
    if len(frames) < 2:
        return np.array([]), np.array([])

    ref = frames[reference_frame]

    if element:
        if name and name != "(all)":
            def atom_filter(a):
                return a.element.symbol == element and a.name.strip() == name
        else:
            def atom_filter(a):
                return a.element.symbol == element
    else:
        def atom_filter(a):
            return True

    ref_indices = [i for i, a in enumerate(ref.atoms) if atom_filter(a)]
    if not ref_indices:
        return np.array([]), np.array([])

    ref_pos = np.array([ref.atoms[i].x for i in ref_indices])
    n_atoms = len(ref_indices)
    n_frames = len(frames)

    frame_idx = np.arange(n_frames, dtype=float)
    rmsd_vals = np.zeros(n_frames)

    for fi, mol in enumerate(frames):
        indices = [i for i, a in enumerate(mol.atoms) if atom_filter(a)]
        n_avail = min(len(indices), n_atoms)
        if n_avail == 0:
            continue
        pos = np.array([mol.atoms[indices[i]].x for i in range(n_avail)])
        diff = pos[:n_avail] - ref_pos[:n_avail]
        rmsd_vals[fi] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        if progress_callback and fi % max(1, n_frames // 20) == 0:
            progress_callback(fi / n_frames)

    if progress_callback:
        progress_callback(1.0)

    return frame_idx, rmsd_vals
