"""Hydrogen bond detection and analysis."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def detect_hbonds(
    mol: Molecule,
    d_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
) -> list[tuple[int, int, int, float, float]]:
    """Detect hydrogen bonds: D-H...A where D and A are electronegative.

    Args:
        mol: Molecule with bonds detected.
        d_cutoff: Max D-A distance in Angstroms.
        angle_cutoff: Min D-H-A angle in degrees.

    Returns:
        List of (donor_idx, h_idx, acceptor_idx, distance, angle).
    """
    _donors_acceptors = {"N", "O", "F", "S", "Cl"}

    # Build neighbor map
    neighbors: dict[int, list[int]] = {}
    for a, b in mol.bonds:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    # Find H atoms bonded to donor
    h_donors: list[tuple[int, int]] = []  # (h_idx, donor_idx)
    for i, atom in enumerate(mol.atoms):
        if atom.element.symbol != "H":
            continue
        for nb in neighbors.get(i, []):
            if mol.atoms[nb].element.symbol in _donors_acceptors:
                h_donors.append((i, nb))
                break

    # Find acceptor atoms
    acceptor_indices = [i for i, a in enumerate(mol.atoms) if a.element.symbol in _donors_acceptors]

    if not h_donors or not acceptor_indices:
        return []

    use_pbc = mol.pbc is not None
    hbonds = []

    for h_idx, d_idx in h_donors:
        h_pos = mol.atoms[h_idx].x
        d_pos = mol.atoms[d_idx].x
        for a_idx in acceptor_indices:
            if a_idx == d_idx or a_idx == h_idx:
                continue
            a_pos = mol.atoms[a_idx].x
            diff_da = a_pos - d_pos
            if use_pbc:
                frac = diff_da @ mol.pbc.reciprocal_basis_matrix
                frac -= np.round(frac)
                diff_da = frac @ mol.pbc.basis_matrix
            dist = float(np.linalg.norm(diff_da))
            if dist > d_cutoff:
                continue
            # Angle D-H...A
            dh = d_pos - h_pos
            ha = a_pos - h_pos
            if use_pbc:
                frac = ha @ mol.pbc.reciprocal_basis_matrix
                frac -= np.round(frac)
                ha = frac @ mol.pbc.basis_matrix
            cos_a = np.dot(dh, ha) / (np.linalg.norm(dh) * np.linalg.norm(ha) + 1e-10)
            angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if angle >= angle_cutoff:
                hbonds.append((d_idx, h_idx, a_idx, dist, angle))

    return hbonds


def count_hbonds_trajectory(
    frames: list[Molecule],
    d_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count hydrogen bonds per frame over a trajectory.

    Returns:
        (frame_indices, hbond_counts) arrays.
    """
    n = len(frames)
    frame_idx = np.arange(n, dtype=float)
    counts = np.zeros(n)

    for fi, mol in enumerate(frames):
        if not mol.bonds:
            if fi > 0 and frames[0].bonds and len(frames[0].atoms) == len(mol.atoms):
                mol.bonds = frames[0].bonds
            else:
                mol.detect_bonds()
        hb = detect_hbonds(mol, d_cutoff, angle_cutoff)
        counts[fi] = len(hb)
        if progress_callback and fi % max(1, n // 20) == 0:
            progress_callback(fi / n)

    if progress_callback:
        progress_callback(1.0)

    return frame_idx, counts
