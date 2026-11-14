"""Radius of gyration calculation."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def _gyration_bonds(mol: Molecule, reuse=None) -> list:
    """The molecule's bonds: its own list, else ``reuse`` (an earlier frame's) when every
    one of those is still a bond here, else detected now."""
    if mol.bonds:
        return mol.bonds
    if reuse:
        pairs = np.asarray(reuse, dtype=int)
        if pairs.max() < len(mol.atoms):
            xyz = np.array([a.x for a in mol.atoms])
            radii = np.array([a.bond_r for a in mol.atoms], dtype=float)
            d = xyz[pairs[:, 1]] - xyz[pairs[:, 0]]
            d -= np.round(d @ mol.pbc.reciprocal_basis_matrix) @ mol.pbc.basis_matrix
            if np.all(np.linalg.norm(d, axis=1) < 0.5 * (radii[pairs[:, 0]] + radii[pairs[:, 1]])):
                return reuse
    probe = Molecule(atoms=mol.atoms, pbc=mol.pbc)
    probe.detect_bonds()
    return probe.bonds


def _gyration_whole_positions(mol: Molecule, bonds: list):
    """Positions with molecules that a cell face has split put back together, or None
    when no bond crosses a face (nothing to do).

    Each bonded group is rebuilt from its first atom along the bonds (minimum image).
    A group bonded to its own periodic image, such as a framework, cannot be made whole
    and keeps its coordinates. The molecule itself is not changed.
    """
    if not bonds:
        return None
    pbc = mol.pbc
    xyz = np.array([a.x for a in mol.atoms])
    pairs = np.asarray(bonds, dtype=int)
    if pairs.max() >= len(xyz):
        return None
    raw = xyz[pairs[:, 1]] - xyz[pairs[:, 0]]
    image = np.round(raw @ pbc.reciprocal_basis_matrix)
    if not image.any():
        return None
    step = raw - image @ pbc.basis_matrix  # bond vectors, nearest image
    adj: dict = {}
    for k, (i, j) in enumerate(pairs.tolist()):
        adj.setdefault(i, []).append((j, k, 1.0))
        adj.setdefault(j, []).append((i, k, -1.0))
    out = xyz.copy()
    seen: set = set()
    for k0 in np.nonzero(image.any(axis=1))[0]:
        start = int(pairs[k0, 0])
        if start in seen:
            continue
        group = [start]
        seen.add(start)
        placed = {start: xyz[start]}
        queue = [start]
        while queue:
            i = queue.pop()
            for j, k, sign in adj[i]:
                if j not in placed:
                    placed[j] = placed[i] + sign * step[k]
                    seen.add(j)
                    group.append(j)
                    queue.append(j)
        members = set(group)
        closed = all(np.allclose(placed[int(pairs[k, 1])] - placed[int(pairs[k, 0])], step[k],
                                 atol=1e-6)
                     for i in group for _j, k, sign in adj[i] if sign > 0 and _j in members)
        if closed:  # a finite molecule; otherwise it is bonded to its own image
            first = min(group)  # keep the lowest-numbered atom where it is
            offset = xyz[first] - placed[first]
            for i in group:
                out[i] = placed[i] + offset
    return out


def compute_gyration_radius(
    mol: Molecule,
    element: str | None = None,
    name: str | None = None,
    bonds: list | None = None,
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
    if mol.pbc is not None:
        whole = _gyration_whole_positions(mol, _gyration_bonds(mol, bonds))
        if whole is not None:
            index = {id(a): i for i, a in enumerate(mol.atoms)}
            positions = whole[[index[id(a)] for a in atoms]]
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

    bonds = None
    for fi, mol in enumerate(frames):
        if mol.pbc is not None:
            bonds = _gyration_bonds(mol, bonds)
        rg[fi] = compute_gyration_radius(mol, element, name, bonds)
        if progress_callback and fi % max(1, n // 20) == 0:
            progress_callback(fi / n)

    if progress_callback:
        progress_callback(1.0)

    return frame_idx, rg
