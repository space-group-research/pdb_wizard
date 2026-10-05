"""Geometry analysis functions (bonds, angles, contacts, overlaps)."""

from __future__ import annotations

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids

# Two atoms closer than this (Å, minimum image) are treated as duplicates.
OVERLAP_CUTOFF = 0.05
# Upper bound on (rows x atoms) distance entries evaluated per chunk.
_OVERLAP_CHUNK_BUDGET = 4_000_000


def find_overlapping_pairs(
    mol: Molecule, cutoff: float = OVERLAP_CUTOFF,
) -> list[tuple[int, int]]:
    """Return index pairs (i, j), i < j, of atoms within ``cutoff`` of each other.

    Uses the minimum-image convention, so requires ``mol.pbc``. Pairs are
    sorted by (i, j). Distances are evaluated in row chunks so the memory
    footprint stays bounded for large frameworks.
    """
    pbc = mol.pbc
    n = len(mol.atoms)
    if pbc is None or n < 2:
        return []
    coords = np.array([a.x for a in mol.atoms], dtype=float)
    recip = pbc.reciprocal_basis_matrix
    basis = pbc.basis_matrix
    cut2 = cutoff * cutoff
    # Keep each chunk's (chunk x n x 3) intermediates to a few million floats.
    chunk = max(1, _OVERLAP_CHUNK_BUDGET // n)
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = coords[start:stop, None, :] - coords[None, :, :]
        frac = dx @ recip
        wrapped = dx - np.round(frac) @ basis
        r2 = np.einsum("ijk,ijk->ij", wrapped, wrapped)
        ii, jj = np.nonzero(r2 < cut2)
        ii = ii + start
        keep = ii < jj
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


def remove_overlapping_atoms(
    mol: Molecule, cutoff: float = OVERLAP_CUTOFF,
) -> tuple[Molecule, list[str]]:
    """Delete duplicate atoms that sit on top of each other.

    For each cluster of overlapping atoms the first one in file order is kept
    and the later ones are removed, matching the behaviour of the original
    pdb_wizard. Returns the molecule and one human-readable message per
    deleted atom. Nothing is printed here; callers decide how to report.
    """
    if mol.pbc is None:
        return mol, []
    system = mol.atoms
    set_atom_ids(system)
    pairs = find_overlapping_pairs(mol, cutoff)
    if not pairs:
        return mol, []

    removed: set[int] = set()
    messages: list[str] = []
    for i, j in pairs:
        if i in removed or j in removed:
            continue
        removed.add(j)
        a, b = system[i], system[j]
        messages.append(
            f"Deleting overlapping atoms\n"
            f"{a.element.symbol:>3} {a.id:>5} --- {b.element.symbol:>3} {b.id:>5}\n"
            f" {a.x}\n {b.x}"
        )

    mol.atoms = [atom for k, atom in enumerate(system) if k not in removed]
    set_atom_ids(mol.atoms)
    return mol, messages


def overlap_detector(mol: Molecule) -> Molecule:
    """Classic-menu entry point: remove overlapping atoms and print what was deleted."""
    mol, messages = remove_overlapping_atoms(mol)
    for message in messages:
        print(message)
    return mol


def get_close_contacts(mol: Molecule) -> list[str]:
    if mol.pbc is None:
        return []
    pbc = mol.pbc
    system = mol.atoms
    set_atom_ids(system)
    i_idx, j_idx, rs = pbc.get_all_rs_min_image(system)
    vdw_radii = np.array([a.vdw for a in system], dtype=float)
    bond_radii = np.array([a.bond_r for a in system], dtype=float)
    mixed_vdw = 0.5 * (vdw_radii[i_idx] + vdw_radii[j_idx])
    mixed_bond = 0.5 * (bond_radii[i_idx] + bond_radii[j_idx])
    mask = (mixed_vdw > rs) & (rs > mixed_bond)
    messages: list[str] = []
    for idx in np.where(mask)[0]:
        i, j, r = int(i_idx[idx]), int(j_idx[idx]), rs[idx]
        el_str = f"{system[i].element.symbol}-{system[j].element.symbol}"
        messages.append(f"{el_str:<5} {i+1:>5} {j+1:>5}   r = {np.round(r, 6)}")
    return messages


def get_bonds_list(mol: Molecule) -> list[str]:
    """Return formatted bond list. Uses cached mol.bonds if present, else
    runs detect_bonds — both vectorized."""
    if mol.pbc is None:
        return []
    if not mol.bonds:
        mol.detect_bonds()
    if not mol.bonds:
        return []
    pbc = mol.pbc
    system = mol.atoms
    set_atom_ids(system)

    bond_arr = np.asarray(mol.bonds, dtype=int)
    coords = np.array([a.x for a in system])
    dx = coords[bond_arr[:, 0]] - coords[bond_arr[:, 1]]
    # Min image vectorized
    frac = dx @ pbc.reciprocal_basis_matrix
    frac -= np.round(frac)
    dx_min = frac @ pbc.basis_matrix
    rs = np.linalg.norm(dx_min, axis=1)

    symbols = [a.element.symbol for a in system]
    ids = [a.id for a in system]
    return [
        f"{symbols[i]+'-'+symbols[j]:<5} {ids[i]:>5} {ids[j]:>5}   r = {round(float(r), 6)}"
        for (i, j), r in zip(bond_arr, rs)
    ]


def get_angles_list(mol: Molecule) -> list[str]:
    """Return formatted bond-angle list. Iterates only over bonded triples
    (atom_i — atom_center — atom_k where both edges are in mol.bonds), so
    the cost is O(degree² * N) instead of O(N³)."""
    if mol.pbc is None:
        return []
    if not mol.bonds:
        mol.detect_bonds()
    if not mol.bonds:
        return []
    pbc = mol.pbc
    system = mol.atoms
    set_atom_ids(system)

    # Build adjacency map: center_idx -> list of bonded neighbor indices
    adj: dict[int, list[int]] = {}
    for a, b in mol.bonds:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    coords = np.array([a.x for a in system])
    symbols = [a.element.symbol for a in system]
    ids = [a.id for a in system]
    recip = pbc.reciprocal_basis_matrix
    basis = pbc.basis_matrix

    def min_image(v):
        f = v @ recip
        f -= np.round(f)
        return f @ basis

    messages: list[str] = []
    # Iterate centers; for each center, all unordered pairs of bonded neighbors form an angle
    for c, neighbors in adj.items():
        if len(neighbors) < 2:
            continue
        nb_arr = np.array(neighbors)
        # vectors center -> each neighbor (min-imaged)
        dx_all = coords[nb_arr] - coords[c]
        dx_all = min_image(dx_all)
        norms = np.linalg.norm(dx_all, axis=1)
        for ai in range(len(neighbors)):
            for bi in range(ai + 1, len(neighbors)):
                v1, v2 = dx_all[ai], dx_all[bi]
                n1, n2 = norms[ai], norms[bi]
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_a)))
                i, k = neighbors[ai], neighbors[bi]
                el_str = f"{symbols[i]}-{symbols[c]}-{symbols[k]}"
                messages.append(
                    f"{el_str:<7} {ids[i]:>5} {ids[c]:>5} {ids[k]:>5}   "
                    f"angle = {round(angle, 2):>6} "
                    f"r1, r2 = {round(float(n1), 3):>6}, {round(float(n2), 3):>6}"
                )
    return messages


def get_lone_atoms(mol: Molecule) -> list[Atom]:
    if mol.pbc is None:
        return []
    pbc = mol.pbc
    system = mol.atoms
    set_atom_ids(system)
    n = len(system)
    i_idx, j_idx, rs = pbc.get_all_rs_min_image(system)
    vdw_radii = np.array([a.vdw for a in system], dtype=float)
    mixed_vdw = 0.5 * (vdw_radii[i_idx] + vdw_radii[j_idx])
    has_neighbor = np.zeros(n, dtype=bool)
    contact_mask = rs < mixed_vdw
    has_neighbor[i_idx[contact_mask]] = True
    has_neighbor[j_idx[contact_mask]] = True
    return [system[i] for i in range(n) if not has_neighbor[i]]


def remove_lone_atoms(mol: Molecule) -> tuple[Molecule, list[Atom]]:
    """Delete atoms with no vdW-contact neighbour; return (mol, deleted atoms).

    Requires a cell: without ``mol.pbc`` nothing is removed and the returned
    list is empty, so callers can distinguish "no lone atoms" from "no cell".
    """
    lone_atoms = get_lone_atoms(mol)
    if not lone_atoms:
        return mol, []
    lone = set(id(a) for a in lone_atoms)
    keep = [i for i, a in enumerate(mol.atoms) if id(a) not in lone]
    keep_set = set(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}
    mol.atoms = [mol.atoms[i] for i in keep]
    mol.bonds = [
        (old_to_new[a], old_to_new[b])
        for a, b in mol.bonds if a in keep_set and b in keep_set
    ]
    set_atom_ids(mol.atoms)
    return mol, lone_atoms


def delete_lone_atoms(mol: Molecule) -> Molecule:
    """Delete lone atoms, returning only the molecule (see remove_lone_atoms)."""
    mol, _ = remove_lone_atoms(mol)
    return mol


def adjust_h_distances(
    mol: Molecule, second_element: str, distance: float,
) -> tuple[Molecule, list[str]]:
    """Set every bonded ``second_element``-H distance to ``distance`` Å.

    Returns the molecule and one message per adjusted pair, in the format the
    original pdb_wizard printed (``"O-H       12    13"``). Requires a cell:
    with ``mol.pbc`` None nothing is changed and no messages are returned.
    """
    if mol.pbc is None:
        return mol, []
    pbc = mol.pbc
    system = mol.atoms
    set_atom_ids(system)
    messages: list[str] = []
    for atom in system:
        for atom2 in system:
            if atom2.id > atom.id:
                is_h_pair = (
                    (atom.element.symbol == "H" and atom2.element.symbol == second_element) or
                    (atom2.element.symbol == "H" and atom.element.symbol == second_element)
                )
                if not is_h_pair:
                    continue
                h_atom = atom if atom.element.symbol == "H" else atom2
                other = atom2 if atom.element.symbol == "H" else atom
                dx = h_atom.x - other.x
                r = pbc.min_image(dx)
                bond_r = 0.5 * (atom.bond_r + atom2.bond_r)
                if r < bond_r:
                    el_str = f"{atom.element.symbol}-{atom2.element.symbol}"
                    messages.append(f"{el_str:<5} {atom.id:>5} {atom2.id:>5}")
                    dx = pbc.wrap(dx)
                    dx *= distance / r
                    h_atom.x = other.x + dx
    return mol, messages


def edit_h_dist(mol: Molecule, second_element: str, distance: float) -> Molecule:
    """Adjust X-H distances, returning only the molecule (see adjust_h_distances)."""
    mol, _ = adjust_h_distances(mol, second_element, distance)
    return mol
