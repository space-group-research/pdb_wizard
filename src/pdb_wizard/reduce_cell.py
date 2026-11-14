"""Reduce a supercell back to a single unit cell, with sorbate selection."""

from __future__ import annotations

import copy

import numpy as np

from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

# Two movable molecules whose sites all coincide within this (A) once folded into the
# small cell are periodic copies of one molecule.
_REDUCE_COPY_TOL = 0.05


def identify_framework_and_sorbates(
    mol: Molecule,
    na: int, nb: int, nc: int,
    overlap_tol: float = 0.5,
) -> tuple[list[int], list[list[int]]]:
    """Identify framework atoms (periodic copies) vs sorbate atoms (unique).

    Given a supercell of dimensions na x nb x nc, wrap all atoms into the
    primitive cell. Framework atoms will cluster into groups of na*nb*nc
    overlapping copies. Sorbate atoms are unique — they don't overlap.

    Args:
        mol: Supercell molecule with PBC.
        na, nb, nc: Supercell dimensions along a, b, c.
        overlap_tol: Distance tolerance for considering atoms as copies (A).

    Returns:
        (framework_indices, sorbate_groups):
        framework_indices: indices of one representative atom per framework site
        sorbate_groups: list of groups, each group is a list of atom indices
            belonging to the same sorbate molecule
    """
    if mol.pbc is None or not mol.atoms:
        return [], []

    pbc = mol.pbc
    n_mult = na * nb * nc
    n_atoms = len(mol.atoms)

    # Compute fractional coordinates in the supercell
    coords = np.array([a.x for a in mol.atoms])
    frac = coords @ pbc.reciprocal_basis_matrix

    # Scale fractional coords to primitive cell [0, na) -> [0, 1)
    prim_frac = frac.copy()
    prim_frac[:, 0] *= na
    prim_frac[:, 1] *= nb
    prim_frac[:, 2] *= nc
    # Wrap to [0, 1)
    prim_frac -= np.floor(prim_frac)

    # Primitive cell basis
    prim_basis = pbc.basis_matrix.copy()
    prim_basis[0] /= na
    prim_basis[1] /= nb
    prim_basis[2] /= nc

    # Convert back to Cartesian in primitive cell
    prim_cart = prim_frac @ prim_basis

    # Cluster atoms by position: framework atoms cluster into groups of n_mult
    # Precompute inverse for min-image in primitive cell
    prim_inv = np.linalg.inv(prim_basis)

    np.zeros(n_atoms, dtype=bool)
    groups: list[list[int]] = []

    # Group by element first to reduce O(n^2) comparisons. Movable (M) molecules are
    # sorbates by definition and are handled below: their sites can sit within the
    # tolerance of each other (PHAHST H2, the CON/COE sites of CO2) and would otherwise
    # be mistaken for periodic copies of one framework atom.
    elem_indices: dict[str, list[int]] = {}
    for i, a in enumerate(mol.atoms):
        if a.mpmc_flag != "M":
            elem_indices.setdefault(a.element.symbol, []).append(i)

    for el, indices in elem_indices.items():
        el_used = set()
        for ii, i in enumerate(indices):
            if i in el_used:
                continue
            group = [i]
            el_used.add(i)
            for j in indices[ii + 1:]:
                if j in el_used:
                    continue
                diff = prim_cart[j] - prim_cart[i]
                frac_diff = diff @ prim_inv
                frac_diff -= np.round(frac_diff)
                cart_diff = frac_diff @ prim_basis
                dist = float(np.linalg.norm(cart_diff))
                if dist < overlap_tol:
                    group.append(j)
                    el_used.add(j)
            groups.append(group)

    # Framework groups have exactly n_mult members (periodic copies)
    # Sorbate groups have fewer
    framework_indices = []
    sorbate_groups = []

    for group in groups:
        if len(group) == n_mult:
            # Framework — keep one representative
            framework_indices.append(group[0])
        else:
            sorbate_groups.append(group)

    n_frozen = sum(len(indices) for indices in elem_indices.values())
    n_periodic = n_mult * len(framework_indices)
    if n_mult > 1 and n_frozen and 2 * n_periodic < n_frozen:
        raise ValueError(
            f"this is not a {na} x {nb} x {nc} supercell: only {n_periodic} of its "
            f"{n_frozen} framework atoms repeat {n_mult} times under the sub-cell "
            f"translations. Check the numbers of cells along a, b and c.")

    # Merge sorbate atoms into molecules using bond connectivity
    # Build neighbor map from existing bonds
    neighbors: dict[int, set[int]] = {}
    for a, b in mol.bonds:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    # BFS to find connected sorbate molecules
    sorbate_atoms = set()
    for sg in sorbate_groups:
        sorbate_atoms.update(sg)

    visited = set()
    sorbate_molecules: list[list[int]] = []
    for seed_group in sorbate_groups:
        for seed in seed_group:
            if seed in visited:
                continue
            # BFS from seed
            mol_atoms = []
            queue = [seed]
            while queue:
                atom = queue.pop(0)
                if atom in visited:
                    continue
                visited.add(atom)
                mol_atoms.append(atom)
                for nb_idx in neighbors.get(atom, set()):
                    if nb_idx in sorbate_atoms and nb_idx not in visited:
                        queue.append(nb_idx)
            if mol_atoms:
                sorbate_molecules.append(sorted(mol_atoms))

    # Movable molecules, one entry per molecule id. Periodic copies of the same
    # molecule (a supercell made with Extend Axis after the sorbates were added) land
    # on top of each other in the small cell: keep the first of each.
    movable: dict = {}
    for i, a in enumerate(mol.atoms):
        if a.mpmc_flag == "M":
            movable.setdefault(a.mpmc_mol if a.mpmc_mol is not None else ("atom", i), []).append(i)
    kept: list[list[int]] = []
    for indices in movable.values():
        duplicate = False
        for other in kept:
            if len(other) != len(indices):
                continue
            diff = (prim_cart[indices] - prim_cart[other]) @ prim_inv
            diff -= np.round(diff)
            if (np.linalg.norm(diff @ prim_basis, axis=1) < _REDUCE_COPY_TOL).all() and all(
                    mol.atoms[i].name == mol.atoms[j].name for i, j in zip(indices, other)):
                duplicate = True
                break
        if not duplicate:
            kept.append(indices)
    sorbate_molecules.extend(kept)

    return framework_indices, sorbate_molecules


def reduce_to_primitive(
    mol: Molecule,
    na: int, nb: int, nc: int,
    keep_sorbate_indices: list[int] | None = None,
    overlap_tol: float = 0.5,
) -> Molecule:
    """Reduce a supercell to a single unit cell.

    Args:
        mol: Supercell molecule.
        na, nb, nc: Supercell dimensions.
        keep_sorbate_indices: Which sorbate molecule indices to keep
            (from identify_framework_and_sorbates). None = keep all.
        overlap_tol: Distance tolerance for overlap detection.

    Returns:
        New Molecule with primitive cell PBC and reduced atoms.
    """
    if mol.pbc is None:
        return mol

    framework_idx, sorbate_mols = identify_framework_and_sorbates(
        mol, na, nb, nc, overlap_tol,
    )

    pbc = mol.pbc
    prim_pbc = PBC(
        pbc.a / na, pbc.b / nb, pbc.c / nc,
        pbc.alpha, pbc.beta, pbc.gamma,
    )

    # Collect atoms: framework + selected sorbates
    keep_atoms = list(framework_idx)
    if keep_sorbate_indices is not None:
        for si in keep_sorbate_indices:
            if 0 <= si < len(sorbate_mols):
                keep_atoms.extend(sorbate_mols[si])
    else:
        for sm in sorbate_mols:
            keep_atoms.extend(sm)

    # Build new atom list with positions wrapped into primitive cell
    prim_basis = prim_pbc.basis_matrix
    prim_recip = prim_pbc.reciprocal_basis_matrix
    # A movable molecule moves as a whole (by the lattice vector that brings its first
    # site into the cell), so it stays the rigid model MPMC expects.
    first_site: dict = {}
    for sm in sorbate_mols:
        if all(mol.atoms[i].mpmc_flag == "M" for i in sm):
            for i in sm:
                first_site[i] = sm[0]
    new_atoms = []
    for idx in sorted(keep_atoms):
        a = mol.atoms[idx]
        if idx in first_site:
            origin = mol.atoms[first_site[idx]].x
            rel = (a.x - origin) @ prim_recip
            rel -= np.round(rel)
            frac = origin @ prim_recip
            frac = frac - np.floor(frac) + rel
        else:
            # Convert to fractional in primitive cell
            frac = a.x @ prim_recip
            frac -= np.floor(frac)
        new_pos = frac @ prim_basis
        # a full copy: charge and LJ parameters, and also C6/C8/C10, the movable flag,
        # molecule id, residue name and model mass
        new_atom = copy.deepcopy(a)
        new_atom.x = np.array([float(new_pos[0]), float(new_pos[1]), float(new_pos[2])])
        new_atoms.append(new_atom)

    result = Molecule(atoms=new_atoms, pbc=prim_pbc)
    result.detect_bonds()
    return result
