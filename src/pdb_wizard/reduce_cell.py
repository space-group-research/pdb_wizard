"""Reduce a supercell back to a single unit cell, with sorbate selection."""

from __future__ import annotations

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC


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

    # Group by element first to reduce O(n^2) comparisons
    elem_indices: dict[str, list[int]] = {}
    for i, a in enumerate(mol.atoms):
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
    new_atoms = []
    for idx in sorted(keep_atoms):
        a = mol.atoms[idx]
        # Convert to fractional in primitive cell
        frac = a.x @ prim_pbc.reciprocal_basis_matrix
        frac -= np.floor(frac)
        new_pos = frac @ prim_basis
        new_atom = Atom(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]), a.name)
        new_atom.charge = a.charge
        new_atom.alpha = a.alpha
        new_atom.epsilon = a.epsilon
        new_atom.sigma = a.sigma
        new_atoms.append(new_atom)

    result = Molecule(atoms=new_atoms, pbc=prim_pbc)
    result.detect_bonds()
    return result
