"""Molecule class combining moltui geometry methods with pdb_wizard PBC-aware logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.pbc import PBC


@dataclass
class Molecule:
    atoms: list[Atom]
    bonds: list[tuple[int, int]] = field(default_factory=list)
    pbc: Optional[PBC] = None

    def center(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros(3)
        positions = np.array([a.x for a in self.atoms])
        return positions.mean(axis=0)

    def radius(self) -> float:
        if not self.atoms:
            return 1.0
        positions = np.array([a.x for a in self.atoms])
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(distances.max()) if len(distances) > 0 else 1.0

    def _adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(len(self.atoms))}
        for i, j in self.bonds:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def get_bond_lengths(self) -> list[tuple[int, int, float]]:
        results = []
        for i, j in self.bonds:
            if self.pbc is not None:
                dx = self.atoms[i].x - self.atoms[j].x
                dist = self.pbc.min_image(dx)
            else:
                dist = float(np.linalg.norm(self.atoms[i].x - self.atoms[j].x))
            results.append((i, j, dist))
        return results

    def get_angles(self) -> list[tuple[int, int, int, float]]:
        adj = self._adjacency()
        results = []
        for j, neighbors in adj.items():
            for ni, i in enumerate(neighbors):
                for k in neighbors[ni + 1:]:
                    if self.pbc is not None:
                        v1 = self.pbc.wrap(self.atoms[i].x - self.atoms[j].x)
                        v2 = self.pbc.wrap(self.atoms[k].x - self.atoms[j].x)
                    else:
                        v1 = self.atoms[i].x - self.atoms[j].x
                        v2 = self.atoms[k].x - self.atoms[j].x
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                    results.append((i, j, k, angle))
        return results

    def get_dihedrals(self) -> list[tuple[int, int, int, int, float]]:
        adj = self._adjacency()
        results: list[tuple[int, int, int, int, float]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for j, k in self.bonds:
            for i in adj[j]:
                if i == k:
                    continue
                for l_atom in adj[k]:
                    if l_atom == j or l_atom == i:
                        continue
                    key = (i, j, k, l_atom) if i < l_atom else (l_atom, k, j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    if self.pbc is not None:
                        b1 = self.pbc.wrap(self.atoms[j].x - self.atoms[i].x)
                        b2 = self.pbc.wrap(self.atoms[k].x - self.atoms[j].x)
                        b3 = self.pbc.wrap(self.atoms[l_atom].x - self.atoms[k].x)
                    else:
                        b1 = self.atoms[j].x - self.atoms[i].x
                        b2 = self.atoms[k].x - self.atoms[j].x
                        b3 = self.atoms[l_atom].x - self.atoms[k].x
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    n1_norm = np.linalg.norm(n1) + 1e-10
                    n2_norm = np.linalg.norm(n2) + 1e-10
                    cos_d = np.dot(n1, n2) / (n1_norm * n2_norm)
                    angle = float(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0))))
                    results.append((i, j, k, l_atom, angle))
        return results

    def detect_bonds(self, tolerance: float = 1.3, progress_callback=None) -> None:
        self.bonds = []
        n = len(self.atoms)
        if n < 2:
            return

        if self.pbc is not None:
            # Use all-pairs for small systems (vectorized numpy beats Python loop overhead)
            if n < 1500:
                coords = np.array([a.x for a in self.atoms])
                bond_rs = np.array([a.bond_r for a in self.atoms], dtype=float)
                self._detect_bonds_all_pairs(coords, bond_rs, progress_callback)
                return

            coords = np.array([a.x for a in self.atoms])
            bond_rs = np.array([a.bond_r for a in self.atoms], dtype=float)
            cutoff = float(bond_rs.max())

            recip = self.pbc.reciprocal_basis_matrix
            basis = self.pbc.basis_matrix
            frac = coords @ recip
            frac -= np.floor(frac)

            box = np.array([self.pbc.a, self.pbc.b, self.pbc.c])
            n_cells = np.maximum(1, (box / cutoff).astype(int))
            if n_cells.min() < 3:
                self._detect_bonds_all_pairs(coords, bond_rs, progress_callback)
                return

            cell_idx = np.floor(frac * n_cells).astype(int) % n_cells
            grid: dict[tuple, list[int]] = {}
            for ai in range(n):
                key = (int(cell_idx[ai, 0]), int(cell_idx[ai, 1]), int(cell_idx[ai, 2]))
                grid.setdefault(key, []).append(ai)

            # For each atom, only check atoms in its cell + 26 neighbors
            bonds_set: set[tuple[int, int]] = set()
            for ai in range(n):
                ci = (int(cell_idx[ai, 0]), int(cell_idx[ai, 1]), int(cell_idx[ai, 2]))
                candidates = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            key = (
                                (ci[0] + di) % n_cells[0],
                                (ci[1] + dj) % n_cells[1],
                                (ci[2] + dk) % n_cells[2],
                            )
                            if key in grid:
                                candidates.extend(grid[key])
                if not candidates:
                    continue
                cands = np.array([c for c in candidates if c > ai], dtype=int)
                if len(cands) == 0:
                    continue
                dx = coords[cands] - coords[ai]
                f = dx @ recip
                f -= np.round(f)
                w = f @ basis
                rs = np.sqrt((w * w).sum(axis=1))
                mixed = 0.5 * (bond_rs[cands] + bond_rs[ai])
                mask = rs < mixed
                for j in cands[mask]:
                    bonds_set.add((ai, int(j)))
                if progress_callback is not None and ai % max(1, n // 20) == 0:
                    progress_callback(ai / n)
            self.bonds = sorted(bonds_set)
            if progress_callback is not None:
                progress_callback(1.0)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(self.atoms[i].x - self.atoms[j].x))
                    max_bond = (
                        self.atoms[i].element.covalent_radius
                        + self.atoms[j].element.covalent_radius
                    ) * tolerance
                    if dist < max_bond:
                        self.bonds.append((i, j))

    def _detect_bonds_all_pairs(self, coords, bond_rs, progress_callback=None) -> None:
        """Fallback all-pairs algorithm for small cells where spatial hashing has too few cells."""
        n = len(coords)
        i_all, j_all = np.triu_indices(n, k=1)
        total_pairs = len(i_all)
        chunk_size = max(500000, total_pairs // 20)
        all_bonds_i = []
        all_bonds_j = []
        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            i_chunk = i_all[start:end]
            j_chunk = j_all[start:end]
            dx = coords[i_chunk] - coords[j_chunk]
            frac = np.matmul(dx, self.pbc.reciprocal_basis_matrix)
            wrapped = dx - np.matmul(np.round(frac), self.pbc.basis_matrix)
            rs = np.linalg.norm(wrapped, axis=1)
            mixed = 0.5 * (bond_rs[i_chunk] + bond_rs[j_chunk])
            mask = rs < mixed
            all_bonds_i.append(i_chunk[mask])
            all_bonds_j.append(j_chunk[mask])
            if progress_callback is not None:
                progress_callback(end / total_pairs)
        if all_bonds_i:
            bi = np.concatenate(all_bonds_i)
            bj = np.concatenate(all_bonds_j)
            self.bonds = list(zip(bi.tolist(), bj.tolist()))

    def find_molecules(self) -> list[Molecule]:
        # Use cached bonds when available — detect_bonds uses an O(N) spatial
        # hash, much faster than _find_molecules_pbc's O(N²) pairwise scan.
        if not self.bonds:
            self.detect_bonds()
        return _find_molecules_from_bonds(self.atoms, self.bonds, self.pbc)


def set_atom_ids(system: list[Atom]) -> None:
    for ind, atom in enumerate(system):
        atom.id = ind + 1


def _find_molecules_pbc(atoms: list[Atom], pbc: PBC) -> list[Molecule]:
    i_idxs, j_idxs, rs = pbc.get_all_rs_min_image(atoms)
    bond_rs = np.array([a.bond_r for a in atoms], dtype=float)
    mixed = 0.5 * (bond_rs[i_idxs] + bond_rs[j_idxs])
    bond_mask = mixed > rs
    edge_list = np.stack((i_idxs[bond_mask], j_idxs[bond_mask]), axis=1) if bond_mask.any() else np.empty((0, 2), dtype=int)

    try:
        import graph_tool.all as gt
        g = gt.Graph(g=len(atoms), directed=False)
        g.add_edge_list(edge_list)
        comp, _ = gt.label_components(g)
        mol_labels = np.array(comp.a)
        n_mols = int(np.max(mol_labels)) + 1 if len(mol_labels) > 0 else 0
        mols = []
        for mol_idx in range(n_mols):
            mol_atoms = [atoms[i] for i in range(len(atoms)) if mol_labels[i] == mol_idx]
            mols.append(Molecule(atoms=mol_atoms, pbc=pbc))
        return mols
    except ImportError:
        pass

    set_atom_ids(atoms)
    bonds: dict[int, list[int]] = {}
    for i_idx, j_idx in edge_list:
        bonds.setdefault(int(i_idx), [])
        if int(j_idx) not in bonds[int(i_idx)]:
            bonds[int(i_idx)].append(int(j_idx))
        bonds.setdefault(int(j_idx), [])
        if int(i_idx) not in bonds[int(j_idx)]:
            bonds[int(j_idx)].append(int(i_idx))

    visited: set[int] = set()
    mols_by_idx: list[list[int]] = []

    for idx in range(len(atoms)):
        if idx in visited:
            continue
        if idx not in bonds:
            mols_by_idx.append([idx])
            visited.add(idx)
            continue
        new_mol = [idx]
        visited.add(idx)
        queue = list(bonds.get(idx, []))
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            new_mol.append(current)
            for neighbor in bonds.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        mols_by_idx.append(new_mol)

    return [Molecule(atoms=[atoms[i] for i in mol_idxs], pbc=pbc) for mol_idxs in mols_by_idx]


def _find_molecules_from_bonds(
    atoms: list[Atom], bonds: list[tuple[int, int]], pbc: PBC | None
) -> list[Molecule]:
    adj: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    visited: set[int] = set()
    mols: list[Molecule] = []
    for idx in range(len(atoms)):
        if idx in visited:
            continue
        component = []
        queue = [idx]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        mol_atoms = [atoms[i] for i in component]
        comp_set = set(component)
        old_to_new = {old: new for new, old in enumerate(component)}
        mol_bonds = [
            (old_to_new[i], old_to_new[j])
            for i, j in bonds if i in comp_set and j in comp_set
        ]
        mols.append(Molecule(atoms=mol_atoms, bonds=mol_bonds, pbc=pbc))
    return mols
