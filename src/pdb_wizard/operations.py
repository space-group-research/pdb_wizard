"""System operations: wrap, sort, extend axis, formula unit, etc."""

from __future__ import annotations

import copy
import sys
from functools import reduce
from math import gcd
from typing import Any, Iterator, TextIO

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.io import site_element
from pdb_wizard.molecule import Molecule, set_atom_ids


def progressbar(
    it: list, prefix: str = "", size: int = 60, out: TextIO = sys.stdout,
) -> Iterator:
    count = len(it)
    if count == 0:
        return

    def show(j: int) -> None:
        x = int(size * j / count)
        bar = "#" * x + "." * (size - x)
        print(
            f"{prefix}[{bar}] {j}/{count}",
            end="\r", file=out, flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("", flush=True, file=out)


def gcd_list(values: list[int]) -> int:
    return reduce(gcd, values)


def site_mass(atom: Atom) -> float:
    """Mass of an atom: the model's own value for a movable MPMC site (massless sites are
    0, a united-atom site keeps its group mass), the element mass otherwise."""
    if atom.mpmc_flag == "M" and atom.file_mass is not None:
        return float(atom.file_mass)
    return atom.mass


def real_atoms(mol: Molecule) -> list:
    """(atom, element) for every atom that is a real atom. Sites of MPMC sorbate models
    get their true element ('COG' is carbon, not cobalt) and massless sites are left out."""
    out = []
    for atom in mol.atoms:
        element = site_element(atom)
        if element is not None:
            out.append((atom, element))
    return out


def formula_unit(mol: Molecule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _atom, element in real_atoms(mol):
        sym = element.symbol
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def formula_unit_reduced(mol: Molecule) -> dict[str, int]:
    counts = formula_unit(mol)
    if not counts:
        return counts
    g = gcd_list(list(counts.values()))
    return {k: v // g for k, v in counts.items()}


def wrap_atoms(mol: Molecule, forward: bool = False) -> Molecule:
    if mol.pbc is None:
        return mol
    pbc = mol.pbc
    wrap_fn = pbc.wrap_forward if forward else pbc.wrap
    for atom in mol.atoms:
        atom.x = wrap_fn(atom.x)
    return mol


def sort_system(mol: Molecule, output: bool = True) -> tuple[Molecule, list[Molecule]]:
    if mol.pbc is None:
        return mol, [mol]

    submols = mol.find_molecules()

    for submol in submols:
        submol.atoms.sort(key=lambda a: a.element.symbol)

    submols.sort(key=lambda m: m.atoms[0].element.symbol)
    submols.sort(key=lambda m: len(m.atoms), reverse=True)
    submols.sort(key=lambda m: 0 if max(a.mass for a in m.atoms) > 16 else 1)

    all_atoms: list[Atom] = []
    for submol in submols:
        all_atoms.extend(submol.atoms)
    mol.atoms = all_atoms
    set_atom_ids(mol.atoms)
    mol.detect_bonds()
    return mol, submols


def extend_axis(mol: Molecule, axis: int, times: int) -> Molecule:
    if mol.pbc is None:
        return mol
    pbc = mol.pbc
    new_atoms: list[Atom] = []
    mol_ids = [a.mpmc_mol for a in mol.atoms if a.mpmc_flag == "M" and a.mpmc_mol is not None]
    id_step = max(mol_ids, default=0)
    for i in range(1, times + 1):
        for atom in mol.atoms:
            new_atom = copy.deepcopy(atom)
            new_atom.x = atom.x + i * pbc.basis_matrix[axis]
            if new_atom.mpmc_flag == "M" and new_atom.mpmc_mol is not None:
                new_atom.mpmc_mol += i * id_step  # each copy is its own molecule
            new_atoms.append(new_atom)
    mol.atoms.extend(new_atoms)

    params = [pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma]
    params[axis] *= (times + 1)
    pbc.update(*params)
    set_atom_ids(mol.atoms)
    # The cached bond list only covers the original atoms; drop it so the
    # geometry listings recompute instead of showing a stale, short list.
    mol.bonds = []
    return mol


def void_volume(mol: Molecule, n_samples: int = 100000, probe_radius: float = 0.0) -> dict[str, float]:
    """Calculate void volume using Monte Carlo sampling.

    Samples random points in the unit cell and checks if each point is
    further than (vdw_radius + probe_radius) from all atoms, using PBC
    min-image distances.

    Args:
        mol: Molecule with PBC.
        n_samples: Number of random sample points.
        probe_radius: Radius of probe molecule (0 = geometric void,
            1.2 = H2-sized probe, 1.4 = N2-sized probe).

    Returns:
        Dict with void_volume (A^3), total_volume (A^3), void_fraction.
    """
    if mol.pbc is None:
        return {"void_volume": 0.0, "total_volume": 0.0, "void_fraction": 0.0}

    pbc = mol.pbc
    sites = real_atoms(mol)
    if not sites:  # nothing in the cell: all of it is void
        return {"void_volume": pbc.volume, "total_volume": pbc.volume, "void_fraction": 1.0}
    coords = np.array([a.x for a, _ in sites])
    vdw_radii = np.array([el.vdw_radius for _, el in sites])

    # Generate random fractional coordinates
    rng = np.random.default_rng(42)
    frac_points = rng.random((n_samples, 3))
    cart_points = frac_points @ pbc.basis_matrix

    # Loop over atoms with all sample points vectorized.
    # Uses orthorhombic fast path (no matrix multiply) when possible.
    effective_radii_sq = (vdw_radii + probe_radius) ** 2
    box = np.array([pbc.a, pbc.b, pbc.c])
    is_ortho = (abs(pbc.alpha - 90) < 0.01 and abs(pbc.beta - 90) < 0.01
                and abs(pbc.gamma - 90) < 0.01)

    occupied = np.zeros(n_samples, dtype=bool)

    if is_ortho:
        for ai in range(len(coords)):
            dx = cart_points - coords[ai]
            dx -= np.round(dx / box) * box
            dist_sq = (dx * dx).sum(axis=1)
            occupied |= (dist_sq <= effective_radii_sq[ai])
    else:
        recip = pbc.reciprocal_basis_matrix
        basis = pbc.basis_matrix
        for ai in range(len(coords)):
            dx = cart_points - coords[ai]
            frac_dx = dx @ recip
            frac_dx -= np.round(frac_dx)
            wrapped = frac_dx @ basis
            dist_sq = (wrapped * wrapped).sum(axis=1)
            occupied |= (dist_sq <= effective_radii_sq[ai])

    void_frac = (~occupied).sum() / n_samples
    total_vol = pbc.volume
    return {
        "void_volume": void_frac * total_vol,
        "total_volume": total_vol,
        "void_fraction": void_frac,
    }


def surface_area(
    mol: Molecule, n_samples: int = 100000, probe_radius: float = 1.4,
) -> dict[str, float]:
    """Compute solvent-accessible surface area (SASA) via Monte Carlo.

    For each atom, random points are sampled on a sphere of radius
    (vdw_radius + probe_radius).  A point contributes to SASA only if it
    is not inside any *other* atom's (vdw + probe) sphere.  PBC
    min-image convention is used for all distance calculations.

    Args:
        mol: Molecule with PBC.
        n_samples: Total number of sample points distributed across atoms
            (weighted by sphere surface area).
        probe_radius: Probe molecule radius in angstroms (default 1.4,
            typical N2 probe for BET-style calculations).

    Returns:
        Dict with ``surface_area`` (A^2) and ``area_per_volume`` (m^2/g,
        using total_mass * 1.66054e-24 for grams).
    """
    sites = real_atoms(mol)
    if mol.pbc is None or len(sites) == 0:
        return {"surface_area": 0.0, "area_per_volume": 0.0}

    pbc = mol.pbc
    coords = np.array([a.x for a, _ in sites])              # (N, 3)
    vdw_radii = np.array([el.vdw_radius for _, el in sites])  # (N,)
    radii = vdw_radii + probe_radius                        # (N,)
    n_atoms = len(sites)

    # Distribute samples per atom proportional to sphere surface area (4*pi*r^2).
    areas = radii ** 2                                      # proportional to 4*pi*r^2
    weights = areas / areas.sum()
    samples_per_atom = np.round(weights * n_samples).astype(int)
    samples_per_atom = np.maximum(samples_per_atom, 1)

    rng = np.random.default_rng(42)
    total_sa = 0.0

    basis = pbc.basis_matrix                                # (3, 3)
    recip = pbc.reciprocal_basis_matrix                     # (3, 3)

    # Pre-filter via cell list (O(N) instead of O(N^2)). Cell size = max pair
    # cutoff so a query only inspects the home cell and its 26 neighbors.
    r_max = float(radii.max())
    cutoff = 2 * r_max  # generous: any j within r_i+r_j ≤ 2*r_max
    # Map each atom to a wrapped fractional cell index
    frac = (coords @ recip) % 1.0
    # cells must be at least `cutoff` thick: in a skewed cell that is set by the
    # perpendicular widths, not by the edge lengths
    n_cells = max(1, int(float(pbc.perpendicular_widths().min()) // cutoff))
    cell_idx = np.minimum((frac * n_cells).astype(int), n_cells - 1)
    cell_map: dict[tuple, list[int]] = {}
    for ai, (cx, cy, cz) in enumerate(cell_idx.tolist()):
        cell_map.setdefault((cx, cy, cz), []).append(ai)

    neighbors: list[np.ndarray] = [None] * n_atoms
    for i in range(n_atoms):
        cx, cy, cz = int(cell_idx[i, 0]), int(cell_idx[i, 1]), int(cell_idx[i, 2])
        cand: list[int] = []
        for dx_c in (-1, 0, 1):
            for dy_c in (-1, 0, 1):
                for dz_c in (-1, 0, 1):
                    key = ((cx + dx_c) % n_cells,
                           (cy + dy_c) % n_cells,
                           (cz + dz_c) % n_cells)
                    cand.extend(cell_map.get(key, ()))
        # Filter by actual distance (min-image) and exclude self
        if not cand:
            neighbors[i] = np.empty(0, dtype=int)
            continue
        cand_arr = np.array(cand, dtype=int)
        cand_arr = cand_arr[cand_arr != i]
        if cand_arr.size == 0:
            neighbors[i] = cand_arr
            continue
        dx = coords[cand_arr] - coords[i]
        f = dx @ recip
        f -= np.round(f)
        dx = f @ basis
        dists = np.sqrt((dx ** 2).sum(axis=1))
        cutoff_pair = radii[i] + radii[cand_arr]
        keep_mask = dists < cutoff_pair
        neighbors[i] = cand_arr[keep_mask]

    # Process per atom — only against pre-filtered neighbors
    for i in range(n_atoms):
        n_pts = int(samples_per_atom[i])
        r_i = radii[i]
        nb = neighbors[i]

        if len(nb) == 0:
            total_sa += 4.0 * np.pi * r_i ** 2
            continue

        # Random points on the unit sphere (Marsaglia method)
        pts = rng.standard_normal((n_pts, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts * r_i + coords[i]                         # (n_pts, 3)

        nb_coords = coords[nb]                              # (M, 3)
        nb_radii = radii[nb]                                # (M,)

        # Vectorised min-image distance: pts (n_pts,3) vs nb_coords (M,3)
        # Process in sub-batches to keep memory bounded.
        chunk = max(1, 200_000_000 // (len(nb) * 3 * 8))  # ~200 MB limit
        accessible = np.ones(n_pts, dtype=bool)
        for start in range(0, n_pts, chunk):
            end = min(start + chunk, n_pts)
            batch = pts[start:end]                          # (B, 3)
            dx = nb_coords[np.newaxis, :, :] - batch[:, np.newaxis, :]  # (B, M, 3)
            frac_dx = dx @ recip
            dx = dx - np.round(frac_dx) @ basis
            dists = np.sqrt((dx ** 2).sum(axis=2))          # (B, M)
            inside = (dists < nb_radii[np.newaxis, :]).any(axis=1)
            accessible[start:end] &= ~inside

        frac_accessible = accessible.sum() / n_pts
        total_sa += frac_accessible * 4.0 * np.pi * r_i ** 2

    total_mass = sum(site_mass(a) for a in mol.atoms)
    mass_grams = total_mass * 1.66054e-24                   # amu -> grams
    # surface_area is in A^2; convert to m^2: 1 A^2 = 1e-20 m^2
    area_per_volume = (total_sa * 1e-20) / mass_grams if mass_grams > 0 else 0.0

    return {
        "surface_area": total_sa,
        "area_per_volume": area_per_volume,
    }


def pore_size_distribution(
    mol: Molecule,
    n_samples: int = 50000,
    n_bins: int = 50,
    max_pore: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pore size distribution via Monte Carlo sampling.

    Random points are generated uniformly in the unit cell.  For each
    point the distance to the nearest atom *surface* (centre distance
    minus vdw_radius) is recorded as the pore radius at that point.
    Results are histogrammed and normalised so that the histogram sums
    to 1.  PBC min-image convention is used.

    Args:
        mol: Molecule with PBC.
        n_samples: Number of random sample points.
        n_bins: Number of histogram bins.
        max_pore: Upper limit of the histogram (angstroms).

    Returns:
        Tuple ``(bin_centers, histogram)`` as numpy arrays.
    """
    bin_edges = np.linspace(0.0, max_pore, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    sites = real_atoms(mol)
    if mol.pbc is None or len(sites) == 0:
        return bin_centers, np.zeros(n_bins)

    pbc = mol.pbc
    coords = np.array([a.x for a, _ in sites])              # (N, 3)
    vdw_radii = np.array([el.vdw_radius for _, el in sites])  # (N,)
    n_atoms = len(sites)

    basis = pbc.basis_matrix                                # (3, 3)
    recip = pbc.reciprocal_basis_matrix                     # (3, 3)

    rng = np.random.default_rng(42)
    frac_points = rng.random((n_samples, 3))
    cart_points = frac_points @ basis                       # (n_samples, 3)

    # Process sample points in chunks to keep memory bounded
    chunk = max(1, 200_000_000 // (n_atoms * 3 * 8))       # ~200 MB limit
    pore_radii = np.empty(n_samples)

    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        batch = cart_points[start:end]                      # (B, 3)
        dx = coords[np.newaxis, :, :] - batch[:, np.newaxis, :]  # (B, N, 3)
        frac_dx = dx @ recip
        dx = dx - np.round(frac_dx) @ basis                # min-image
        dists = np.sqrt((dx ** 2).sum(axis=2))              # (B, N)
        # Pore radius = distance to nearest atom surface
        surface_dists = dists - vdw_radii[np.newaxis, :]    # (B, N)
        pore_radii[start:end] = surface_dists.min(axis=1)

    # Clamp negatives to zero (point inside an atom)
    pore_radii = np.clip(pore_radii, 0.0, None)

    histogram, _ = np.histogram(pore_radii, bins=bin_edges)
    histogram = histogram.astype(float)
    total = histogram.sum()
    if total > 0:
        histogram /= total

    return bin_centers, histogram


def system_info(mol: Molecule, filename: str) -> dict[str, Any]:
    info: dict[str, Any] = {"filename": filename}
    if mol.pbc is not None:
        pbc = mol.pbc
        info["cell"] = {
            "a": pbc.a, "b": pbc.b, "c": pbc.c,
            "alpha": pbc.alpha, "beta": pbc.beta, "gamma": pbc.gamma,
        }
        info["volume"] = pbc.volume
        total_mass = sum(site_mass(a) for a in mol.atoms)
        info["density"] = total_mass * 1.66054 / pbc.volume
        info["basis_matrix"] = pbc.basis_matrix.tolist()
    info["n_atoms"] = len(mol.atoms)
    info["formula"] = formula_unit(mol)
    info["formula_reduced"] = formula_unit_reduced(mol)
    return info


def print_info(mol: Molecule, filename: str) -> None:
    print("")
    print(r"   ___  ___  ___   __    __ _                  _ ")
    print(r"  / _ \/   \/ __\ / / /\ \ (_)______ _ _ __ __| |")
    print(r" / /_)/ /\ /__\// \ \/  \/ / |_  / _` | '__/ _` |")
    print(r"/ ___/ /_// \/  \  \  /\  /| |/ / (_| | | | (_| |")
    print(r"\/  /___,'\_____/   \/  \/ |_/___\__,_|_|  \__,_|")
    print(f"\nfilename: {filename}")
    if mol.pbc is not None:
        pbc = mol.pbc
        print(
            f"\nCell:\n{round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
            f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6}\n"
        )
        for row in pbc.basis_matrix:
            print(f"{row[0]:20.14f} {row[1]:20.14f} {row[2]:20.14f}")
        total_mass = sum(site_mass(a) for a in mol.atoms)
        density = total_mass * 1.66054 / pbc.volume
        print(f"Volume: {pbc.volume:10.2f} A^3 Density: {density:10.4f} g/cm^3")
    counts = formula_unit(mol)
    print("\nTotal number of atoms:\n")
    for ele, n in counts.items():
        print(f"{ele} {n}")
    reduced = formula_unit_reduced(mol)
    print("\nFormula unit\n")
    for ele, n in reduced.items():
        print(f"{ele} {n}")


def print_info_movie(n_frames: int, filename: str) -> None:
    print("")
    print(r"   ___  ___  ___   __    __ _                  _ ")
    print(r"  / _ \/   \/ __\ / / /\ \ (_)______ _ _ __ __| |")
    print(r" / /_)/ /\ /__\// \ \/  \/ / |_  / _` | '__/ _` |")
    print(r"/ ___/ /_// \/  \  \  /\  /| |/ / (_| | | | (_| |")
    print(r"\/  /___,'\_____/   \/  \/ |_/___\__,_|_|  \__,_|")
    print("")
    print(f"Trajectory detected\n{n_frames} frames\nExtract a single frame for more detailed options")
