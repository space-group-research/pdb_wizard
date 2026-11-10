"""QEq charge equilibration (Rappe & Goddard, J. Phys. Chem. 1991, 95, 3358).

Assigns partial atomic charges by equalizing electronegativity across all atoms
subject to total charge neutrality. Only requires atomic parameters (chi, J)
and interatomic distances.
"""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule

# Rappe-Goddard QEq parameters: chi (eV), J (eV)
# From OpenBabel qeq.txt / UFF GMP parameters
# fmt: off
QEQ_PARAMS: dict[str, tuple[float, float]] = {
    "H":  (4.5280, 13.8904),
    "He": (9.66,   29.84),
    "Li": (3.006,   4.772),
    "Be": (4.877,   8.886),
    "B":  (5.11,    9.50),
    "C":  (5.343,  10.126),
    "N":  (6.899,  11.760),
    "O":  (8.741,  13.364),
    "F":  (10.874, 14.948),
    "Ne": (11.04,  21.10),
    "Na": (2.843,   4.592),
    "Mg": (3.951,   7.386),
    "Al": (4.06,    7.18),
    "Si": (4.168,   6.974),
    "P":  (5.463,   8.000),
    "S":  (6.928,   8.972),
    "Cl": (8.564,   9.892),
    "Ar": (9.465,  12.71),
    "K":  (2.421,   3.840),
    "Ca": (3.231,   5.76),
    "Sc": (3.395,   6.16),
    "Ti": (3.47,    6.76),
    "V":  (3.65,    6.82),
    "Cr": (3.415,   7.73),
    "Mn": (3.325,   8.21),
    "Fe": (3.76,    8.28),
    "Co": (4.105,   8.35),
    "Ni": (4.465,   8.41),
    "Cu": (4.20,    8.44),
    "Zn": (5.106,   8.57),
    "Ga": (3.641,   6.32),
    "Ge": (4.051,   6.876),
    "As": (5.188,   7.618),
    "Se": (6.428,   8.262),
    "Br": (7.790,   8.850),
    "Kr": (8.505,  11.43),
    "Rb": (2.331,   3.692),
    "Sr": (3.024,   4.88),
    "Y":  (3.83,    5.62),
    "Zr": (3.40,    7.10),
    "Nb": (3.55,    6.76),
    "Mo": (3.465,   7.51),
    "Tc": (3.29,    7.98),
    "Ru": (3.575,   8.03),
    "Rh": (3.975,   8.01),
    "Pd": (4.32,    8.00),
    "Ag": (4.436,   6.268),
    "Cd": (5.034,   7.914),
    "In": (3.506,   5.792),
    "Sn": (3.987,   6.248),
    "Sb": (4.899,   6.684),
    "Te": (5.816,   7.052),
    "I":  (6.822,   7.524),
    "Xe": (7.595,   9.95),
    "Cs": (2.183,   3.422),
    "Ba": (2.814,   4.792),
    "La": (2.8355,  5.483),
    "Ce": (2.774,   5.384),
    "Pr": (2.858,   5.128),
    "Nd": (2.8685,  5.241),
    "Pm": (2.881,   5.346),
    "Sm": (2.9115,  5.439),
    "Eu": (2.8785,  5.575),
    "Gd": (3.1665,  5.949),
    "Tb": (3.018,   5.668),
    "Dy": (3.0555,  5.743),
    "Ho": (3.127,   5.782),
    "Er": (3.1865,  5.829),
    "Tm": (3.2514,  5.8658),
    "Yb": (3.2889,  5.93),
    "Lu": (2.9629,  4.9258),
    "Hf": (3.70,    6.80),
    "Ta": (5.10,    5.70),
    "W":  (4.63,    6.62),
    "Re": (3.96,    7.84),
    "Os": (5.14,    7.26),
    "Ir": (5.00,    8.00),
    "Pt": (4.79,    8.86),
    "Au": (4.894,   5.172),
    "Hg": (6.27,    8.32),
    "Tl": (3.20,    5.80),
    "Pb": (3.90,    7.06),
    "Bi": (4.69,    7.48),
}
# fmt: on

# Coulomb constant: 14.4 eV*A/e^2
_KE = 14.3996


def compute_qeq_charges(
    mol: Molecule,
    total_charge: float = 0.0,
    progress_callback=None,
) -> np.ndarray:
    """Compute QEq charges for all atoms in the molecule.

    Uses Rappe-Goddard charge equilibration with a cutoff distance for
    the Coulomb interaction to keep memory and compute manageable.

    Args:
        mol: Molecule with atoms (and optionally PBC).
        total_charge: Total system charge (0 for neutral).
        progress_callback: Optional callable(frac: float) for progress updates.

    Returns:
        Array of partial charges (electrons).
    """
    n = len(mol.atoms)
    if n == 0:
        return np.array([])

    # Get parameters
    chi = np.zeros(n)
    J_diag = np.zeros(n)
    for i, atom in enumerate(mol.atoms):
        sym = atom.element.symbol
        if sym in QEQ_PARAMS:
            chi[i], J_diag[i] = QEQ_PARAMS[sym]
        else:
            chi[i], J_diag[i] = 5.0, 10.0

    coords = np.array([a.x for a in mol.atoms])

    if progress_callback:
        progress_callback(0.05)

    if n > 3000:
        raise ValueError(
            f"System too large for QEq ({n} atoms). "
            "Try on a single unit cell before extending."
        )

    b = np.zeros(n + 1)
    for i in range(n):
        b[i] = -chi[i]
    b[n] = total_charge

    if progress_callback:
        progress_callback(0.1)

    # Compute pairwise Coulomb interactions chunked
    i_all, j_all = np.triu_indices(n, k=1)
    total_pairs = len(i_all)
    chunk_size = max(500000, total_pairs // 10)

    # Collect COO data for off-diagonal entries
    row_list = []
    col_list = []
    val_list = []

    for start in range(0, total_pairs, chunk_size):
        end = min(start + chunk_size, total_pairs)
        ic = i_all[start:end]
        jc = j_all[start:end]
        dx = coords[ic] - coords[jc]
        if mol.pbc is not None:
            frac = dx @ mol.pbc.reciprocal_basis_matrix
            dx = dx - np.round(frac) @ mol.pbc.basis_matrix
        rs = np.linalg.norm(dx, axis=1)
        rs = np.maximum(rs, 0.1)
        j_ij = _KE / rs

        # Apply cutoff for memory: ignore very weak interactions
        cutoff = 20.0
        mask = rs < cutoff
        row_list.append(ic[mask])
        row_list.append(jc[mask])
        col_list.append(jc[mask])
        col_list.append(ic[mask])
        val_list.append(j_ij[mask])
        val_list.append(j_ij[mask])

        if progress_callback:
            progress_callback(0.1 + 0.6 * end / total_pairs)

    # Build dense matrix (safe up to ~5000 atoms = 200MB)
    A = np.zeros((n + 1, n + 1))

    # Diagonal
    for i in range(n):
        A[i, i] = J_diag[i]

    # Off-diagonal from COO lists
    if row_list:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
        # Direct assignment is safe: each (row,col) appears exactly once
        A[rows, cols] = vals

    # Lagrange multiplier
    A[n, :n] = 1.0
    A[:n, n] = 1.0

    if progress_callback:
        progress_callback(0.8)

    # Solve
    try:
        result = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    charges = result[:n]

    if progress_callback:
        progress_callback(1.0)

    return charges


def apply_qeq_charges(mol: Molecule, progress_callback=None) -> None:
    """Compute and apply QEq charges to the framework atoms.

    Movable (M) molecules read from an MPMC file keep their model charges and
    are left out of the equalization.
    """
    framework = [a for a in mol.atoms if a.mpmc_flag != "M"]
    if len(framework) != len(mol.atoms):
        mol = Molecule(atoms=framework, pbc=mol.pbc)
    charges = compute_qeq_charges(mol, progress_callback=progress_callback)
    for i, atom in enumerate(mol.atoms):
        atom.charge = float(charges[i])
