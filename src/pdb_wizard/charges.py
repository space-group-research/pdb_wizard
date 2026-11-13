"""QEq charge equilibration (Rappe & Goddard, J. Phys. Chem. 1991, 95, 3358).

Assigns partial atomic charges by equalizing electronegativity across all atoms
subject to total charge neutrality. Only requires atomic parameters (chi, J)
and interatomic distances.
"""

from __future__ import annotations

from typing import Optional

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

# GMP orbital radii R (Å) from Rappé & Goddard; with chi and J they are the UFF "Xi",
# "Hard" (= J/2) and "Radius" columns (RDKit/Open Babel UFF.prm).
# fmt: off
QEQ_RADII: dict[str, float] = {
    "H": 0.371, "He": 1.3, "Li": 1.557, "Be": 1.24, "B": 0.822, "C": 0.759, "N": 0.715,
    "O": 0.669, "F": 0.706, "Ne": 1.768, "Na": 2.085, "Mg": 1.5, "Al": 1.201, "Si": 1.176,
    "P": 1.102, "S": 1.047, "Cl": 0.994, "Ar": 2.108, "K": 2.586, "Ca": 2.0, "Sc": 1.75,
    "Ti": 1.607, "V": 1.47, "Cr": 1.402, "Mn": 1.533, "Fe": 1.393, "Co": 1.406, "Ni": 1.398,
    "Cu": 1.434, "Zn": 1.4, "Ga": 1.211, "Ge": 1.189, "As": 1.204, "Se": 1.224, "Br": 1.141,
    "Kr": 2.27, "Rb": 2.77, "Sr": 2.415, "Y": 1.998, "Zr": 1.758, "Nb": 1.603, "Mo": 1.53,
    "Tc": 1.5, "Ru": 1.5, "Rh": 1.509, "Pd": 1.544, "Ag": 1.622, "Cd": 1.6, "In": 1.404,
    "Sn": 1.354, "Sb": 1.404, "Te": 1.38, "I": 1.333, "Xe": 2.459, "Cs": 2.984, "Ba": 2.442,
    "La": 2.071, "Ce": 1.925, "Pr": 2.007, "Nd": 2.007, "Pm": 2.0, "Sm": 1.978, "Eu": 2.227,
    "Gd": 1.968, "Tb": 1.954, "Dy": 1.934, "Ho": 1.925, "Er": 1.915, "Tm": 2.0, "Yb": 2.158,
    "Lu": 1.896, "Hf": 1.759, "Ta": 1.605, "W": 1.538, "Re": 1.6, "Os": 1.7, "Ir": 1.866,
    "Pt": 1.557, "Au": 1.618, "Hg": 1.6, "Tl": 1.53, "Pb": 1.444, "Bi": 1.514,
}
# fmt: on

# Coulomb constant: 14.4 eV*A/e^2
_KE = 14.3996
# Slater exponent scaling: zeta = LAMBDA (2n + 1) / (2 R)  (Rappé & Goddard 1991)
_LAMBDA = 0.4913
_DEFAULT_PARAMS = (5.0, 10.0, 1.2)  # chi, J, R for elements without parameters

# Short-range shielding is tabulated out to this distance (Å); beyond it J_AB = K/R to
# better than 1e-7 eV for every element pair.
_SHIELD_CUTOFF = 10.0
_SHIELD_STEP = 0.002


def _principal_n(atomic_number: int) -> int:
    """Principal quantum number of the valence shell (the Slater ns orbital used)."""
    for n, last in ((1, 2), (2, 10), (3, 18), (4, 36), (5, 54)):
        if atomic_number <= last:
            return n
    return 6


def _slater_density_ft(k: np.ndarray, zeta: float, n: int) -> np.ndarray:
    """Fourier transform of a normalized Slater ns density |phi|^2 ~ r^(2n-2) e^(-2 zeta r)."""
    a = 2.0 * zeta
    k = np.asarray(k, dtype=float)
    out = np.ones_like(k)
    nz = k > 1e-12
    kk = k[nz]
    out[nz] = a ** (2 * n + 1) * np.imag((a - 1j * kk) ** (-2 * n)) / (2 * n * kk)
    return out


def slater_coulomb(zeta_a: float, n_a: int, zeta_b: float, n_b: int,
                   r: np.ndarray) -> np.ndarray:
    """Coulomb integral (1/Å) between two Slater ns densities centred ``r`` Å apart.

    J(R) = (2/pi) Int_0^inf rhoA(k) rhoB(k) sin(kR)/(kR) dk. Tends to 1/R at long range
    and stays finite at R = 0 (5 zeta / 8 for two identical 1s densities).
    """
    k_max = 80.0 * max(zeta_a, zeta_b)  # rhoA * rhoB < 1e-11 beyond this
    k = np.linspace(0.0, k_max, 60001)
    w = _slater_density_ft(k, zeta_a, n_a) * _slater_density_ft(k, zeta_b, n_b)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(r)
    dk = k[1] - k[0]
    for i, R in enumerate(r):
        kr = k * R
        sinc = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
        f = w * sinc
        out[i] = (2.0 / np.pi) * dk * (f.sum() - 0.5 * (f[0] + f[-1]))
    return out


def _erfc(x: np.ndarray) -> np.ndarray:
    """Complementary error function for numpy arrays (relative error < 1.2e-7).

    Numerical Recipes' erfcc; the package must run with numpy alone.
    """
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277)))))))))
    ans = t * np.exp(-z * z + poly)
    return np.where(x >= 0, ans, 2.0 - ans)


class _Shielding:
    """Tabulated short-range part s_AB(R) = 1/R - J_AB(R) (1/Å) for each pair of orbitals.

    J_AB(R) is the Coulomb integral between the two Slater densities; s vanishes
    (below 1e-7) well inside _SHIELD_CUTOFF.
    """

    def __init__(self) -> None:
        self.grid = np.arange(_SHIELD_STEP, _SHIELD_CUTOFF + _SHIELD_STEP / 2, _SHIELD_STEP)
        self._tables: dict = {}

    def _entry(self, a: tuple, b: tuple) -> tuple:
        key = (a, b) if a <= b else (b, a)
        if key not in self._tables:
            (za, na), (zb, nb) = key
            # J is smooth: integrate on a coarse grid, then interpolate onto the fine one
            coarse = np.concatenate([np.linspace(0.0, 2.0, 81),
                                     np.linspace(2.05, _SHIELD_CUTOFF, 160)])
            j = slater_coulomb(za, na, zb, nb, coarse)
            s_coarse = 1.0 / coarse[1:] - j[1:]
            s_fine = np.interp(self.grid, coarse[1:], s_coarse)
            self._tables[key] = (s_fine, float(j[0]))
        return self._tables[key]

    def s(self, a: tuple, b: tuple, r: np.ndarray) -> np.ndarray:
        """s_AB at distances r > 0 (zero beyond the table)."""
        s_fine, _ = self._entry(a, b)
        return np.interp(r, self.grid, s_fine, right=0.0)

    def j0(self, a: tuple, b: tuple) -> float:
        """J_AB(0) (1/Å): two orbitals on top of each other."""
        return self._entry(a, b)[1]


def _cell_widths(basis: np.ndarray) -> np.ndarray:
    vol = abs(np.linalg.det(basis))
    return np.array([vol / np.linalg.norm(np.cross(basis[(i + 1) % 3], basis[(i + 2) % 3]))
                     for i in range(3)])


def _coulomb_matrix(coords: np.ndarray, orbitals: list, pbc=None,
                    eta: Optional[float] = None, progress=None) -> np.ndarray:
    """Shielded Coulomb matrix (1/Å) between the atoms' Slater orbitals.

    Off-diagonal: J_ij summed over all periodic images (Ewald for the 1/R part, plus the
    short-range shielding). Diagonal: only the interaction with the atom's own images
    (zero without a cell); the QEq hardness J_i is added by the caller.
    """
    n = len(coords)
    shield = _Shielding()
    kinds = sorted(set(orbitals))
    kind_of = np.array([kinds.index(o) for o in orbitals])
    A = np.zeros((n, n))
    periodic = pbc is not None
    if periodic:
        eta = 0.35 if eta is None else eta
        r_real = max(4.0 / eta, _SHIELD_CUTOFF)  # erfc(4) < 2e-8 and s ~ 0 beyond
        basis = pbc.basis_matrix
        frac = coords @ np.linalg.inv(basis)
        widths = _cell_widths(basis)
        reach = np.ceil(r_real / widths + 0.5).astype(int)
        shifts = np.mgrid[-reach[0]:reach[0] + 1, -reach[1]:reach[1] + 1,
                          -reach[2]:reach[2] + 1].reshape(3, -1).T.astype(float)
    else:
        eta = 0.0
        r_real = np.inf
        shifts = np.zeros((1, 3))

    # ---- real space: erfc(eta r)/r - s(r) over images (1/r - s(r) without a cell) ----
    block = max(1, 2_000_000 // max(n, 1))
    two_eta_rpi = 2.0 * eta / np.sqrt(np.pi)
    for b0 in range(0, n, block):
        rows = np.arange(b0, min(b0 + block, n))
        if periodic:
            df = frac[None, :, :] - frac[rows][:, None, :]
            df -= np.round(df)
        else:
            dx = coords[None, :, :] - coords[rows][:, None, :]
        for shift in shifts:
            if periodic:
                d = (df + shift) @ basis
            else:
                d = dx
            r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
            self_term = (rows[:, None] == np.arange(n)[None, :]) & np.all(shift == 0)
            near = (r < r_real) & ~self_term
            if not near.any():
                continue
            ii, jj = np.nonzero(near)
            rr = r[ii, jj]
            val = np.empty_like(rr)
            zero = rr < 1e-8  # two sites on top of each other (e.g. MPMC dummy sites)
            pos = ~zero
            coul = 1.0 / rr[pos] if not periodic else _erfc(eta * rr[pos]) / rr[pos]
            val[pos] = coul
            ki, kj = kind_of[rows[ii]], kind_of[jj]
            for ka in range(len(kinds)):
                for kb in range(len(kinds)):
                    m = (ki == ka) & (kj == kb)
                    if not m.any():
                        continue
                    mp = m & pos
                    if mp.any():
                        val[mp] -= shield.s(kinds[ka], kinds[kb], rr[mp])
                    mz = m & zero
                    if mz.any():
                        val[mz] = shield.j0(kinds[ka], kinds[kb]) - two_eta_rpi
            np.add.at(A, (rows[ii], jj), val)
        if progress:
            progress(min(b0 + block, n) / n)

    if not periodic:
        return A

    # ---- reciprocal space and the self term ------------------------------------------
    vol = abs(np.linalg.det(basis))
    recip = 2.0 * np.pi * np.linalg.inv(basis).T  # rows: reciprocal vectors (with 2 pi)
    k_cut = 9.0 * eta  # exp(-k^2 / 4 eta^2) < 2e-9
    m_max = np.ceil(k_cut * np.linalg.norm(basis, axis=1) / (2.0 * np.pi)).astype(int)
    m = np.mgrid[-m_max[0]:m_max[0] + 1, -m_max[1]:m_max[1] + 1,
                 -m_max[2]:m_max[2] + 1].reshape(3, -1).T
    # half space: k and -k contribute equally
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & (m[:, 1] > 0)) | (
        (m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    kvec = m[half] @ recip
    k2 = np.einsum("ij,ij->i", kvec, kvec)
    keep = (k2 > 0) & (k2 < k_cut * k_cut)
    kvec, k2 = kvec[keep], k2[keep]
    weight = 2.0 * (4.0 * np.pi / vol) * np.exp(-k2 / (4.0 * eta * eta)) / k2
    chunk = max(1, 4_000_000 // max(n, 1))
    for c0 in range(0, len(kvec), chunk):
        phase = coords @ kvec[c0:c0 + chunk].T
        w = np.sqrt(weight[c0:c0 + chunk])
        C, S = np.cos(phase) * w, np.sin(phase) * w
        A += C @ C.T + S @ S.T
    A[np.diag_indices(n)] -= two_eta_rpi
    return A


def compute_qeq_charges(
    mol: Molecule,
    total_charge: float = 0.0,
    progress_callback=None,
    eta: Optional[float] = None,
) -> np.ndarray:
    """QEq charges (Rappé & Goddard, J. Phys. Chem. 1991, 95, 3358).

    Minimizes E(q) = sum_i (chi_i q_i + J_i q_i^2 / 2) + sum_{i<j} J_ij q_i q_j subject to
    sum q = total_charge. J_ij is the Coulomb integral between Slater ns orbitals with
    exponents zeta = 0.4913 (2n + 1) / (2 R) (Rappé-Goddard GMP radii R); it is shielded
    at bonding distances and tends to 14.4/R at long range. Periodic cells use an Ewald
    sum over all images. Rappé-Goddard's charge-dependent hydrogen hardness is not
    included.

    Earlier versions used a bare 14.4/R between bonded atoms, which exceeds the atoms'
    own hardness at bond lengths and inverted the signs (water came out O +0.85).

    Args:
        mol: Molecule with atoms (a cell, if any, is used periodically).
        total_charge: Total system charge (0 for neutral).
        progress_callback: Optional callable(frac: float) for progress updates.
        eta: Ewald splitting parameter (1/Å); the charges do not depend on it.

    Returns:
        Array of partial charges (electrons).
    """
    from pdb_wizard.io import is_fallback_cell

    n = len(mol.atoms)
    if n == 0:
        return np.array([])
    if n > 3000:
        raise ValueError(
            f"System too large for QEq ({n} atoms). "
            "Try on a single unit cell before extending."
        )

    chi = np.zeros(n)
    hard = np.zeros(n)
    orbitals = []
    for i, atom in enumerate(mol.atoms):
        sym = atom.element.symbol
        c, j = QEQ_PARAMS.get(sym, _DEFAULT_PARAMS[:2])
        radius = QEQ_RADII.get(sym, _DEFAULT_PARAMS[2])
        n_shell = _principal_n(atom.element.atomic_number or 6)
        chi[i], hard[i] = c, j
        orbitals.append((round(_LAMBDA * (2 * n_shell + 1) / (2.0 * radius), 10), n_shell))

    pbc = mol.pbc if mol.pbc is not None and not is_fallback_cell(mol.pbc) else None
    coords = np.array([a.x for a in mol.atoms], dtype=float)

    if progress_callback:
        progress_callback(0.05)

    def _prog(frac: float) -> None:
        if progress_callback:
            progress_callback(0.05 + 0.85 * frac)

    J = _KE * _coulomb_matrix(coords, orbitals, pbc, eta, _prog)
    J[np.diag_indices(n)] += hard

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = J
    A[n, :n] = A[:n, n] = 1.0
    b = np.concatenate([-chi, [total_charge]])
    try:
        result = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if progress_callback:
        progress_callback(1.0)
    return result[:n]


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
