"""PXRD peak positions against crystallography.

The reciprocal vectors were taken as the rows of inv(basis) instead of its columns, which
moved every peak of a non-orthogonal cell (Ni-MOF-74's d(100) came out 25.79 Å instead of
22.33 Å). Cubic and orthorhombic cells were unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.pxrd import compute_pxrd, pxrd_reflections

ROOT = Path(__file__).resolve().parent.parent
CU_KA = 1.5406


def two_theta_from_metric(pbc, hkl):
    """2-theta (deg) from the metric tensor: 1/d^2 = h G* h^T, G* = inv(B B^T)."""
    g_star = np.linalg.inv(pbc.basis_matrix @ pbc.basis_matrix.T)
    d = 1.0 / np.sqrt(np.einsum("ij,jk,ik->i", hkl, g_star, hkl))
    return 2 * np.degrees(np.arcsin(CU_KA / (2 * d)))


def test_every_reflection_of_a_triclinic_cell_is_where_bragg_puts_it():
    pbc = PBC(9.1, 10.7, 12.3, 77.0, 84.0, 101.0)
    rng = np.random.default_rng(1)
    atoms = [Atom(*(f @ pbc.basis_matrix), el)
             for f, el in zip(rng.random((12, 3)), ["C", "O", "N", "Cu"] * 3)]
    tt, inten, hkl = pxrd_reflections(Molecule(atoms=atoms, pbc=pbc), CU_KA, 5, 40,
                                      with_hkl=True)
    assert len(tt) > 100
    np.testing.assert_allclose(tt, two_theta_from_metric(pbc, hkl.astype(float)), atol=1e-8)


def test_mof74_strongest_peak_is_110_at_6_85():
    mol = read_file(str(ROOT / "Ni-MOF-74.pqr"))  # hexagonal, a = 25.786 Å
    tt, inten, hkl = pxrd_reflections(mol, CU_KA, 2, 15, with_hkl=True)
    strongest = hkl[np.argmax(inten)]
    assert sorted(np.abs(strongest[:2])) in ([1, 1], [1, 2]) and strongest[2] == 0
    assert tt[np.argmax(inten)] == pytest.approx(6.85, abs=0.01)
    tt100 = two_theta_from_metric(mol.pbc, np.array([[1.0, 0, 0]]))[0]
    d100 = CU_KA / (2 * np.sin(np.radians(tt100 / 2)))
    assert d100 == pytest.approx(mol.pbc.a * np.sqrt(3) / 2, rel=1e-6)  # 22.33 Å


def test_cubic_peaks_follow_a_over_root_hkl():
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    tt, _, hkl = pxrd_reflections(mol, CU_KA, 5, 20, with_hkl=True)
    d = mol.pbc.a / np.sqrt((hkl.astype(float) ** 2).sum(axis=1))
    np.testing.assert_allclose(tt, 2 * np.degrees(np.arcsin(CU_KA / (2 * d))), atol=1e-8)
    x, y = compute_pxrd(mol)
    assert x[np.argmax(y)] == pytest.approx(6.71, abs=0.03)  # HKUST-1's (200)
