"""QEq against reference values and invariances.

The previous QEq used a bare 14.4/R Coulomb term between bonded atoms, which is larger
than the atoms' own hardness at bond lengths, so the charges came out with inverted signs
(water O +0.85). These tests pin the physics, not just the code's previous output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.charges import (
    QEQ_PARAMS,
    QEQ_RADII,
    _coulomb_matrix,
    apply_qeq_charges,
    compute_qeq_charges,
    slater_coulomb,
)
from pdb_wizard.io import fallback_cell, read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis
from pdb_wizard.pbc import PBC

ROOT = Path(__file__).resolve().parent.parent


def water(box=None):
    atoms = [Atom(5, 5, 5, "O"), Atom(5.9572, 5, 5, "H"), Atom(4.76, 5.9266, 5, "H")]
    return Molecule(atoms=atoms, pbc=box)


def test_slater_1s_coulomb_integral_matches_the_closed_form():
    z = 1.7
    r = np.array([0.0, 0.3, 0.8, 1.5, 3.0, 6.0])
    exact = np.empty_like(r)
    exact[0] = 5 * z / 8
    rr = r[1:]
    exact[1:] = 1 / rr - np.exp(-2 * z * rr) * (1 / rr + 11 * z / 8 + 3 * z * z * rr / 4
                                                + z ** 3 * rr * rr / 6)
    np.testing.assert_allclose(slater_coulomb(z, 1, z, 1, r), exact, atol=1e-9)
    far = slater_coulomb(1.2, 2, 2.0, 3, np.array([12.0]))
    assert far[0] == pytest.approx(1 / 12.0, rel=1e-7)


@pytest.mark.parametrize("eta", [0.3, 0.45])
def test_ewald_gives_the_nacl_madelung_constant(eta):
    a = 5.64
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    coords = np.vstack([base * a, (base + [0.5, 0, 0]) * a % a])
    q = np.array([1.0] * 4 + [-1.0] * 4)
    A = _coulomb_matrix(coords, [(60.0, 1)] * 8, PBC(a, a, a, 90, 90, 90), eta)
    madelung = -(0.5 * q @ A @ q) / 4 * (a / 2)
    assert madelung == pytest.approx(1.7475645946, abs=2e-7)


def test_water_has_the_right_signs():
    q = compute_qeq_charges(water())
    assert q[0] < -0.5 and q[1] > 0.25 and q[2] > 0.25
    assert q.sum() == pytest.approx(0.0, abs=1e-10)
    boxed = compute_qeq_charges(water(PBC(30, 30, 30, 90, 90, 90)))
    np.testing.assert_allclose(boxed, q, atol=2e-3)  # a big box barely matters
    stand_in = compute_qeq_charges(water(fallback_cell()))  # no real cell: non-periodic
    np.testing.assert_allclose(stand_in, q, atol=1e-10)


def test_total_charge_and_coincident_sites():
    mol = water()
    assert compute_qeq_charges(mol, total_charge=-1.0).sum() == pytest.approx(-1.0)
    mol.atoms.append(Atom(5, 5, 5, "H"))  # on top of O, like an MPMC dummy site
    q = compute_qeq_charges(mol)
    assert np.all(np.isfinite(q)) and q.sum() == pytest.approx(0.0, abs=1e-10)


def test_periodic_charges_are_invariant():
    mol = read_file(str(ROOT / "Ni-MOF-74.pqr"))
    q = compute_qeq_charges(mol)
    np.testing.assert_allclose(compute_qeq_charges(mol, eta=0.45), q, atol=5e-6)
    sup = read_file(str(ROOT / "Ni-MOF-74.pqr"))
    extend_axis(sup, 2, 1)
    qs = compute_qeq_charges(sup)
    np.testing.assert_allclose(qs[:len(q)], q, atol=1e-8)
    np.testing.assert_allclose(qs[len(q):], q, atol=1e-8)


def element_means(mol, charges):
    out: dict = {}
    for atom, c in zip(mol.atoms, charges):
        out.setdefault(atom.element.symbol, []).append(c)
    return {k: float(np.mean(v)) for k, v in out.items()}


def test_mof74_signs_follow_its_dft_charges():
    """Ni-MOF-74.pqr carries DFT-derived charges (Ni +1.50, O -0.67..-0.78, H +0.23)."""
    mol = read_file(str(ROOT / "Ni-MOF-74.pqr"))
    dft = element_means(mol, [a.charge for a in mol.atoms])
    qeq = element_means(mol, compute_qeq_charges(mol))
    for el in ("Ni", "O", "H"):
        assert np.sign(qeq[el]) == np.sign(dft[el]), el
    assert 0.2 < qeq["Ni"] < 1.6 and -0.8 < qeq["O"] < -0.2


def test_hkust1_metal_and_carboxylate_signs():
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    apply_qeq_charges(mol)
    means = element_means(mol, [a.charge for a in mol.atoms])
    assert means["Cu"] > 0.3 and means["O"] < -0.3 and means["H"] > 0
    carboxylate_c = [a.charge for a in mol.atoms if a.element.symbol == "C" and a.charge > 0.3]
    assert len(carboxylate_c) == 96  # 3 per BTC, 32 BTC per cell


def test_parameter_tables_cover_the_same_elements():
    assert set(QEQ_RADII) == set(QEQ_PARAMS)
    assert QEQ_RADII["H"] == 0.371 and QEQ_RADII["O"] == 0.669  # Rappe-Goddard GMP radii
