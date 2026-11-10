"""Tests for supercell extension and the MPMC box-size checks in pdb_wizard.

Run from the repo root:  python -m pytest tests/
Needs numpy, textual, pytest and pytest-asyncio.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "dist" / "pdb_wizard.py"  # built by conftest.py
spec = importlib.util.spec_from_file_location("pdb_wizard_dist", SRC)
pw = importlib.util.module_from_spec(spec)
sys.modules["pdb_wizard_dist"] = pw  # dataclasses look the module up while it loads; not "pdb_wizard", which is the package
spec.loader.exec_module(pw)


def make_mol(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """A neutral two-atom box with LJ set, so only cell warnings can fire."""
    atoms = [pw.Atom(1.0, 1.0, 1.0, "C"), pw.Atom(2.5, 1.0, 1.0, "O")]
    for atom, q in zip(atoms, (0.5, -0.5)):
        atom.charge = q
        atom.epsilon = 50.0
        atom.sigma = 3.4
    return pw.Molecule(atoms=atoms, pbc=pw.PBC(a, b, c, alpha, beta, gamma))


def cell_warnings(mol, **kw):
    _, warnings = pw.validate_for_mpmc(mol, sorbate_pos=(10.0, 10.0, 10.0), **kw)
    return [w for w in warnings if "cutoff" in w or "skewed" in w or "abcbasis" in w]


# ---- PBC geometry --------------------------------------------------------------------

def test_perpendicular_widths_orthorhombic():
    assert np.allclose(pw.PBC(20, 25, 30, 90, 90, 90).perpendicular_widths(), [20, 25, 30])


def test_perpendicular_widths_skewed():
    # gamma = 60°: the a and b faces are 20·sin(60°) apart, c is unchanged.
    w = pw.PBC(20, 20, 20, 90, 90, 60).perpendicular_widths()
    assert np.allclose(w, [20 * math.sin(math.radians(60))] * 2 + [20])


def test_shortest_lattice_vector():
    assert pw.PBC(20, 25, 30, 90, 90, 90).shortest_lattice_vector() == pytest.approx(20)
    # gamma = 30°: a − b is shorter than either vector, 2·20·sin(15°).
    skew = pw.PBC(20, 20, 40, 90, 90, 30).shortest_lattice_vector()
    assert skew == pytest.approx(2 * 20 * math.sin(math.radians(15)))


# ---- validate_for_mpmc box checks -----------------------------------------------------

def test_big_orthorhombic_box_has_no_cell_warnings():
    assert cell_warnings(make_mol(25, 25, 25)) == []


def test_small_box_warns_with_the_limit():
    (w,) = cell_warnings(make_mol(15, 25, 25))
    assert "too thin" in w and "7.50" in w and "Extend Axis" in w


def test_skewed_cell_uses_perpendicular_width_not_lengths():
    # a = b = c = 30 Å passes the old min(a,b,c)/2 check, but with gamma = 30°
    # the a and b faces are only 15 Å apart, so the limit is 7.5 Å.
    (w,) = cell_warnings(make_mol(30, 30, 30, gamma=30))
    assert "too thin" in w and "7.50" in w


def test_hexagonal_cell_warns_about_mpmc_auto_cutoff():
    # Wide enough for 10 Å, but MPMC's automatic cutoff (15 Å) is beyond the
    # 12.99 Å minimum-image limit of a gamma = 120° cell.
    (w,) = cell_warnings(make_mol(30, 30, 30, gamma=120))
    assert "skewed" in w and "15.00" in w and "12.99" in w


def test_mildly_triclinic_cell_is_fine():
    # abcbasis carries the angles, so a big triclinic cell needs no warning.
    assert cell_warnings(make_mol(30, 30, 30, alpha=80, beta=85, gamma=95)) == []


def test_custom_cutoff():
    assert cell_warnings(make_mol(22, 22, 22), lj_cutoff=12.5)
    assert cell_warnings(make_mol(22, 22, 22), lj_cutoff=10.0) == []


# ---- extend_axis and the TUI dialog ------------------------------------------------------

def test_extend_axis_counts_extra_copies():
    mol = make_mol(10, 10, 10)
    pw.extend_axis(mol, 0, 1)  # classic "extend 1 time" doubles the axis
    assert len(mol.atoms) == 4 and mol.pbc.a == pytest.approx(20)


@pytest.mark.asyncio
async def test_tui_extend_dialog_counts_like_classic():
    mol = make_mol(10, 10, 10)
    app = pw.PdbWizardApp(mol)
    async with app.run_test() as pilot:
        modal = pw.ExtendAxisModal(mol.pbc)
        app.push_screen(modal, callback=app._do_extend)
        await pilot.pause()
        spins = [modal.query_one(f"#extend-n{x}", pw.SpinBox) for x in "abc"]
        assert [s.value for s in spins] == [0, 0, 0]
        spins[0]._update(1)  # extend a once, b twice, c once
        spins[1]._update(2)
        spins[2]._update(1)
        await pilot.click("#extend-ok")
        for _ in range(50):
            await pilot.pause(0.05)
            if len(app.molecule.atoms) == 24:
                break
        assert len(app.molecule.atoms) == 2 * 2 * 3 * 2
        assert (app.molecule.pbc.a, app.molecule.pbc.b, app.molecule.pbc.c) == \
            pytest.approx((20, 30, 20))


@pytest.mark.asyncio
async def test_tui_extend_dialog_all_zero_is_a_no_op():
    mol = make_mol(10, 10, 10)
    app = pw.PdbWizardApp(mol)
    async with app.run_test() as pilot:
        app._do_extend((0, 0, 0))
        app._do_extend(())
        await pilot.pause()
        assert len(app.molecule.atoms) == 2
