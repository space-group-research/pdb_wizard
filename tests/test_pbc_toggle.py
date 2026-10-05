"""Tests for the on/off semantics of PBC.

PBC is `Optional[PBC]` on Molecule — None means a finite system. The user
must be able to:
  - Open a structure that has no box info (PubChem SDFs, no-CRYST1 PDBs, …)
    without crashes or fake boxes.
  - Add a box via 'Update Unit Cell'.
  - REMOVE the box by typing 'none'/'off'/etc. in the same modal.

These tests pin those three behaviors.
"""

from __future__ import annotations

import tempfile

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import PdbWizardApp


# ---------------------------------------------------------------------------
# Reading a no-PBC structure
# ---------------------------------------------------------------------------

class TestReadingWithoutPBC:
    def test_pdb_without_cryst1_yields_pbc_none(self):
        """A PDB file with atoms but no CRYST1 / REMARK carbasis line must
        load successfully and have mol.pbc is None."""
        pdb = (
            "HEADER    NO BOX\n"
            "ATOM      1  Cu  X A   1       0.000   0.000   0.000  1.00  0.00          Cu\n"
            "ATOM      2   O  X A   1       1.500   0.000   0.000  1.00  0.00           O\n"
            "END\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(pdb)
            path = f.name
        mol = read_file(path)
        assert mol.pbc is None, "no CRYST1 → pbc should be None"
        assert len(mol.atoms) == 2

    def test_no_pbc_system_supports_basic_operations(self):
        """detect_bonds, radius, center must all work without PBC."""
        atoms = [
            Atom(0.0, 0.0, 0.0, "C"),
            Atom(1.5, 0.0, 0.0, "O"),
            Atom(0.0, 1.0, 0.0, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=None)
        # No-pbc bond detection
        mol.detect_bonds()
        assert mol.bonds, "detect_bonds should still produce bonds without PBC"
        # No-pbc radius
        r = mol.radius()
        assert r > 0
        # No-pbc center
        c = mol.center()
        assert c.shape == (3,)


# ---------------------------------------------------------------------------
# update_cell: removal sentinels
# ---------------------------------------------------------------------------

class TestUpdateCellRemovesPBC:
    def _mof_with_pbc(self):
        atoms = [Atom(float(i), 0, 0, "Cu") for i in range(4)]
        return Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sentinel", ["none", "None", "off", "OFF",
                                           "remove", "delete", "no", "0"])
    async def test_sentinel_word_removes_box(self, sentinel):
        mol = self._mof_with_pbc()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.molecule.pbc is not None  # pre-state
            app._do_update_cell(sentinel)
            await pilot.pause()
            await pilot.pause()
            assert app.molecule.pbc is None, (
                f"input {sentinel!r} should have removed PBC; "
                f"still {app.molecule.pbc!r}"
            )

    @pytest.mark.asyncio
    async def test_six_numbers_still_set_box(self):
        """The legacy 6-number flow must still work — adding a box."""
        atoms = [Atom(float(i), 0, 0, "C") for i in range(3)]
        mol = Molecule(atoms=atoms, pbc=None)
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.molecule.pbc is None
            app._do_update_cell("12.0 13.0 14.0 90.0 90.0 90.0")
            await pilot.pause()
            await pilot.pause()
            assert app.molecule.pbc is not None
            assert app.molecule.pbc.a == 12.0
            assert app.molecule.pbc.b == 13.0
            assert app.molecule.pbc.c == 14.0

    @pytest.mark.asyncio
    async def test_remove_when_already_no_pbc_notifies(self):
        """Asking to remove PBC from a system that already has none should
        produce a user-visible notification, not a silent no-op."""
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=None)
        notifications: list[str] = []
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            orig = app.notify

            def capture(msg, *a, **kw):
                notifications.append(str(msg))
                return orig(msg, *a, **kw)
            app.notify = capture

            app._do_update_cell("none")
            await pilot.pause()

            assert any("no PBC" in n.lower() or "already" in n.lower()
                       for n in notifications), (
                f"expected a notice that PBC is already off; got {notifications}"
            )
            # State unchanged
            assert app.molecule.pbc is None

    @pytest.mark.asyncio
    async def test_bad_input_notifies_with_remove_hint(self):
        """A garbled input should explain BOTH the 6-number format AND the
        'none' shortcut — the hint mentions both options."""
        mol = self._mof_with_pbc()
        notifications: list[str] = []
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            orig = app.notify
            app.notify = lambda msg, *a, **kw: (notifications.append(str(msg)),
                                                  orig(msg, *a, **kw))[1]
            app._do_update_cell("a b c d e f")  # not numbers, not 'none'
            await pilot.pause()
            assert any("none" in n.lower() for n in notifications), (
                f"error message should hint that 'none' removes the box; "
                f"got {notifications}"
            )


# ---------------------------------------------------------------------------
# Round-trip add → remove
# ---------------------------------------------------------------------------

class TestPBCRoundTrip:
    @pytest.mark.asyncio
    async def test_add_then_remove(self):
        atoms = [Atom(float(i), 0, 0, "C") for i in range(3)]
        mol = Molecule(atoms=atoms, pbc=None)
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.molecule.pbc is None

            app._do_update_cell("10 10 10 90 90 90")
            await pilot.pause()
            await pilot.pause()
            assert app.molecule.pbc is not None
            assert app.molecule.pbc.a == 10.0

            app._do_update_cell("none")
            await pilot.pause()
            await pilot.pause()
            assert app.molecule.pbc is None
