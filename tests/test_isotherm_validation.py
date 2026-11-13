"""Tests for the Isotherm Planner's pre-generation validation.

Real-world failure mode: user opens a fresh CIF, never applies a force field
or QEq charges, clicks Generate. MPMC runs with all-zero electrostatics and
all-zero LJ — uptake is meaningless. The validator catches this BEFORE files
get written.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Label, TextArea

from pdb_wizard.atom import Atom
from pdb_wizard.isotherm import (
    _DEFAULT_LJ_CUTOFF,
    generate_isotherm,
    validate_for_mpmc,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import IsothermPlannerScreen, PdbWizardApp


def _well_charged_mof() -> Molecule:
    """A small periodic 'MOF' with non-zero charges and LJ params."""
    atoms = []
    for i, el in enumerate(["Cu", "C", "C", "O", "O", "H"]):
        a = Atom(float(i) * 1.5, 0.0, 0.0, el)
        # Net-neutral: +0.5 + 0.1 + 0.1 - 0.3 - 0.3 - 0.1 = 0
        a.charge = {"Cu": 0.5, "C": 0.1, "O": -0.3, "H": -0.1}[el]
        a.epsilon = 30.0  # Kelvin
        a.sigma = 3.5     # Å
        atoms.append(a)
    return Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))


def _zero_charge_mof() -> Molecule:
    """Charges all zero (the 'fresh CIF' state)."""
    mol = _well_charged_mof()
    for a in mol.atoms:
        a.charge = 0.0
    return mol


def _zero_lj_mof() -> Molecule:
    """LJ params all zero."""
    mol = _well_charged_mof()
    for a in mol.atoms:
        a.epsilon = 0.0
        a.sigma = 0.0
    return mol


# ---------------------------------------------------------------------------
# Pure validator
# ---------------------------------------------------------------------------

class TestValidatorErrors:
    def test_no_pbc_is_an_error(self):
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=None)
        errors, warnings = validate_for_mpmc(mol)
        assert errors and any("PBC" in e for e in errors)

    def test_all_zero_charges_blocks_generation(self):
        errors, warnings = validate_for_mpmc(_zero_charge_mof())
        assert errors, "all-zero charges must produce an error"
        assert any("charges are zero" in e.lower()
                   or "all atom charges are zero" in e.lower()
                   for e in errors), errors

    def test_all_zero_lj_blocks_generation(self):
        errors, warnings = validate_for_mpmc(_zero_lj_mof())
        assert errors, "all-zero LJ epsilons must produce an error"
        assert any("epsilon" in e.lower() and "zero" in e.lower() for e in errors), errors

    def test_well_set_mof_has_no_errors(self):
        errors, warnings = validate_for_mpmc(_well_charged_mof())
        assert not errors, f"clean MOF should have no errors; got {errors}"


class TestValidatorWarnings:
    def test_net_charge_nonzero_warns(self):
        mol = _well_charged_mof()
        # Add a single +0.5 e atom — net no longer zero
        mol.atoms[0].charge += 0.5
        errors, warnings = validate_for_mpmc(mol)
        assert not errors
        assert any("net" in w.lower() and "charge" in w.lower() for w in warnings), warnings

    def test_undersized_box_warns(self):
        mol = _well_charged_mof()
        # Box < 2 * lj_cutoff — too small
        mol.pbc = PBC(15, 15, 15, 90, 90, 90)
        errors, warnings = validate_for_mpmc(mol, lj_cutoff=_DEFAULT_LJ_CUTOFF)
        assert any("box" in w.lower() or "cutoff" in w.lower() for w in warnings), warnings

    def test_triclinic_cell_does_not_warn(self):
        """'abcbasis a b c α β γ' builds the same basis as PBC, so a skewed but
        roomy cell needs no warning (space-group-environment PR #8)."""
        mol = _well_charged_mof()
        mol.pbc = PBC(30, 30, 30, 90.0, 100.0, 90.0)  # β skewed, wide enough for the cutoff
        errors, warnings = validate_for_mpmc(mol)
        assert not any("orthorhombic" in w.lower() or "abcbasis" in w.lower()
                       for w in warnings), warnings

    def test_sorbate_overlap_warns(self):
        mol = _well_charged_mof()
        # Place sorbate AT the first atom's position → overlap guaranteed
        first = mol.atoms[0].x
        errors, warnings = validate_for_mpmc(
            mol, sorbate_pos=(float(first[0]), float(first[1]), float(first[2])),
        )
        assert any("overlap" in w.lower() for w in warnings), warnings

    def test_well_placed_sorbate_no_overlap_warning(self):
        mol = _well_charged_mof()
        # Place far from atoms (atoms are along x at y=z=0)
        errors, warnings = validate_for_mpmc(mol, sorbate_pos=(15.0, 15.0, 15.0))
        assert not any("overlap" in w.lower() for w in warnings), warnings


# ---------------------------------------------------------------------------
# UI: errors block Generate; warnings let it proceed
# ---------------------------------------------------------------------------

class TestIsothermUiBlocking:
    @pytest.mark.asyncio
    async def test_zero_charges_blocks_generate_button(self, tmp_path):
        """Pressing Generate on a zero-charge MOF must NOT write any files."""
        mol = _zero_charge_mof()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            screen = IsothermPlannerScreen(mol, "t.pdb")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()
            # Set the output dir to a clean tmp_path; nothing should be written there
            from textual.widgets import Input
            screen.query_one("#gen-outdir", Input).value = str(tmp_path / "iso")
            await pilot.pause()

            await pilot.click("#gen-generate")
            await pilot.pause()
            await pilot.pause()

            # Status reports refusal
            status = str(screen.query_one("#gen-status", Label).render())
            assert "Refusing" in status or "error" in status.lower(), status

            # Most importantly: NO directory was created
            assert not (tmp_path / "iso").exists(), (
                "Generate must not create files when validation fails"
            )

    @pytest.mark.asyncio
    async def test_validation_runs_on_open(self):
        """Opening the planner pre-fills the preview with the validation
        result so the user sees errors immediately, not after pressing
        Generate."""
        mol = _zero_charge_mof()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            screen = IsothermPlannerScreen(mol, "t.pdb")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()

            preview = screen.query_one("#gen-preview", TextArea)
            text = preview.text
            # Should mention 'error' or '✗' since charges are zero
            assert "✗" in text or "error" in text.lower(), text
            assert "charge" in text.lower(), text

    @pytest.mark.asyncio
    async def test_clean_mof_can_generate(self, tmp_path):
        """A MOF with charges + LJ + neutral net charge should generate
        without complaints."""
        mol = _well_charged_mof()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            screen = IsothermPlannerScreen(mol, "t.pdb")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()

            from textual.widgets import Input
            outdir = tmp_path / "iso_out"
            screen.query_one("#gen-outdir", Input).value = str(outdir)
            # Use a single pressure to keep it quick
            screen.query_one("#gen-pressures", Input).value = "1.0"
            await pilot.pause()

            await pilot.click("#gen-generate")
            await pilot.pause()
            await pilot.pause()

            status = str(screen.query_one("#gen-status", Label).render())
            assert "Generated" in status, status
            # One subdirectory and a master script
            assert outdir.exists()
            sub = list(outdir.iterdir())
            assert any(p.name.startswith("P_") and p.is_dir() for p in sub), sub
            assert (outdir / "run_all.sh").exists()


# ---------------------------------------------------------------------------
# generate_isotherm pure-API check (no UI)
# ---------------------------------------------------------------------------

class TestGenerateIsothermFiles:
    def test_creates_one_dir_per_pressure_with_expected_files(self, tmp_path):
        mol = _well_charged_mof()
        dirs = generate_isotherm(
            mol, "h2_bss", 77.0, [0.5, 1.0, 2.0], str(tmp_path / "iso"),
            numsteps=10000,
        )
        assert len(dirs) == 3
        for d in dirs:
            p = Path(d)
            assert (p / "input.pqr").exists()
            assert (p / "mpmc.inp").exists()
            assert (p / "insert.pqr").exists()
            assert (p / "run.sh").exists()
        assert (tmp_path / "iso" / "run_all.sh").exists()

    def test_mpmc_inp_records_temperature_and_pressure(self, tmp_path):
        mol = _well_charged_mof()
        dirs = generate_isotherm(
            mol, "h2_bss", 123.0, [4.5], str(tmp_path / "iso"),
            numsteps=10000,
        )
        inp = (Path(dirs[0]) / "mpmc.inp").read_text()
        assert "temperature           123.0" in inp, inp
        assert "pressure              4.5" in inp, inp
