"""Tests for charge-file loading (CP2K RESP auto-detection + raw columns).

Regression: the TUI charge loader used to require a plain one-column file and
would choke on a CP2K .resp file (header lines + a trailing "Total charge"
line). The legacy CLI auto-detected .resp and stripped those; this restores
that behavior through a shared io.read_charges_file helper and exercises it
through the real UI modal flow as well.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.io import parse_resp_charges, read_charges_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

FIXTURES = Path(__file__).parent / "fixtures"
RESP_FILE = FIXTURES / "water_dimer.resp"

# Charges in water_dimer.resp, in file order (the "Total" line is excluded).
EXPECTED_RESP = [-0.831024, 0.415512, 0.415512, -0.831024, 0.415512, 0.415512]


# ---------------------------------------------------------------------------
# parse_resp_charges (pure)
# ---------------------------------------------------------------------------

class TestParseResp:
    def test_parses_real_cp2k_format(self):
        charges = parse_resp_charges(RESP_FILE.read_text().splitlines())
        assert charges == pytest.approx(EXPECTED_RESP)

    def test_skips_total_line_even_though_last_token_is_float(self):
        lines = [
            "RESP charges:",
            "  Type | Atom | Charge",
            "   1   1   -0.5",
            "   1   2    0.5",
            "Total charge of the system:   0.0000000",
        ]
        # The Total line's last token (0.0) must NOT be read as an atom charge.
        assert parse_resp_charges(lines) == pytest.approx([-0.5, 0.5])

    def test_skips_blank_and_header_lines(self):
        lines = ["RESP charges:", "", "  ignore me header", "  1 C  0.123", ""]
        assert parse_resp_charges(lines) == pytest.approx([0.123])

    def test_empty_input(self):
        assert parse_resp_charges([]) == []


# ---------------------------------------------------------------------------
# read_charges_file (dispatch on extension)
# ---------------------------------------------------------------------------

class TestReadChargesFile:
    def test_resp_auto_detected(self):
        assert read_charges_file(str(RESP_FILE)) == pytest.approx(EXPECTED_RESP)

    def test_resp_extension_case_insensitive(self, tmp_path):
        p = tmp_path / "charges.RESP"
        p.write_text(RESP_FILE.read_text())
        assert read_charges_file(str(p)) == pytest.approx(EXPECTED_RESP)

    def test_raw_single_column(self, tmp_path):
        p = tmp_path / "q.dat"
        p.write_text("-0.5\n0.25\n0.25\n")
        assert read_charges_file(str(p)) == pytest.approx([-0.5, 0.25, 0.25])

    def test_raw_multi_column_takes_last_token(self, tmp_path):
        p = tmp_path / "q.txt"
        p.write_text("C 1 -0.5\nH 2 0.25\nH 3 0.25\n")
        assert read_charges_file(str(p)) == pytest.approx([-0.5, 0.25, 0.25])

    def test_raw_skip_first_and_last(self, tmp_path):
        p = tmp_path / "q.txt"
        p.write_text("HEADER\n-0.5\n0.25\n0.25\nFOOTER\n")
        out = read_charges_file(str(p), skip_first=1, skip_last=1)
        assert out == pytest.approx([-0.5, 0.25, 0.25])

    def test_raw_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "q.dat"
        p.write_text("-0.5\n\n0.5\n")
        assert read_charges_file(str(p)) == pytest.approx([-0.5, 0.5])

    def test_non_numeric_raw_line_raises(self, tmp_path):
        p = tmp_path / "q.dat"
        p.write_text("-0.5\nnot_a_number\n")
        with pytest.raises(ValueError):
            read_charges_file(str(p))


# ---------------------------------------------------------------------------
# End-to-end through the real UI modal + apply path
# ---------------------------------------------------------------------------

def _dimer() -> Molecule:
    """Two waters, 6 atoms — matches water_dimer.resp."""
    atoms = [
        Atom(0.0, 0.0, 0.0, "O"),
        Atom(0.96, 0.0, 0.0, "H"),
        Atom(-0.24, 0.93, 0.0, "H"),
        Atom(3.0, 0.0, 0.0, "O"),
        Atom(3.96, 0.0, 0.0, "H"),
        Atom(2.76, 0.93, 0.0, "H"),
    ]
    mol = Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))
    mol.detect_bonds()
    return mol


@pytest.mark.asyncio
async def test_resp_round_trips_through_modal_and_applies_charges():
    """A user picking a .resp file in ChargesFileModal must get the charges
    auto-parsed (preview + applied to atoms), with skip controls ignored."""
    from textual.widgets import TextArea

    from pdb_wizard.viewer import ChargesFileModal, PdbWizardApp

    app = PdbWizardApp(molecule=_dimer(), filepath="dimer.pdb")
    async with app.run_test(size=(140, 45)) as pilot:
        modal = ChargesFileModal()
        app.push_screen(modal)
        await pilot.pause()

        # Simulate the user selecting the .resp file in the directory tree.
        modal._file_path = str(RESP_FILE)
        modal._raw_lines = RESP_FILE.read_text().splitlines(keepends=True)
        modal._update_preview()
        await pilot.pause()

        preview = modal.query_one("#charges-preview", TextArea).text
        assert "auto-parsed" in preview
        # All six charges should appear; the Total line should not leak in.
        assert "-0.831024" in preview
        assert preview.count("\n") >= 6

        # Submit and feed the result to the apply handler (as the callback does).
        # Stub the downstream force-field step so this test stays focused on the
        # charge-apply path (the FF modal is exercised by its own tests).
        app._mpmc_step_ask_ff = lambda: None
        modal._submit()
        await pilot.pause()
        app._mpmc_step_charges_file({"path": str(RESP_FILE)})

        applied = [a.charge for a in app.molecule.atoms]
        assert applied == pytest.approx(EXPECTED_RESP)


# CP2K's RESP_CHARGES_TO_FILE layout with a kind column and a short Total line.
CP2K_RESP = """ RESP charges:
  Type |   Atom |    Charge

  RESP       1  Cu    1.064519
  RESP       2  O    -0.582110
  RESP       3  O    -0.482409
 Total                 0.000000
"""


def test_resp_is_recognised_by_content_whatever_the_file_is_called(tmp_path):
    """Students rename these files; the header, not the extension, says what it is."""
    for name in ("charges.txt", "HKUST-RESP_CHARGES.resp", "resp_charges"):
        path = tmp_path / name
        path.write_text(CP2K_RESP)
        assert read_charges_file(str(path)) == [1.064519, -0.582110, -0.482409], name
    renamed = tmp_path / "dimer.dat"
    renamed.write_text(RESP_FILE.read_text())
    assert read_charges_file(str(renamed)) == EXPECTED_RESP


def test_plain_column_files_are_still_read_as_plain_columns(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("0.5\n-0.25\n-0.25\n")
    assert read_charges_file(str(path)) == [0.5, -0.25, -0.25]
    labelled = tmp_path / "labelled.txt"
    labelled.write_text("Cu 0.5\nO -0.5\n")
    assert read_charges_file(str(labelled)) == [0.5, -0.5]
