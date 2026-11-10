"""Tests that widget content is actually rendered visibly.

Earlier `test_widget_sizing.py` checks heights and widths, but a SpinBox can
have plenty of width and still hide its value (text color matches background,
border eats the content area, etc.). This file walks the rendered terminal
output and asserts that every important widget's text appears on screen.

Test strategy:
  1. Push a modal / screen with realistic state.
  2. Render via the compositor and extract per-row text.
  3. For each SpinBox: verify str(value) appears within the SpinBox region.
  4. For each Button: verify the button label appears within its region.
  5. For each Label: verify the label text appears within its region.
  6. For each Input: verify the value-or-placeholder appears.

If a future CSS change hides values (color collision, border eating content,
zero-width inner area, …) these tests fail with a precise message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Input, Label

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    ChargesFileModal,
    CoordinationScreen,
    DensityScreen,
    ExtendAxisModal,
    FileSaveModal,
    ForceFieldModal,
    GyrationScreen,
    HbondScreen,
    InputModal,
    MsdScreen,
    PdbWizardApp,
    PxrdScreen,
    RdfScreen,
    RmsdScreen,
    SorbateModal,
    SpinBox,
)


# ---------------------------------------------------------------------------
# Helpers — render a modal/screen and extract its text per row
# ---------------------------------------------------------------------------

def _mol() -> Molecule:
    atoms = []
    for el in ("C", "O", "N", "H"):
        for i in range(4):
            atoms.append(Atom(float(i + 1), float(el == "O"), 0, el))
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _render_text_grid(app) -> list[str]:
    """Return one string per row of the rendered terminal."""
    comp = app.screen._compositor
    layout = comp.render_full_update()
    rows: list[str] = []
    for strip in layout.strips:
        rows.append("".join(seg.text for seg in strip))
    return rows


def _text_in_region(rows: list[str], region, slack: int = 0) -> str:
    """Concatenate the text inside a widget's region across all its rows.

    `slack` widens the slice horizontally on each side. Useful when a parent
    container has padding/margin that shifts a child's text by ±1 column
    relative to its reported region — we still want to consider that
    'visible' for the user."""
    chunks: list[str] = []
    x0 = max(0, region.x - slack)
    x1 = region.x + region.width + slack
    for y in range(region.y, region.y + region.height):
        if 0 <= y < len(rows):
            chunks.append(rows[y][x0:x1])
    return " ".join(chunks)


def _check_widget_visibility(app, screen, *, label: str) -> None:
    """Walk every Label, Button, SpinBox, Input on the given screen and
    assert the visible text in their region matches their content."""
    rows = _render_text_grid(app)
    failures: list[str] = []

    # --- SpinBoxes: the user-visible value must appear inside the widget ---
    for sb in screen.query(SpinBox):
        expected = str(sb.value)
        text = _text_in_region(rows, sb.region)
        if expected not in text:
            failures.append(
                f"{label}: SpinBox#{sb.id} value {expected!r} not visible in "
                f"region {sb.region} (rendered: {text!r})"
            )

    # --- Buttons: their label text must appear inside the button region ---
    for btn in screen.query(Button):
        btn_label = str(btn.label).strip()
        if not btn_label:
            continue
        # Single-character labels (+, -) are noisy because they appear all
        # over the SpinBox borders; skip those.
        if len(btn_label) <= 1:
            continue
        text = _text_in_region(rows, btn.region)
        if btn_label not in text:
            failures.append(
                f"{label}: Button#{btn.id} label {btn_label!r} not visible "
                f"in region {btn.region} (rendered: {text!r})"
            )

    # --- Standalone Labels (NOT inside SpinBox/Select/etc): visible text ---
    sb_label_ids = {id(child) for sb in screen.query(SpinBox) for child in sb.walk_children()}
    for lab in screen.query(Label):
        if id(lab) in sb_label_ids:
            continue
        # skip empty / decorative labels
        rendered = str(lab.render()).strip()
        if not rendered:
            continue
        # Skip very short labels (single-char like "a:" handled separately)
        if len(rendered) < 2:
            continue
        # Use slack=2 so labels inside a parent with padding still count as
        # visible — parent CSS padding can shift children by ±1 column.
        text = _text_in_region(rows, lab.region, slack=2)
        snippet = rendered[:6]
        if snippet not in text:
            failures.append(
                f"{label}: Label#{lab.id or '<no-id>'} text {rendered!r} "
                f"snippet {snippet!r} not visible in region {lab.region} "
                f"(rendered: {text!r})"
            )

    # --- Inputs: placeholder or value should be visible ---
    for inp in screen.query(Input):
        expected = inp.value or inp.placeholder
        if not expected:
            continue
        # First few chars must appear
        text = _text_in_region(rows, inp.region)
        snippet = expected[:6]
        if snippet not in text:
            failures.append(
                f"{label}: Input#{inp.id} content {expected!r} not visible "
                f"in region {inp.region} (rendered: {text!r})"
            )

    if failures:
        msg = f"Widget visibility failures on {label}:\n  " + "\n  ".join(failures)
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# ExtendAxisModal — the canonical complaint case
# ---------------------------------------------------------------------------

class TestExtendAxisModalVisibility:
    @pytest.mark.asyncio
    async def test_default_values_are_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExtendAxisModal(mol.pbc))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="ExtendAxisModal default")

    @pytest.mark.asyncio
    async def test_max_values_are_visible(self):
        """Set each axis to its max (20). The two-digit values are the
        regression case — earlier the SpinBox was too narrow to show '20'."""
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExtendAxisModal(mol.pbc))
            await pilot.pause()
            for sid in ("extend-na", "extend-nb", "extend-nc"):
                app.screen.query_one(f"#{sid}", SpinBox)._update(20)
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="ExtendAxisModal max(20)")

    @pytest.mark.asyncio
    async def test_axis_labels_a_b_c_visible(self):
        """The 'a:', 'b:', 'c:' axis labels must be readable next to each
        SpinBox so the user knows which is which."""
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExtendAxisModal(mol.pbc))
            await pilot.pause()
            rows = _render_text_grid(app)
            # All three labels should appear somewhere on screen
            joined = "\n".join(rows)
            for axis in ("a:", "b:", "c:"):
                assert axis in joined, f"axis label {axis!r} missing from rendering"


# ---------------------------------------------------------------------------
# Other input-bearing modals
# ---------------------------------------------------------------------------

class TestInputModalVisibility:
    @pytest.mark.asyncio
    async def test_default_value_visible(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(InputModal("Enter a number:", default="42"))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="InputModal")
            rows = _render_text_grid(app)
            assert "42" in "\n".join(rows), "default value must be in input"
            assert "Enter a number" in "\n".join(rows), "prompt must be visible"


class TestFileSaveModalVisibility:
    @pytest.mark.asyncio
    async def test_default_filename_visible(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(FileSaveModal("Save as:", default="my_export.csv"))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="FileSaveModal")
            rows = _render_text_grid(app)
            # The filename should be visible (it's the user's anchor)
            assert "my_export" in "\n".join(rows)


class TestForceFieldModalVisibility:
    @pytest.mark.asyncio
    async def test_buttons_and_title_visible(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ForceFieldModal(_mol()))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="ForceFieldModal")


class TestSorbateModalVisibility:
    @pytest.mark.asyncio
    async def test_select_visible(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(SorbateModal())
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="SorbateModal")


class TestChargesFileModalVisibility:
    @pytest.mark.asyncio
    async def test_input_and_buttons_visible(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ChargesFileModal())
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="ChargesFileModal")


# ---------------------------------------------------------------------------
# Analysis screens — the SpinBox values for cutoffs/bins must be readable
# ---------------------------------------------------------------------------

class TestRdfScreenVisibility:
    @pytest.mark.asyncio
    async def test_all_spinbox_defaults_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RdfScreen(mol))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="RdfScreen")


class TestMsdScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(MsdScreen(mol, [mol, mol]))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="MsdScreen")


class TestCoordinationScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(CoordinationScreen(mol))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="CoordinationScreen")


class TestRmsdScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RmsdScreen(mol, [mol, mol]))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="RmsdScreen")


class TestHbondScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(HbondScreen(mol, [mol]))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="HbondScreen")


class TestGyrationScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(GyrationScreen(mol, [mol, mol]))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="GyrationScreen")


class TestDensityScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(DensityScreen(mol))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="DensityScreen")


class TestPxrdScreenVisibility:
    @pytest.mark.asyncio
    async def test_visible(self):
        mol = _mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(PxrdScreen(mol))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="PxrdScreen")


# ---------------------------------------------------------------------------
# HKUST-1 sanity: real-data run on the modal — catches issues that only
# show up with realistic Select-option widths (e.g. "Cu", "C", "O", "H").
# ---------------------------------------------------------------------------

HKUST = Path(__file__).parent.parent / "HKUST-1.pdb"


@pytest.mark.skipif(not HKUST.exists(), reason="HKUST-1.pdb not available")
class TestHkust1Visibility:
    @pytest.mark.asyncio
    async def test_extend_axis_on_real_mof(self):
        from pdb_wizard.io import read_file
        mol = read_file(str(HKUST))
        app = PdbWizardApp(molecule=mol, filepath=str(HKUST))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExtendAxisModal(mol.pbc))
            await pilot.pause()
            for sid in ("extend-na", "extend-nb", "extend-nc"):
                app.screen.query_one(f"#{sid}", SpinBox)._update(19)  # the max: 19 extra copies
            await pilot.pause()
            rows = _render_text_grid(app)
            joined = "\n".join(rows)
            # Each axis should clearly show "19" in its dedicated row
            assert joined.count("19") >= 3, (
                f"At least one of the 3 axis SpinBoxes is hiding its value: "
                f"only {joined.count('19')} occurrences of '19' in render"
            )

    @pytest.mark.asyncio
    async def test_rdf_screen_on_real_mof(self):
        from pdb_wizard.io import read_file
        mol = read_file(str(HKUST))
        app = PdbWizardApp(molecule=mol, filepath=str(HKUST))
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RdfScreen(mol))
            await pilot.pause()
            _check_widget_visibility(app, app.screen, label="RdfScreen[HKUST]")
