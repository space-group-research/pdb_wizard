"""Tests verifying HbondScreen, GyrationScreen, and DensityScreen inherit
from _AnalysisScreen and use its shared methods correctly.

Covers: inheritance, _prefix, _get_frame_slice, _yield_frame_range,
_update_name_select, _do_export_csv, and inherited BINDINGS/action_close.
"""

import os
import tempfile

import numpy as np
import pytest
from textual.widgets import Checkbox, Select

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    DensityScreen,
    GyrationScreen,
    HbondScreen,
    PdbWizardApp,
    SpinBox,
    _AnalysisScreen,
)

TERM = (120, 40)


def _mol(n=4):
    """Create a small test molecule with mixed elements."""
    atoms = [
        Atom(0.0, 0.0, 0.0, "O"),
        Atom(1.0, 0.0, 0.0, "H"),
        Atom(2.0, 0.0, 0.0, "H"),
        Atom(3.0, 0.0, 0.0, "C"),
    ][:n]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _app():
    return PdbWizardApp(molecule=_mol(), filepath="test.pdb")


async def _push(app, pilot, screen):
    app.push_screen(screen)
    await pilot.pause()
    return app.screen


# ===========================================================================
# Inheritance checks
# ===========================================================================

class TestInheritance:
    def test_hbond_inherits(self):
        assert issubclass(HbondScreen, _AnalysisScreen)

    def test_gyration_inherits(self):
        assert issubclass(GyrationScreen, _AnalysisScreen)

    def test_density_inherits(self):
        assert issubclass(DensityScreen, _AnalysisScreen)


class TestPrefix:
    def test_hbond_prefix(self):
        assert HbondScreen._prefix == "hb"

    def test_gyration_prefix(self):
        assert GyrationScreen._prefix == "rg"

    def test_density_prefix(self):
        assert DensityScreen._prefix == "dens"


class TestBindingsInherited:
    """BINDINGS should come from _AnalysisScreen, not be redefined."""

    def test_hbond_bindings(self):
        # Should not have its own BINDINGS attribute on the class dict
        assert "BINDINGS" not in HbondScreen.__dict__

    def test_gyration_bindings(self):
        assert "BINDINGS" not in GyrationScreen.__dict__

    def test_density_bindings(self):
        assert "BINDINGS" not in DensityScreen.__dict__

    def test_all_have_escape_binding(self):
        for cls in (HbondScreen, GyrationScreen, DensityScreen):
            keys = [b.key for b in cls.BINDINGS]
            assert "escape" in keys
            assert "q" in keys


class TestActionCloseInherited:
    """action_close should not be redefined in the subclasses."""

    def test_hbond_no_own_action_close(self):
        assert "action_close" not in HbondScreen.__dict__

    def test_gyration_no_own_action_close(self):
        assert "action_close" not in GyrationScreen.__dict__

    def test_density_no_own_action_close(self):
        assert "action_close" not in DensityScreen.__dict__


# ===========================================================================
# __init__ sets base attributes
# ===========================================================================

class TestInitSetsBaseAttributes:
    def test_hbond_init(self):
        m = _mol()
        s = HbondScreen(m, [m, m])
        assert s._molecule is m
        assert s._frames == [m, m]
        assert isinstance(s._elements, list)
        assert "O" in s._elements

    def test_gyration_init(self):
        m = _mol()
        s = GyrationScreen(m, [m, m, m])
        assert s._molecule is m
        assert len(s._frames) == 3
        assert "C" in s._elements

    def test_density_init(self):
        m = _mol()
        s = DensityScreen(m, [m])
        assert s._molecule is m
        assert s._elements == sorted(set(a.element.symbol for a in m.atoms))


# ===========================================================================
# _yield_frame_range produces SpinBoxes when frames > 1
# ===========================================================================

class TestYieldFrameRange:
    @pytest.mark.asyncio
    async def test_hbond_frame_spinboxes(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, HbondScreen(m, [m, m, m]))
            start = s.query_one("#hb-start", SpinBox)
            end = s.query_one("#hb-end", SpinBox)
            assert start.value == 1
            assert end.value == 3

    @pytest.mark.asyncio
    async def test_gyration_frame_spinboxes(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, GyrationScreen(m, [m, m, m, m]))
            start = s.query_one("#rg-start", SpinBox)
            end = s.query_one("#rg-end", SpinBox)
            assert start.value == 1
            assert end.value == 4

    @pytest.mark.asyncio
    async def test_density_frame_spinboxes(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m, m]))
            start = s.query_one("#dens-start", SpinBox)
            end = s.query_one("#dens-end", SpinBox)
            assert start.value == 1
            assert end.value == 2

    @pytest.mark.asyncio
    async def test_no_frame_spinboxes_single_frame(self):
        """With only one frame, no start/end SpinBoxes should be created."""
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, HbondScreen(m, [m]))
            spinboxes = s.query(SpinBox)
            ids = {sb.id for sb in spinboxes}
            assert "hb-start" not in ids
            assert "hb-end" not in ids


# ===========================================================================
# _get_frame_slice
# ===========================================================================

class TestGetFrameSlice:
    @pytest.mark.asyncio
    async def test_gyration_frame_slice(self):
        m = _mol()
        frames = [m, m, m, m, m]
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, GyrationScreen(m, frames))
            # Default: start=1, end=5 => full slice
            result = s._get_frame_slice()
            assert len(result) == 5

    @pytest.mark.asyncio
    async def test_gyration_frame_slice_subset(self):
        m = _mol()
        frames = [m, m, m, m, m]
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, GyrationScreen(m, frames))
            s.query_one("#rg-start", SpinBox)._update(2)
            s.query_one("#rg-end", SpinBox)._update(4)
            result = s._get_frame_slice()
            assert len(result) == 3  # frames[1:4]


# ===========================================================================
# _update_name_select (DensityScreen uses it)
# ===========================================================================

class TestUpdateNameSelect:
    @pytest.mark.asyncio
    async def test_density_name_select_updates(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m]))
            # Simulate element change to "O"
            s._update_name_select("O", "#dens-name")
            sel = s.query_one("#dens-name", Select)
            assert sel.value == "(all)"


# ===========================================================================
# _do_export_csv
# ===========================================================================

class TestDoExportCsv:
    @pytest.mark.asyncio
    async def test_hbond_export_csv(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, HbondScreen(m, [m, m]))
            s._last_idx = np.array([0.0, 1.0, 2.0])
            s._last_counts = np.array([5.0, 3.0, 7.0])
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                tmppath = f.name
            try:
                s._export_csv(tmppath)
                with open(tmppath) as f:
                    lines = f.readlines()
                assert lines[0].strip() == "frame,hbond_count"
                assert len(lines) == 4  # header + 3 data rows
                # Values should be .6f formatted
                assert "0.000000" in lines[1]
            finally:
                os.unlink(tmppath)

    @pytest.mark.asyncio
    async def test_gyration_export_csv(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, GyrationScreen(m, [m, m]))
            s._last_idx = np.array([0.0, 1.0])
            s._last_rg = np.array([2.5, 3.1])
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                tmppath = f.name
            try:
                s._export_csv(tmppath)
                with open(tmppath) as f:
                    lines = f.readlines()
                assert lines[0].strip() == "frame,rg_A"
                assert len(lines) == 3
                assert "2.500000" in lines[1]
            finally:
                os.unlink(tmppath)

    @pytest.mark.asyncio
    async def test_density_export_csv(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m]))
            s._last_pos = np.array([1.0, 2.0])
            s._last_dens = np.array([0.5, 0.8])
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                tmppath = f.name
            try:
                s._export_csv(tmppath)
                with open(tmppath) as f:
                    lines = f.readlines()
                assert lines[0].strip() == "position_A,density"
                assert len(lines) == 3
            finally:
                os.unlink(tmppath)

    @pytest.mark.asyncio
    async def test_export_csv_empty_path_noop(self):
        """Empty path should not create a file."""
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, HbondScreen(m, [m, m]))
            s._last_idx = np.array([0.0])
            s._last_counts = np.array([1.0])
            # Should not raise
            s._export_csv("   ")


# ===========================================================================
# CSS uses _ANALYSIS_CSS template
# ===========================================================================

class TestCssTemplate:
    def test_hbond_css_has_class_rules(self):
        css = HbondScreen.DEFAULT_CSS
        assert "HbondScreen > Vertical" in css
        assert "HbondScreen Select" in css
        assert "HbondScreen SpinBox" in css
        assert "HbondScreen Button" in css
        assert "HbondScreen Label" in css

    def test_gyration_css_has_class_rules(self):
        css = GyrationScreen.DEFAULT_CSS
        assert "GyrationScreen > Vertical" in css
        assert "GyrationScreen Select" in css

    def test_density_css_has_class_rules(self):
        css = DensityScreen.DEFAULT_CSS
        assert "DensityScreen > Vertical" in css
        assert "DensityScreen Checkbox" in css

    def test_hbond_css_has_specific_rules(self):
        css = HbondScreen.DEFAULT_CSS
        assert "#hb-controls" in css
        assert "#hb-status" in css

    def test_density_css_has_specific_rules(self):
        css = DensityScreen.DEFAULT_CSS
        assert "#dens-controls" in css
        assert "#dens-status" in css


# ===========================================================================
# DensityScreen traj checkbox (still present)
# ===========================================================================

class TestDensityTrajCheckbox:
    @pytest.mark.asyncio
    async def test_traj_checkbox_present(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m, m]))
            cb = s.query_one("#dens-traj", Checkbox)
            assert cb.value is True

    @pytest.mark.asyncio
    async def test_no_traj_checkbox_single_frame(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m]))
            checkboxes = s.query(Checkbox)
            ids = {cb.id for cb in checkboxes}
            assert "dens-traj" not in ids


# ===========================================================================
# Escape binding works (inherited action_close)
# ===========================================================================

class TestEscapeCloses:
    @pytest.mark.asyncio
    async def test_hbond_escape(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            await _push(app, pilot, HbondScreen(m, [m, m]))
            assert isinstance(app.screen, HbondScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HbondScreen)

    @pytest.mark.asyncio
    async def test_gyration_q_closes(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            await _push(app, pilot, GyrationScreen(m, [m, m]))
            assert isinstance(app.screen, GyrationScreen)
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, GyrationScreen)
