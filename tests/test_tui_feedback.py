"""TUI feedback regressions: the viewer must never report success for a no-op,
lose a background failure, write an empty file, or mutate without an undo entry.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Label

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.panels import OperationsPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import PdbWizardApp, RdfScreen

W, H = 120, 40


def _mol(pbc=True, extra=()):
    atoms = [Atom(0, 0, 0, "O"), Atom(0.757, 0.586, 0, "H"), Atom(-0.757, 0.586, 0, "H")]
    atoms += list(extra)
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90) if pbc else None)
    mol.detect_bonds()
    return mol


def _app(mol=None, **kw):
    return PdbWizardApp(molecule=mol if mol is not None else _mol(), filepath="t.pdb", **kw)


def _toasts(app) -> list[str]:
    return [str(n.message) for n in app._notifications]


async def _settle(app, pilot):
    """Wait for every tracked background task, then let the UI catch up."""
    for _ in range(3):
        pending = [t for t in app._bg_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await pilot.pause()


def _run(app, cmd: str) -> None:
    app.on_operations_panel_run_command(OperationsPanel.RunCommand(cmd))


def _label_text(label) -> str:
    return str(label._Static__content)


# ---------------------------------------------------------------------------
# background task failures are visible
# ---------------------------------------------------------------------------

class TestTaskFailures:
    @pytest.mark.asyncio
    async def test_tracked_task_failure_toasts_and_clears_progress(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()

            async def boom():
                app._show_progress("Doing something...")
                await asyncio.sleep(0)
                raise RuntimeError("kaboom")

            app._track_task(boom())
            await _settle(app, pilot)
            assert any("boom failed: kaboom" in t for t in _toasts(app)), _toasts(app)
            assert not app.query_one("#status-bar").has_class("busy")

    @pytest.mark.asyncio
    async def test_cancelled_tracked_task_is_silent(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()

            async def slow():
                await asyncio.sleep(60)

            t = app._track_task(slow())
            t.cancel()
            await _settle(app, pilot)
            assert not any("failed" in m for m in _toasts(app))

    @pytest.mark.asyncio
    async def test_analysis_screen_failure_updates_status(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            screen = RdfScreen(app.molecule)
            app.push_screen(screen)
            await pilot.pause()

            async def boom():
                raise ValueError("bad bins")

            screen._spawn_task(boom())
            await asyncio.sleep(0)
            await pilot.pause()
            assert "Error: bad bins" in _label_text(screen.query_one("#rdf-status", Label))
            assert any("boom failed: bad bins" in t for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_csv_export_to_bad_path_does_not_crash(self, tmp_path):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            screen = RdfScreen(app.molecule)
            app.push_screen(screen)
            await pilot.pause()
            bad = tmp_path / "no_such_dir" / "x.csv"
            screen._do_export_csv(str(bad), "a,b", [1.0, 2.0], [3.0, 4.0])
            await pilot.pause()
            assert any("Could not write" in t for t in _toasts(app))
            assert app.is_running

    @pytest.mark.asyncio
    async def test_csv_export_success(self, tmp_path):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            screen = RdfScreen(app.molecule)
            app.push_screen(screen)
            await pilot.pause()
            out = tmp_path / "x.csv"
            screen._do_export_csv(str(out), "a,b", [1.0, 2.0], [3.0, 4.0])
            assert out.read_text().splitlines() == ["a,b", "1.000000,3.000000", "2.000000,4.000000"]


# ---------------------------------------------------------------------------
# no cell: guards instead of silent success
# ---------------------------------------------------------------------------

class TestNoCellGuards:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cmd", [
        "wrap_center", "wrap_forward", "sort", "del_lone", "extend", "edit_h",
        "write_cif", "write_poscar", "write_mpmc",
    ])
    async def test_command_refused_without_cell(self, cmd):
        app = _app(_mol(pbc=False))
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            depth = len(app.screen_stack)
            undo = len(app._undo_stack)
            _run(app, cmd)
            await pilot.pause()
            assert any("needs a unit cell" in t for t in _toasts(app)), _toasts(app)
            assert len(app.screen_stack) == depth, "no modal should open"
            assert len(app._undo_stack) == undo

    @pytest.mark.asyncio
    async def test_commands_allowed_with_cell(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            _run(app, "wrap_center")
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert not any("needs a unit cell" in t for t in _toasts(app))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["_write_cif_file", "_write_poscar_file",
                                        "_write_mpmc_file"])
    async def test_writer_without_cell_leaves_no_file(self, tmp_path, method):
        app = _app(_mol(pbc=False))
        app._mpmc_state = {"write_charges": False, "write_ff": False}
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            out = tmp_path / "out.dat"
            getattr(app, method)(str(out))
            await pilot.pause()
            assert not out.exists(), "an empty file was left behind"
            assert any("failed" in t and "unit cell" in t for t in _toasts(app)), _toasts(app)
            assert not any(t.startswith("Wrote") for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_pdb_without_cell_is_allowed_and_non_periodic(self, tmp_path):
        app = _app(_mol(pbc=False))
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            _run(app, "write_pdb")
            await pilot.pause()
            assert len(app.screen_stack) == 2, "save dialog should open"
            app.pop_screen()
            out = tmp_path / "out.pdb"
            app._write_pdb_file(str(out))
            await pilot.pause()
            text = out.read_text()
            assert "CRYST1" not in text and text.count("HETATM") == 3
            assert any(t == f"Wrote {out}" for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_writer_with_cell_writes_and_toasts(self, tmp_path):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            out = tmp_path / "out.pdb"
            app._write_pdb_file(str(out))
            await pilot.pause()
            assert out.exists() and "CRYST1" in out.read_text()
            assert any(t == f"Wrote {out}" for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_mpmc_writer_with_cell_round_trips(self, tmp_path):
        from pdb_wizard.io import read_file
        app = _app()
        for a in app.molecule.atoms:
            a.charge = 0.123
        app._mpmc_state = {"write_charges": True, "write_ff": False}
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            out = tmp_path / "out_mpmc.pdb"
            app._write_mpmc_file(str(out))
            await pilot.pause()
            back = read_file(str(out))
            assert len(back.atoms) == 3
            assert all(a.charge == pytest.approx(0.123) for a in back.atoms)


# ---------------------------------------------------------------------------
# operations report what they did
# ---------------------------------------------------------------------------

class TestOperationFeedback:
    @pytest.mark.asyncio
    async def test_delete_lone_none_found(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            undo = len(app._undo_stack)
            app._do_del_lone(True)
            await _settle(app, pilot)
            assert any("No lone atoms found" in t for t in _toasts(app))
            assert len(app._undo_stack) == undo, "no-op must not leave an undo entry"
            assert not app.query_one("#status-bar").has_class("busy")

    @pytest.mark.asyncio
    async def test_delete_lone_reports_count(self):
        app = _app(_mol(extra=[Atom(5, 5, 5, "C")]))
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_del_lone(True)
            await _settle(app, pilot)
            assert any("Deleted 1 lone atom" in t for t in _toasts(app)), _toasts(app)
            assert len(app.molecule.atoms) == 3
            assert len(app._undo_stack) == 1

    @pytest.mark.asyncio
    async def test_edit_h_reports_pairs_changed(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_edit_h("O 1.1")
            await _settle(app, pilot)
            assert any("Set 2 O-H distances to 1.1" in t for t in _toasts(app)), _toasts(app)
            o = app.molecule.atoms[0]
            for h in app.molecule.atoms[1:]:
                assert app.molecule.pbc.min_image(h.x - o.x) == pytest.approx(1.1, abs=1e-6)

    @pytest.mark.asyncio
    async def test_edit_h_no_pairs_is_reported_not_success(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            undo = len(app._undo_stack)
            app._do_edit_h("C 1.1")
            await _settle(app, pilot)
            assert any("No bonded C-H pairs found" in t for t in _toasts(app)), _toasts(app)
            assert not any(t.startswith("Set ") for t in _toasts(app))
            assert len(app._undo_stack) == undo

    @pytest.mark.asyncio
    async def test_substitute_redetects_bonds(self):
        atoms = [Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C")]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        mol.detect_bonds()
        assert len(mol.bonds) == 1
        app = _app(mol)
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_substitute("C H")  # H-H at 1.5 A is far beyond the H bond radius
            await _settle(app, pilot)
            assert all(a.element.symbol == "H" for a in app.molecule.atoms)
            assert app.molecule.bonds == []
            assert any("Substituted 2 C -> H" in t for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_substitute_unknown_target_no_undo(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            undo = len(app._undo_stack)
            app._do_substitute("Xx O")
            await pilot.pause()
            assert any("No Xx atoms found" in t for t in _toasts(app))
            assert len(app._undo_stack) == undo

    @pytest.mark.asyncio
    async def test_force_field_missing_element_is_toasted(self):
        app = _app(_mol(extra=[Atom(5, 5, 5, "Ar")]))
        app._mpmc_state = {"write_charges": False, "write_ff": False}
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._mpmc_step_ff_apply("PHAHST")
            await pilot.pause()
            assert any("Ar not found in forcefield" in t for t in _toasts(app)), _toasts(app)
            ar = app.molecule.atoms[-1]
            assert ar.sigma == 0.0 and ar.epsilon == 0.0


# ---------------------------------------------------------------------------
# undo coverage
# ---------------------------------------------------------------------------

class TestUndoCoverage:
    @pytest.mark.asyncio
    async def test_update_cell_is_undoable(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_update_cell("12 12 12 90 90 90")
            await _settle(app, pilot)
            assert app.molecule.pbc.a == pytest.approx(12)
            assert len(app._undo_stack) == 1
            app._do_undo()
            assert app.molecule.pbc.a == pytest.approx(10)

    @pytest.mark.asyncio
    async def test_remove_cell_is_undoable(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_update_cell("none")
            await _settle(app, pilot)
            assert app.molecule.pbc is None
            app._do_undo()
            assert app.molecule.pbc is not None

    @pytest.mark.asyncio
    async def test_edit_coord_is_undoable(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_edit_coord(0, "1 1 1")
            await _settle(app, pilot)
            assert app.molecule.atoms[0].x[0] == pytest.approx(1.0)
            assert len(app._undo_stack) == 1
            app._do_undo()
            assert app.molecule.atoms[0].x[0] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_qeq_is_undoable(self):
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            await app._do_qeq_async()
            await pilot.pause()
            assert any(abs(a.charge) > 1e-6 for a in app.molecule.atoms)
            assert len(app._undo_stack) == 1
            app.pop_screen()  # InfoModal
            app._do_undo()
            assert all(a.charge == 0.0 for a in app.molecule.atoms)

    @pytest.mark.asyncio
    async def test_mpmc_wizard_start_saves_undo_and_keeps_charges(self):
        app = _app()
        for a in app.molecule.atoms:
            a.charge = 0.5
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            _run(app, "write_mpmc")
            await pilot.pause()
            assert len(app._undo_stack) == 1
            assert len(app.screen_stack) == 2  # "Charges detected" confirm
            app.pop_screen()
            # Choosing "No, pick new charges" must not zero anything yet.
            app._mpmc_step_keep_charges(False)
            await pilot.pause()
            assert all(a.charge == 0.5 for a in app.molecule.atoms)


# ---------------------------------------------------------------------------
# opening files
# ---------------------------------------------------------------------------

_TWO_FRAMES_XYZ = (
    "3\n10 10 10 90 90 90\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n"
    "3\n10 10 10 90 90 90\nO 0.1 0 0\nH 0.857 0.586 0\nH -0.657 0.586 0\n"
)


class TestOpenFile:
    @pytest.mark.asyncio
    async def test_open_trajectory_into_empty_buffer_loads_all_frames(self, tmp_path):
        p = tmp_path / "traj.xyz"
        p.write_text(_TWO_FRAMES_XYZ)
        app = PdbWizardApp(molecule=Molecule(atoms=[]), filepath="")
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_open_file(str(p))
            await _settle(app, pilot)
            assert app._frames is not None and len(app._frames) == 2
            assert len(app._tabs) == 1, "must reuse the empty tab"
            assert app.filepath == str(p)
            assert app.query_one("#traj-bar").has_class("visible")

    @pytest.mark.asyncio
    async def test_open_single_structure_into_empty_buffer(self, tmp_path):
        p = tmp_path / "one.xyz"
        p.write_text(_TWO_FRAMES_XYZ.split("3\n", 2)[0] + "3\n" + _TWO_FRAMES_XYZ.split("3\n", 2)[1])
        app = PdbWizardApp(molecule=Molecule(atoms=[]), filepath="")
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._do_open_file(str(p))
            await _settle(app, pilot)
            assert len(app.molecule.atoms) == 3
            assert not app._frames or len(app._frames) <= 1

    @pytest.mark.asyncio
    async def test_cell_less_trajectory_warns(self, tmp_path):
        p = tmp_path / "nocell_traj.xyz"
        p.write_text("2\n\nC 0 0 0\nO 1.2 0 0\n2\n\nC 0.1 0 0\nO 1.3 0 0\n")
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            app._open_new_tab(str(p))
            await _settle(app, pilot)
            assert any("no cell information" in t for t in _toasts(app)), _toasts(app)

    @pytest.mark.asyncio
    async def test_broken_trajectory_load_is_reported(self, tmp_path):
        """A reader exception in the background load must toast, not vanish."""
        p = tmp_path / "bad.pdb"
        p.write_text("MODEL 1\nATOM x\nENDMDL\nMODEL 2\nATOM y\nENDMDL\n")
        app = _app()
        async with app.run_test(size=(W, H)) as pilot:
            await pilot.pause()
            from unittest import mock
            with mock.patch("pdb_wizard.io.read_pdb_trajectory", side_effect=RuntimeError("corrupt")):
                app._open_new_tab(str(p))
                await _settle(app, pilot)
            assert any("failed: corrupt" in t for t in _toasts(app)), _toasts(app)
            assert not app.query_one("#status-bar").has_class("busy")
