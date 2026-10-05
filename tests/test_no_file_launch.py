"""Tests for opening pdb_wizard with no file argument and using File > Open."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    MENU_FILE,
    PdbWizardApp,
)


def _empty_app() -> PdbWizardApp:
    return PdbWizardApp(molecule=Molecule(atoms=[]), filepath="")


# ---------------------------------------------------------------------------
# App boots with no file
# ---------------------------------------------------------------------------

class TestEmptyLaunch:
    @pytest.mark.asyncio
    async def test_app_starts_with_empty_molecule(self):
        """No-file launch must produce a working app with an empty molecule
        and an empty filepath — no crashes during compose."""
        app = _empty_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.molecule.atoms) == 0
            assert app.filepath == ""
            # Single empty tab on the stack
            assert len(app._tabs) == 1

    @pytest.mark.asyncio
    async def test_panels_render_without_atoms(self):
        """The Operations / Visual panels should still render — no exceptions
        from None-checks on missing molecule data."""
        from pdb_wizard.panels import OperationsPanel, VisualPanel
        app = _empty_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Both panels should be queryable, even if not yet visible
            assert app.query_one(OperationsPanel) is not None
            assert app.query_one(VisualPanel) is not None


# ---------------------------------------------------------------------------
# File > Open menu wiring
# ---------------------------------------------------------------------------

class TestOpenMenuEntry:
    def test_open_file_entry_is_in_menu_file(self):
        cmds = [c for c, _l in MENU_FILE if c != "---"]
        assert "open_file" in cmds, (
            "MENU_FILE must include 'open_file' so users can pick a file "
            "after launching with no argument"
        )

    def test_open_file_appears_before_save_entries(self):
        """Open should be the first action in the File menu — that's the
        canonical convention. Saves come after."""
        cmds = [c for c, _l in MENU_FILE if c != "---"]
        assert cmds[0] == "open_file", f"first File menu entry is {cmds[0]!r}"


# ---------------------------------------------------------------------------
# _do_open_file callback: empty buffer → reuse tab; populated → new tab
# ---------------------------------------------------------------------------

class TestDoOpenFile:
    def _write_xyz(self, dir_: Path, name: str, n_atoms: int = 3) -> str:
        p = dir_ / name
        lines = [str(n_atoms), ""]
        for i in range(n_atoms):
            lines.append(f"C {float(i):.4f} 0.0000 0.0000")
        p.write_text("\n".join(lines) + "\n")
        return str(p)

    @pytest.mark.asyncio
    async def test_open_into_empty_buffer_reuses_active_tab(self, tmp_path):
        """When the user launched with no file, the first Open should fill
        the existing empty tab — not create a second tab that strands the
        original empty one."""
        path = self._write_xyz(tmp_path, "first.xyz", 4)
        app = _empty_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert len(app._tabs) == 1
            initial_tab = app._tabs[0]
            assert len(initial_tab.molecule.atoms) == 0

            app._do_open_file(path)
            await pilot.pause()
            await pilot.pause()

            assert len(app._tabs) == 1, (
                f"expected the empty tab to be reused, got {len(app._tabs)} tabs"
            )
            assert app.molecule is app._tabs[0].molecule
            assert len(app.molecule.atoms) == 4
            assert app.filepath == path

    @pytest.mark.asyncio
    async def test_open_into_populated_buffer_creates_new_tab(self, tmp_path):
        """If the user already has a structure loaded, Open should add a
        new tab rather than overwriting the current one."""
        atoms = [Atom(0.0, 0.0, 0.0, "Cu")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        app = PdbWizardApp(molecule=mol, filepath="initial.pdb")
        path = self._write_xyz(tmp_path, "second.xyz", 5)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert len(app._tabs) == 1

            app._do_open_file(path)
            await pilot.pause()
            await pilot.pause()

            # Now two tabs: original Cu + new XYZ
            assert len(app._tabs) == 2
            # Active tab is the new one
            assert app._active_tab == 1
            assert len(app.molecule.atoms) == 5

    @pytest.mark.asyncio
    async def test_open_with_empty_input_is_noop(self):
        """Cancel-from-picker (empty string) must not change state."""
        app = _empty_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial_tabs = len(app._tabs)

            app._do_open_file("")
            await pilot.pause()
            assert len(app._tabs) == initial_tabs

            app._do_open_file(None)
            await pilot.pause()
            assert len(app._tabs) == initial_tabs


# ---------------------------------------------------------------------------
# CLI: file arg is now optional
# ---------------------------------------------------------------------------

class TestCliOptional:
    def test_argparse_accepts_no_file(self):
        """`pdb_wizard` (no args) must parse without an error."""
        from pdb_wizard.__main__ import argparse  # noqa: F401
        import argparse as _argparse
        # Reconstruct the parser to mirror main()
        parser = _argparse.ArgumentParser(prog="pdb_wizard")
        parser.add_argument("file", nargs="?")
        args = parser.parse_args([])
        assert args.file is None
