"""Button *contract* tests — verify that a button's plain-English meaning
matches its actual behavior.

This is the deterministic core of the "does the button do what it says?" idea
(see the test-oracle / metamorphic-testing literature): instead of asserting a
specific output, we assert the *invariant* implied by the button's category.

This file pins the **Cancel contract**, the universal metamorphic relation for
any cancel/close affordance:

    Pressing Cancel (button OR escape) on a modal must
      1. dismiss the modal (return to the prior screen), AND
      2. leave the molecule byte-for-byte unchanged, AND
      3. write no files.

That relation is exactly what the SorbateModal/ForceFieldModal "Cancel == Skip"
bug violated. Every cancelable modal is checked through both affordances, and a
completeness guard fails if a new ModalScreen is added without being covered or
explicitly exempted — so coverage can't silently rot.

A natural extension (not done here) is a Hypothesis RuleBasedStateMachine that
fuzzes *sequences* of open/cancel across modals; the per-modal contract below is
the deterministic gate that catches the bug class on its own.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Input

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.panels import OPERATIONS, OperationsPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    ChargesFileModal,
    ConfirmModal,
    ExtendAxisModal,
    FileSaveModal,
    ForceFieldModal,
    InputModal,
    PdbWizardApp,
    SorbateModal,
)


def _mol(n=4):
    """A molecule with positions, charges and LJ params, so any mutation shows."""
    atoms = []
    for i in range(n):
        a = Atom(float(i), 0.5 * i, 0.0, "C" if i % 2 else "O")
        a.charge = 0.1 * (i + 1)
        a.sigma = 3.4
        a.epsilon = 50.0
        atoms.append(a)
    mol = Molecule(atoms=atoms, pbc=PBC(12, 12, 12, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _app():
    return PdbWizardApp(molecule=_mol(), filepath="t.pdb")


def _fingerprint(mol: Molecule):
    """A hashable snapshot of everything a mutation would touch."""
    atoms = tuple(
        (
            a.element.symbol,
            round(float(a.x[0]), 6), round(float(a.x[1]), 6), round(float(a.x[2]), 6),
            round(float(a.charge), 9),
            round(float(a.sigma), 6), round(float(a.epsilon), 6),
        )
        for a in mol.atoms
    )
    pbc = None
    if mol.pbc is not None:
        p = mol.pbc
        pbc = (p.a, p.b, p.c, p.alpha, p.beta, p.gamma)
    return (atoms, pbc)


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# (name, build(app) -> modal instance, cancel-button id)
# Every modal here exposes a literal "Cancel" button AND an escape binding.
CANCELABLE_MODALS = [
    ("InputModal",      lambda app: InputModal("Enter value:"),                  "input-cancel"),
    ("ConfirmModal",    lambda app: ConfirmModal("Title", "Body text"),          "confirm-cancel"),
    ("FileSaveModal",   lambda app: FileSaveModal("Save as:", default="out.pdb"), "file-cancel"),
    ("ForceFieldModal", lambda app: ForceFieldModal(app.molecule),              "ff-cancel"),
    ("ExtendAxisModal", lambda app: ExtendAxisModal(app.molecule.pbc),          "extend-cancel"),
    ("SorbateModal",    lambda app: SorbateModal(),                             "sorb-cancel"),
    ("ChargesFileModal", lambda app: ChargesFileModal(),                        "charges-cancel"),
]


@pytest.mark.parametrize(
    "name,build,cancel_id", CANCELABLE_MODALS,
    ids=[m[0] for m in CANCELABLE_MODALS],
)
@pytest.mark.parametrize("mechanism", ["button", "escape"])
@pytest.mark.asyncio
async def test_cancel_contract(name, build, cancel_id, mechanism, tmp_path, monkeypatch):
    """Cancel must dismiss the modal, mutate nothing, and write nothing —
    whether triggered by the Cancel button or the escape key."""
    monkeypatch.chdir(tmp_path)  # any relative writes would land here
    app = _app()
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        mol_before = _fingerprint(app.molecule)
        files_before = _files(tmp_path)
        depth_before = len(app.screen_stack)

        app.push_screen(build(app))
        await pilot.pause()
        assert len(app.screen_stack) == depth_before + 1, f"{name} did not open"

        if mechanism == "button":
            await pilot.click(f"#{cancel_id}")
        else:
            await pilot.press("escape")
        await pilot.pause()

        # --- the Cancel contract ---
        assert len(app.screen_stack) == depth_before, (
            f"{name}: Cancel via {mechanism} did not dismiss the modal"
        )
        assert _fingerprint(app.molecule) == mol_before, (
            f"{name}: Cancel via {mechanism} mutated the molecule"
        )
        new_files = _files(tmp_path) - files_before
        assert not new_files, (
            f"{name}: Cancel via {mechanism} wrote files: {sorted(new_files)}"
        )


def test_all_cancelable_modals_are_covered():
    """Completeness guard: every ModalScreen in viewer.py must be exercised by
    the cancel contract above OR explicitly exempted with a reason. A new modal
    added without a cancel test will fail here instead of slipping through."""
    from textual.screen import ModalScreen

    import pdb_wizard.viewer as v
    from pdb_wizard.viewer import _AnalysisScreen

    covered = {m[0] for m in CANCELABLE_MODALS}

    # Exempt modals (not Cancel-button contracts) and why.
    exempt = {
        "InfoModal": "informational; only a Close/dismiss, no Cancel semantics",
        "_AnalysisScreen": "abstract base for analysis screens",
        "ReduceCellScreen": "preview screen; close-cancellation covered elsewhere",
        "InputGeneratorScreen": "multi-field generator; cancel covered by its own e2e",
        "IsothermPlannerScreen": "legacy planner; Close (not Cancel), no write on close",
        "DatabaseSearchScreen": "async search; close-cancellation in test_modal_cancel.py",
        "MenuDropdown": "transient menu; selection/dismiss, no destructive Cancel",
    }

    modals = {
        name for name, obj in inspect.getmembers(v, inspect.isclass)
        if issubclass(obj, ModalScreen) and obj is not ModalScreen
        and obj.__module__ == v.__name__
    }
    # Analysis screens are close-cancellable and covered in test_modal_cancel.py.
    analysis = {
        name for name, obj in inspect.getmembers(v, inspect.isclass)
        if inspect.isclass(obj) and issubclass(obj, _AnalysisScreen)
        and obj is not _AnalysisScreen
    }

    uncovered = modals - covered - set(exempt) - analysis
    assert not uncovered, (
        "New ModalScreen(s) lack a cancel contract test and aren't exempted: "
        f"{sorted(uncovered)}. Add them to CANCELABLE_MODALS or the exempt list."
    )


# ===========================================================================
# Persist contract: "Save as X" writes the file at the chosen path AND leaves
# the molecule unchanged. (Saving must never mutate what you're saving.)
# ===========================================================================

# (panel command, filename to save to). The writer is chosen by the command,
# not the extension, so any path works.
PERSIST_COMMANDS = [
    ("write_xyz",    "out.xyz"),
    ("write_pdb",    "out.pdb"),
    ("write_cif",    "out.cif"),
    ("write_poscar", "POSCAR"),
    ("write_lammps", "out.lmp"),
]


@pytest.mark.parametrize(
    "cmd,fname", PERSIST_COMMANDS, ids=[c for c, _ in PERSIST_COMMANDS],
)
@pytest.mark.asyncio
async def test_persist_contract(cmd, fname, tmp_path, monkeypatch):
    """A Save-as command must create exactly the chosen file and not mutate
    the molecule."""
    monkeypatch.chdir(tmp_path)
    app = _app()
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        mol_before = _fingerprint(app.molecule)
        files_before = _files(tmp_path)
        target = tmp_path / fname
        assert not target.exists()

        app.post_message(OperationsPanel.RunCommand(cmd))
        await pilot.pause()
        # FileSaveModal is up — type the destination and Save.
        app.screen.query_one("#file-input", Input).value = str(target)
        await pilot.click("#file-save")

        # Writers are synchronous, but pause/poll a little to be safe.
        for _ in range(20):
            await pilot.pause()
            if target.exists():
                break
            await asyncio.sleep(0.05)

        # --- the Persist contract ---
        assert target.exists(), f"{cmd}: Save did not write {fname}"
        assert target.stat().st_size > 0, f"{cmd}: wrote an empty file"
        new_files = _files(tmp_path) - files_before
        assert new_files == {fname}, (
            f"{cmd}: expected to write only {fname}, got {sorted(new_files)}"
        )
        assert _fingerprint(app.molecule) == mol_before, (
            f"{cmd}: saving mutated the molecule"
        )


def test_all_save_commands_have_a_persist_contract():
    """Completeness guard: every file-writing 'File' command must have a persist
    contract or be explicitly exempted."""
    covered = {c for c, _ in PERSIST_COMMANDS}
    exempt = {
        "write_mpmc": "multi-step charge/FF/sorbate flow; covered by its own e2e",
        "open_file": "reads, not writes",
        "frame_to_tab": "in-memory tab, no file",
        "export_png": "image export, not a structure file",
        "export_gif": "image export, not a structure file",
        "quit": "exits the app",
    }
    save_like = {
        cmd for cmd, _label, group in OPERATIONS
        if group == "export" and (cmd.startswith("write_") or cmd in exempt)
    }
    uncovered = save_like - covered - set(exempt)
    assert not uncovered, (
        f"Save-as commands without a persist contract: {sorted(uncovered)}. "
        "Add them to PERSIST_COMMANDS or the exempt list."
    )


# ===========================================================================
# Mutate contract: an Edit action changes the molecule AND writes no file.
# (Editing must not silently touch disk; and it must actually do something.)
# ===========================================================================

def _mol_outside_box():
    """A molecule with one atom outside the cell, so wrapping definitely moves it."""
    mol = _mol()
    mol.atoms[0].x = np.array([20.0, 20.0, 20.0])  # outside the 12 A box
    return mol


async def _drive_substitute(app, pilot):
    app.post_message(OperationsPanel.RunCommand("substitute"))
    await pilot.pause()
    # InputModal is up; the molecule has O atoms -> substitute O for N.
    app.screen.query_one("#modal-input", Input).value = "O N"
    await pilot.press("enter")
    await pilot.pause()


async def _drive_wrap_center(app, pilot):
    app.post_message(OperationsPanel.RunCommand("wrap_center"))
    await pilot.pause()
    await pilot.click("#confirm-ok")  # ConfirmModal -> _do_wrap (async)
    await pilot.pause()


# (name, make_mol, drive). Representative deterministic edits — one sync
# (substitute) and one async (wrap) — exercising both dispatch styles.
MUTATE_ACTIONS = [
    ("substitute",  _mol,             _drive_substitute),
    ("wrap_center", _mol_outside_box, _drive_wrap_center),
]


@pytest.mark.parametrize(
    "name,make_mol,drive", MUTATE_ACTIONS, ids=[m[0] for m in MUTATE_ACTIONS],
)
@pytest.mark.asyncio
async def test_mutate_contract(name, make_mol, drive, tmp_path, monkeypatch):
    """An Edit action must change the molecule and write nothing to disk."""
    monkeypatch.chdir(tmp_path)
    app = PdbWizardApp(molecule=make_mol(), filepath="t.pdb")
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        mol_before = _fingerprint(app.molecule)
        files_before = _files(tmp_path)

        await drive(app, pilot)
        # Edits may run in a background task — poll until the molecule changes.
        for _ in range(40):
            if _fingerprint(app.molecule) != mol_before:
                break
            await asyncio.sleep(0.05)
            await pilot.pause()

        # --- the Mutate contract ---
        assert _fingerprint(app.molecule) != mol_before, (
            f"{name}: edit did not change the molecule"
        )
        new_files = _files(tmp_path) - files_before
        assert not new_files, (
            f"{name}: edit wrote files: {sorted(new_files)}"
        )
