"""The classic (no-textual) text menu must match the original pre-textual
PDB Wizard v0.3.0 menu ordering, option numbering, and labels.

The TUI rewrite briefly reorganized the classic menu into nested submenus
(``extend axis, wrap, or sort`` / ``write files``) and added items the original
never had (``list molecules``, ``sort atoms``). Muscle-memory users of the old
tool expect the exact same numbered choices, so this file pins the main menu and
the geometry-analysis submenu to the original's flat layout.

Reference (original v0.3.0):

  Main menu                              Geometry analysis
  1 = geometry analysis                  1 = list bonds
  2 = extend along axis                  2 = list close vdw contacts
  3 = wrap atoms from (0,0,0)..(1,1,1)   3 = list angles
  4 = wrap atoms from (-1/2..)..(1/2..)  4 = list lone atoms
  5 = update cell dimensions             5 = delete lone atoms
  6 = write .xyz                         6 = list coordinates
  7 = write MPMC .pdb                    7 = edit hydrogen bond distances
  8 = write standardized .pdb            8 = preview with VMD
  9 = quit                               9 = back to main menu
"""

from __future__ import annotations

import re
from unittest import mock

import pdb_wizard.menu as menu
from pdb_wizard.atom import Atom
from pdb_wizard.menu import _main_loop_movie, _main_loop_single, _menu_geom_analysis
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

EXPECTED_MAIN = [
    (1, "geometry analysis"),
    (2, "extend along axis"),
    (3, "wrap atoms from (0, 0, 0) to (1, 1, 1)"),
    (4, "wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2)"),
    (5, "update cell dimensions"),
    (6, "write .xyz"),
    (7, "write MPMC .pdb"),
    (8, "write standardized .pdb"),
    (9, "quit"),
]

EXPECTED_GEOM = [
    (1, "list bonds"),
    (2, "list close vdw contacts"),
    (3, "list angles"),
    (4, "list lone atoms"),
    (5, "delete lone atoms"),
    (6, "list coordinates"),
    (7, "edit hydrogen bond distances"),
    (8, "preview with VMD"),
    (9, "back to main menu"),
]

# The trajectory/movie menu has no original to mirror, so it follows the same
# numbering/labels as the single-structure menu, annotated with the scope
# (all frames vs. a single chosen frame).
EXPECTED_MOVIE = [
    (1, "geometry analysis (single frame)"),
    (2, "extend along axis (all frames)"),
    (3, "wrap atoms from (0, 0, 0) to (1, 1, 1) (all frames)"),
    (4, "wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2) (all frames)"),
    (5, "update cell dimensions (all frames)"),
    (6, "write .xyz (all frames)"),
    (7, "write MPMC .pdb (single frame)"),
    (8, "write standardized .pdb (all frames)"),
    (9, "quit"),
]


def _mol() -> Molecule:
    atoms = [Atom(float(i), 0.0, 0.0, "C") for i in range(3)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _traj(n_frames: int = 3, n_atoms: int = 3) -> list[Molecule]:
    frames = []
    for f in range(n_frames):
        atoms = [Atom(float(i) + f * 0.01, 0.0, 0.0, "C") for i in range(n_atoms)]
        m = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        m.detect_bonds()
        frames.append(m)
    return frames


def _capture_menu(func, responses, *args):
    """Run a menu function feeding `responses` to input(), returning the list
    of prompt strings that input() was called with."""
    prompts: list[str] = []
    it = iter(responses)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(it)

    with mock.patch("builtins.input", fake_input):
        func(*args)
    return prompts


def _parse_options(prompt: str) -> list[tuple[int, str]]:
    """Extract (number, label) pairs from a menu prompt, in order."""
    pairs = []
    for m in re.finditer(r"(\d+)\s*=\s*([^\n]+)", prompt):
        pairs.append((int(m.group(1)), m.group(2).strip()))
    return pairs


class TestMainMenuMatchesOriginal:
    def test_main_menu_ordering_and_labels(self):
        # "9" quits immediately after the menu is shown once.
        prompts = _capture_menu(_main_loop_single, ["9"], _mol(), "test.pdb")
        main = next(p for p in prompts if "geometry analysis" in p)
        assert _parse_options(main) == EXPECTED_MAIN

    def test_main_menu_has_no_nested_submenu_labels(self):
        prompts = _capture_menu(_main_loop_single, ["9"], _mol(), "test.pdb")
        main = next(p for p in prompts if "geometry analysis" in p)
        # The nested reorganization these strings came from must be gone.
        assert "extend axis, wrap, or sort" not in main
        assert "write files" not in main
        assert "sort atoms" not in main


class TestGeometryMenuMatchesOriginal:
    def test_geometry_menu_ordering_and_labels(self):
        # "9" returns to the main menu immediately.
        prompts = _capture_menu(_menu_geom_analysis, ["9"], _mol())
        geom = next(p for p in prompts if "list bonds" in p)
        assert _parse_options(geom) == EXPECTED_GEOM

    def test_geometry_menu_dropped_list_molecules(self):
        prompts = _capture_menu(_menu_geom_analysis, ["9"], _mol())
        geom = next(p for p in prompts if "list bonds" in p)
        assert "list molecules" not in geom


class TestMovieMenu:
    def test_movie_menu_ordering_and_labels(self):
        prompts = _capture_menu(_main_loop_movie, ["9"], _traj(3), "traj.xyz")
        m = next(p for p in prompts if "geometry analysis" in p)
        assert _parse_options(m) == EXPECTED_MOVIE

    def test_option_one_is_not_a_no_op(self):
        # Regression: the old movie menu's first option did nothing (`pass`).
        seen = []
        with mock.patch.object(menu, "_menu_geom_analysis",
                               lambda mol: seen.append(mol) or mol):
            _capture_menu(_main_loop_movie, ["1", "2", "9"], _traj(3), "traj.xyz")
        assert len(seen) == 1, "option 1 must actually route to geometry analysis"

    def test_wrap_forward_applies_to_all_frames(self):
        traj = _traj(3)
        calls = []
        real = menu.wrap_atoms

        def spy(mol, forward=False):
            calls.append(forward)
            return real(mol, forward)

        with mock.patch.object(menu, "wrap_atoms", spy):
            _capture_menu(_main_loop_movie, ["3", "9"], traj, "traj.xyz")
        assert calls == [True, True, True]

    def test_extend_applies_to_every_frame(self):
        traj = _traj(3)
        calls = []
        with mock.patch.object(menu, "extend_axis",
                               lambda mol, axis, times: calls.append((axis, times)) or mol):
            _capture_menu(_main_loop_movie, ["2", "x", "2", "9"], traj, "traj.xyz")
        assert calls == [(0, 2), (0, 2), (0, 2)]

    def test_geometry_analysis_routes_to_selected_frame(self):
        traj = _traj(3)
        seen = []
        with mock.patch.object(menu, "_menu_geom_analysis",
                               lambda mol: seen.append(mol) or mol):
            _capture_menu(_main_loop_movie, ["1", "2", "9"], traj, "traj.xyz")
        assert seen == [traj[1]]  # frame "2" -> index 1

    def test_mpmc_write_routes_to_selected_frame(self):
        traj = _traj(3)
        seen = []
        with mock.patch.object(menu, "_write_mpmc_options", lambda mol: seen.append(mol)):
            _capture_menu(_main_loop_movie, ["7", "3", "9"], traj, "traj.xyz")
        assert seen == [traj[2]]  # frame "3" -> index 2

    def test_write_xyz_writes_all_frames(self, tmp_path):
        traj = _traj(3, n_atoms=3)
        out = tmp_path / "out.xyz"
        _capture_menu(_main_loop_movie, ["6", str(out), "9"], traj, "traj.xyz")
        text = out.read_text()
        # each frame starts with a bare atom-count line ("3")
        headers = [ln for ln in text.splitlines() if ln.strip() == "3"]
        assert len(headers) == 3

    def test_write_standard_pdb_writes_all_frames(self, tmp_path):
        traj = _traj(4, n_atoms=2)
        out = tmp_path / "out.pdb"
        _capture_menu(_main_loop_movie, ["8", str(out), "9"], traj, "traj.xyz")
        # 4 frames * 2 atoms = 8 ATOM/HETATM records
        records = [ln for ln in out.read_text().splitlines()
                   if ln.startswith(("ATOM", "HETATM"))]
        assert len(records) == 8


class TestOverlapReporting:
    """The original v0.3.0 ran overlap_detector() on load and printed every
    deletion. The classic port must still print them, not swallow them."""

    def _overlap_mol(self) -> Molecule:
        atoms = [
            Atom(0.0, 0.0, 0.0, "C"),
            Atom(0.01, 0.0, 0.0, "C"),
            Atom(3.0, 0.0, 0.0, "C"),
        ]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        mol.detect_bonds()
        return mol

    def test_single_loop_prints_deletion_on_entry(self, capsys):
        mol = self._overlap_mol()
        _capture_menu(_main_loop_single, ["9"], mol, "x.pdb")
        out = capsys.readouterr().out
        assert "Deleting overlapping atoms" in out
        assert "C     1 ---   C     2" in out

    def test_single_loop_actually_removes_atom(self):
        mol = self._overlap_mol()
        _capture_menu(_main_loop_single, ["9"], mol, "x.pdb")
        assert len(mol.atoms) == 2

    def test_clean_structure_prints_nothing_about_overlaps(self, capsys):
        _capture_menu(_main_loop_single, ["9"], _mol(), "x.pdb")
        assert "Deleting overlapping atoms" not in capsys.readouterr().out
