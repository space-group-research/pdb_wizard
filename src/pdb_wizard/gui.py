"""Desktop GUI for PDB Wizard (Qt / PySide6).

``pdb_wizard --gui [file]`` opens a window with a 3D view, the structure's details, an atom
table, geometry tables and tool panels. It has the terminal viewer's features: every edit
and analysis, the input generator, database search, trajectories, tabs, view styles and
PNG/GIF export, plus sorbate packing, force fields and charges. Computations go through the
same shared modules as the terminal viewer (see gui_tools.py).

The view is drawn with QPainter (depth-sorted, shaded spheres), not OpenGL, so it works on
any display, over SSH X-forwarding, and headless (QT_QPA_PLATFORM=offscreen).

PySide6 is optional (``pip install "pdb-wizard[gui]"``). Without it this module still
imports, so the single-file build keeps working; run_gui() then explains what to install.
"""

from __future__ import annotations

import copy
import io as _io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from pdb_wizard.analyses import (
    charge_report,
    elements_of,
    find_energy_file,
    run_density_3d,
    surface_area_text,
    system_info_text,
    unavailable,
    void_volume_text,
)
from pdb_wizard.atom import Atom
from pdb_wizard.charges import apply_qeq_charges
from pdb_wizard.constants import get_element
from pdb_wizard.energy import read_energy_dat
from pdb_wizard.forcefields import (
    apply_ff_to_system_report,
    get_forcefield,
    missing_ff_messages,
)
from pdb_wizard.formats import write_cif, write_lammps_data, write_poscar
from pdb_wizard.geometry import (
    OVERLAP_CUTOFF,
    adjust_h_distances,
    coordination_polyhedra,
    remove_lone_atoms,
    water_atom_indices,
)
from pdb_wizard.gui_tools import (
    AnalysisDialog,
    CellDialog,
    DatabaseDialog,
    EnergyDialog,
    GuiFrameBar,
    GuiGeometryPanel,
    GuiTask,
    InputGeneratorDialog,
    ReduceCellDialog,
    TextReport,
)
from pdb_wizard.io import (
    _is_mpmc_dummy_site,
    assign_framework_charges,
    is_fallback_cell,
    mpmc_movable_count,
    next_mpmc_mol_id,
    read_charges_file,
    read_file,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis, sort_system, system_info, wrap_atoms
from pdb_wizard.pbc import PBC
from pdb_wizard.renderer import ImageRenderer, classify_periodic_bonds
from pdb_wizard.sorbates import (
    SORBATE_MODELS,
    insert_sorbates,
    read_sorbate_template,
    sorbate_atoms,
)
from pdb_wizard.trajectory import (
    frame_bonds,
    is_trajectory_file,
    read_trajectory,
    trajectory_caveats,
)

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    _HAS_QT = True
except ImportError:  # optional extra; see run_gui()
    _HAS_QT = False

if _HAS_QT:
    _QWidget = QtWidgets.QWidget
    _QMainWindow = QtWidgets.QMainWindow
    _QTableModel = QtCore.QAbstractTableModel
else:  # keeps the module (and the single-file build) importable without Qt
    _QWidget = _QMainWindow = _QTableModel = object

GUI_INSTALL_HINT = 'The desktop GUI needs PySide6:  pip install "pdb-wizard[gui]"'

# writers by output suffix; MPMC's dialect is .pqr
_WRITERS = {
    ".xyz": write_xyz,
    ".pdb": write_standard_pdb,
    ".pqr": write_mpmc_pdb,
    ".cif": write_cif,
    ".vasp": write_poscar,
    ".poscar": write_poscar,
    ".lmp": write_lammps_data,
    ".data": write_lammps_data,
}
_SAVE_FILTER = ("MPMC (*.pqr);;PDB (*.pdb);;XYZ (*.xyz);;CIF (*.cif);;"
                "POSCAR (*.vasp *.poscar);;LAMMPS data (*.lmp *.data)")
_OPEN_FILTER = ("Structures (*.pdb *.pqr *.ent *.cif *.xyz *.vasp *.poscar POSCAR CONTCAR "
                "*.lmp *.data *.sdf *.mol2 *.log *.com *.gjf *.zmat *.dcd);;All files (*)")
_FORCE_FIELDS = [("None", None), ("OPLS-AA / UFF", 0), ("PHAHST", 1)]
_DUMMY_RADIUS = 0.15  # A, drawn size of massless model sites (water M/LP, H2 centres)

# The terminal viewer's style and colour cycles
STYLES = [("cpk", "Ball and stick"), ("licorice", "Licorice"), ("vdw", "Space filling"),
          ("ribbon", "Ribbon (protein backbone)")]
COLOR_MODES = [("element", "Element"), ("charge", "Charge"), ("residue", "Residue"),
               ("chain", "Chain"), ("index", "Index")]
BACKGROUNDS = [("theme", "Window theme"), ("dark", "Dark"), ("light", "Light")]
_DEFAULT_BOND_RADIUS = 0.11
CONTROLS_HELP = """\
Mouse (in the 3D view)
  Left drag            rotate
  Right or middle drag pan (left drag also pans in pan mode)
  Wheel                zoom
  Click an atom        select it (shown in the Structure panel and the Atoms table)
  Shift+click atoms    2 atoms give a distance, 3 an angle (in the status bar)
  Double click         reset the view

Keys (click the 3D view first)
  Arrows or h j k l    rotate (pan in pan mode)
  + / -  or  K / J     zoom in / out
  , / .                roll
  r                    reset the view
  c                    centre the structure
  t                    pan mode on/off
  v                    next style
  C                    next colour mode
  i                    dark/light background
  b                    bonds
  #                    atom numbers
  w                    hide water
  P                    polyhedra
  O                    orthographic / perspective
  m                    measure mode: click two atoms to pin their distance

Trajectories
  [ / ]                previous / next frame
  Home / End           first / last frame
  Space                play / pause

Window
  Ctrl+O open    Ctrl+S save as    Ctrl+Z undo    Ctrl+W close tab
  Ctrl+D database    Ctrl+I input generator    Ctrl+E export image
  Ctrl+G geometry panel    F1 this help
"""
_LICORICE_BOND_RADIUS = 0.15


def save_structure(mol: Molecule, path: str) -> None:
    """Write ``mol`` in the format given by the file suffix (``.pqr`` = MPMC)."""
    writer = _WRITERS.get(Path(path).suffix.lower())
    if writer is None:
        raise ValueError(f"unknown output format: {Path(path).suffix or path}")
    with open(path, "w") as fh:
        writer(mol, fh)


# ======================================================================
# 3D view
# ======================================================================

def _rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mix(a: "QtGui.QColor", b: "QtGui.QColor", t: float) -> "QtGui.QColor":
    t = float(np.clip(t, 0.0, 1.0)) if np.isfinite(t) else 0.0
    return QtGui.QColor(int(a.red() * (1 - t) + b.red() * t),
                        int(a.green() * (1 - t) + b.green() * t),
                        int(a.blue() * (1 - t) + b.blue() * t))


def _scaled(c: "QtGui.QColor", f: float) -> "QtGui.QColor":
    return QtGui.QColor(*(int(np.clip(v * f, 0, 255)) for v in (c.red(), c.green(), c.blue())))


def _ca_chains(mol: Molecule) -> List[List[int]]:
    """Backbone traces for the ribbon style: CA atoms (or C bonded to N and C), split
    into chains where consecutive CAs are more than 5 Å apart (as the terminal viewer)."""
    atoms = mol.atoms
    ca = [i for i, a in enumerate(atoms) if a.name.strip() == "CA"]
    if len(ca) < 2 and mol.bonds:
        nbrs: dict = {}
        for a, b in mol.bonds:
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
        ca = [i for i, a in enumerate(atoms) if a.element.symbol == "C"
              and any(atoms[j].element.symbol == "N" for j in nbrs.get(i, ()))
              and any(atoms[j].element.symbol == "C" for j in nbrs.get(i, ()))]
    if len(ca) < 2:
        return []
    chains, cur = [], [ca[0]]
    for prev, i in zip(ca, ca[1:]):
        if np.linalg.norm(atoms[i].position - atoms[prev].position) > 5.0:
            chains.append(cur)
            cur = []
        cur.append(i)
    chains.append(cur)
    return [c for c in chains if len(c) >= 2]


class StructureView(_QWidget):
    """3D structure view drawn with QPainter.

    Left drag rotates (pans in pan mode), right or middle drag pans, the wheel zooms, a
    double click resets the view and a click selects the atom under the cursor. In measure
    mode two clicks pin a distance; shift-click up to three atoms for a distance or angle.
    The terminal viewer's keys work when the view has focus (b, v, i, #, C, w, m, P, O, t,
    c, r, arrows, +/-, ',' and '.').
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.mol: Optional[Molecule] = None
        self.show_bonds = True
        self.show_cell = True
        self.depth_cue = True
        self.style_name = "cpk"
        self.color_mode = "element"
        self.background = "theme"
        self.show_atom_numbers = False
        self.hide_water = False
        self.show_polyhedra = False
        self.orthographic = True
        self.pan_mode = False
        self.measure_mode = False
        self.atom_scale = 1.0
        self.bond_radius = _DEFAULT_BOND_RADIUS
        self.ambient, self.diffuse, self.specular, self.shininess = 0.50, 0.60, 0.40, 32.0
        self.selected: Optional[int] = None
        self.highlighted: set = set()
        self.picked: List[int] = []
        self.measurements: List[tuple] = []  # (atom indices, label)
        self.density: Optional[tuple] = None  # (positions, values)
        self.show_density = True
        self.on_hover: Optional[Callable[[Optional[int]], None]] = None
        self.on_select: Optional[Callable[[Optional[int]], None]] = None
        self.on_message: Optional[Callable[[str], None]] = None
        self.on_option: Optional[Callable[[], None]] = None  # a key changed a view option
        self._rot = np.eye(3)
        self._zoom = 20.0
        self._pan = QtCore.QPointF(0, 0)
        self._center = np.zeros(3)
        self._radius = 10.0
        self._fit_pending = False
        self._drag_pos: Optional[QtCore.QPointF] = None
        self._drag_button = None
        self._screen = np.zeros((0, 2))
        self._radii_px = np.zeros(0)
        self._depth = np.zeros(0)
        self._visible = np.zeros(0, dtype=bool)
        self._bonds: list = []
        self._periodic = np.zeros(0, dtype=bool)
        self._d_min = np.zeros((0, 3))
        self._cache: dict = {}

    # ---- data ---------------------------------------------------------------------------

    def set_molecule(self, mol: Optional[Molecule], reset_view: bool = True,
                     redetect: bool = True) -> None:
        self.mol = mol
        self.selected = None
        self.picked = []
        self.highlighted = set()
        n = len(mol.atoms) if mol is not None else 0
        if reset_view:
            self.measurements = []
        else:  # frames of one trajectory: keep pins whose atoms still exist
            self.measurements = [m for m in self.measurements if all(i < n for i in m[0])]
        self.refresh_bonds(redetect)
        if reset_view:
            self.reset_view()
        self.update()

    def refresh_bonds(self, redetect: bool = True) -> None:
        """Re-detect bonds (after an edit) and rebuild the drawing caches."""
        self._cache = {}
        mol = self.mol
        if mol is None or not mol.atoms:
            self._bonds, self._periodic, self._d_min = [], np.zeros(0, dtype=bool), np.zeros((0, 3))
            return
        if redetect or not mol.bonds:
            mol.detect_bonds()
        dummy = {i for i, a in enumerate(mol.atoms) if _is_mpmc_dummy_site(a)}
        self._bonds = [(i, j) for i, j in mol.bonds if i not in dummy and j not in dummy]
        pos = np.array([a.x for a in mol.atoms])
        self._periodic, self._d_min = classify_periodic_bonds(pos, self._bonds, mol.pbc)

    def invalidate(self) -> None:
        self._cache = {}
        self.update()

    def has_cell(self) -> bool:
        """A real cell to draw. Trajectories without one get a 10^6 Å stand-in box from
        the reader; drawing or fitting the view to that would shrink the atoms to a dot."""
        return (self.mol is not None and self.mol.pbc is not None
                and not is_fallback_cell(self.mol.pbc))

    def cell_corners(self) -> np.ndarray:
        """Cell corners around the atoms: the box is placed in whichever periodic image
        the structure is stored (0..1, -1/2..1/2, ...), snapped to half cells."""
        pbc = self.mol.pbc
        frac = np.array([a.x for a in self.mol.atoms]) @ pbc.reciprocal_basis_matrix
        origin = np.round((frac.mean(axis=0) - 0.5) * 2) / 2
        return pbc.corners() + origin @ pbc.basis_matrix

    def reset_view(self) -> None:
        """Look down c with a slight tilt, structure centred and filling the view."""
        self._rot = _rotation(0, -0.35) @ _rotation(1, 0.45)
        self._pan = QtCore.QPointF(0, 0)
        mol = self.mol
        if mol is None or not mol.atoms:
            return
        pts = np.array([a.x for a in mol.atoms])
        if self.has_cell():
            corners = self.cell_corners()
            self._center = corners.mean(axis=0)
            pts = np.vstack([corners, pts])
        else:
            self._center = pts.mean(axis=0)
        self._radius = max(float(np.linalg.norm(pts - self._center, axis=1).max()), 2.0)
        if self.width() < 50 or self.height() < 50:
            self._fit_pending = True  # not laid out yet: fit on the first paint
            return
        self._fit_pending = False
        self._zoom = 0.47 * min(self.width(), self.height()) / self._radius
        self.update()

    def center_view(self) -> None:
        """Undo panning (the terminal viewer's 'c')."""
        self._pan = QtCore.QPointF(0, 0)
        self.update()

    def rotate(self, d_yaw: float, d_pitch: float) -> None:
        self._rot = _rotation(0, d_pitch) @ _rotation(1, d_yaw) @ self._rot
        self.update()

    def roll(self, angle: float) -> None:
        self._rot = _rotation(2, angle) @ self._rot
        self.update()

    def zoom(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 500.0))
        self.update()

    def view_state(self) -> tuple:
        return self._rot.copy(), self._zoom, QtCore.QPointF(self._pan), self._center.copy()

    def set_view_state(self, state: tuple) -> None:
        self._rot, self._zoom, self._pan, self._center = state
        self._fit_pending = False
        self.update()

    # ---- options ------------------------------------------------------------------------

    def set_style(self, name: str) -> None:
        self.style_name = name
        self.bond_radius = _LICORICE_BOND_RADIUS if name == "licorice" else _DEFAULT_BOND_RADIUS
        self.invalidate()

    def cycle_style(self) -> str:
        names = [k for k, _ in STYLES]
        self.set_style(names[(names.index(self.style_name) + 1) % len(names)])
        return self.style_name

    def cycle_color_mode(self) -> str:
        modes = [k for k, _ in COLOR_MODES]
        self.color_mode = modes[(modes.index(self.color_mode) + 1) % len(modes)]
        self.invalidate()
        return self.color_mode

    def set_measure_mode(self, on: bool) -> None:
        self.measure_mode = on
        self.picked = []
        self.highlighted = set()
        self.update()

    def clear_measurements(self) -> None:
        self.measurements = []
        self.picked = []
        self.update()

    def set_density(self, positions, values) -> None:
        self.density = None if positions is None else (np.asarray(positions),
                                                       np.asarray(values, float))
        self.update()

    # ---- projection ---------------------------------------------------------------------

    def _camera(self) -> float:
        return 3.0 * self._radius

    def _project(self, xyz: np.ndarray) -> tuple:
        """Screen points, depth (larger is nearer) and per-point perspective scale."""
        p = (np.asarray(xyz, dtype=float).reshape(-1, 3) - self._center) @ self._rot.T
        if self.orthographic:
            f = np.ones(len(p))
        else:
            cam = self._camera()
            f = cam / np.maximum(cam - p[:, 2], 0.1 * cam)
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        return (np.column_stack([cx + p[:, 0] * self._zoom * f, cy - p[:, 1] * self._zoom * f]),
                p[:, 2], f)

    def atom_radius(self, atom: Atom) -> float:
        """Drawn radius in A for the current style."""
        if _is_mpmc_dummy_site(atom):
            return _DUMMY_RADIUS
        if self.style_name == "licorice":
            return self.bond_radius
        if self.style_name == "vdw":
            return atom.element.vdw_radius * self.atom_scale
        return 0.28 * atom.element.vdw_radius * self.atom_scale

    def atom_at(self, x: float, y: float) -> Optional[int]:
        """Index of the front-most atom drawn under the point, or None."""
        if not len(self._screen):
            return None
        d = np.hypot(self._screen[:, 0] - x, self._screen[:, 1] - y)
        hit = np.where((d <= np.maximum(self._radii_px, 3.0)) & self._visible)[0]
        if not len(hit):
            return None
        return int(hit[np.argmax(self._depth[hit])])

    # ---- cached per-structure data ------------------------------------------------------

    def _cached(self, key: str, fn: Callable):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def atom_colors(self) -> List["QtGui.QColor"]:
        mol = self.mol

        def build():
            if self.color_mode == "element":
                rgb = [a.element.cpk_color for a in mol.atoms]
            else:  # the terminal renderer's palettes, so both viewers colour alike
                rgb = ImageRenderer(1, 1)._compute_atom_colors(mol, self.color_mode)
            return [QtGui.QColor(*c) for c in rgb]

        return self._cached(f"colors:{self.color_mode}", build)

    def atom_color(self, atom: Atom) -> "QtGui.QColor":
        return self.atom_colors()[self.mol.atoms.index(atom)]

    def _hidden(self) -> set:
        if not self.hide_water:
            return set()
        return self._cached("water", lambda: water_atom_indices(self.mol))

    def _background(self) -> "QtGui.QColor":
        if self.background == "dark":
            return QtGui.QColor(14, 16, 20)
        if self.background == "light":
            return QtGui.QColor(250, 250, 250)
        return self.palette().color(QtGui.QPalette.Base)

    def _text_color(self) -> "QtGui.QColor":
        bg = self._background()
        return QtGui.QColor(30, 30, 30) if bg.lightness() > 140 else QtGui.QColor(225, 228, 232)

    def _fog(self, color: "QtGui.QColor", t: float) -> "QtGui.QColor":
        """Blend far objects (t near 0) toward the background."""
        if not self.depth_cue:
            return color
        return _mix(color, self._background(), 0.55 * (1.0 - t))

    def _shaded(self, center, r: float, base: "QtGui.QColor"):
        """Sphere shading from the ambient / diffuse / specular / shininess settings."""
        lit = _scaled(base, self.ambient + self.diffuse)
        g = QtGui.QRadialGradient(center, r, center + QtCore.QPointF(-0.35 * r, -0.4 * r))
        g.setColorAt(0.0, _mix(lit, QtGui.QColor(255, 255, 255), self.specular * 1.3))
        g.setColorAt(float(np.clip(0.9 / np.sqrt(self.shininess / 8.0), 0.08, 0.9)), lit)
        g.setColorAt(1.0, _scaled(base, self.ambient * 1.08))
        return g

    # ---- painting -----------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QtGui.QPainter(self)
        try:  # an exception with the painter still active crashes Qt on the next paint
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        bg = self._background()
        grad = QtGui.QRadialGradient(self.width() / 2, self.height() / 2,
                                     0.75 * max(self.width(), self.height()))
        grad.setColorAt(0.0, bg.lighter(115) if bg.lightness() < 128 else bg)
        grad.setColorAt(1.0, bg.darker(115))
        painter.fillRect(self.rect(), grad)
        mol = self.mol
        if self._fit_pending and mol is not None and mol.atoms:
            self.reset_view()
        if mol is None or not mol.atoms:
            painter.setPen(self.palette().color(QtGui.QPalette.PlaceholderText))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "Open a structure (File → Open or drop a file here)")
            self._screen = np.zeros((0, 2))
            return
        pos = np.array([a.x for a in mol.atoms])
        screen, depth, persp = self._project(pos)
        radii = np.array([self.atom_radius(a) for a in mol.atoms])
        hidden = self._hidden()
        visible = np.ones(len(mol.atoms), dtype=bool)
        if hidden:
            visible[list(hidden)] = False
        self._screen, self._depth, self._visible = screen, depth, visible
        self._radii_px = radii * self._zoom * persp
        zmin, zmax = float(depth.min()), float(depth.max())
        span = max(zmax - zmin, 1e-6)
        colors = self.atom_colors()

        if self.show_cell and self.has_cell():
            self._paint_cell(painter, mol)
        if self.density is not None and self.show_density:
            self._paint_density(painter)
        chains = _ca_chains(mol) if self.style_name == "ribbon" else []
        if chains:
            self._paint_ribbon(painter, chains, screen)
        else:
            if self.show_bonds and self._bonds and self.style_name != "vdw":
                self._paint_bonds(painter, pos, screen, depth, persp, zmin, span, colors,
                                  visible)
            self._paint_atoms(painter, mol, screen, depth, zmin, span, colors, visible)
        if self.show_polyhedra:
            self._paint_polyhedra(painter, mol)
        self._paint_marks(painter, mol, screen)
        if self.show_atom_numbers:
            self._paint_numbers(painter, screen, depth, visible)
        self._paint_measurements(painter, mol, screen)
        self._paint_axes(painter, mol)

    def _paint_atoms(self, painter, mol, screen, depth, zmin, span, colors, visible) -> None:
        outline = QtGui.QPen(QtGui.QColor(0, 0, 0, 90), 0.8)
        for i in np.argsort(depth):  # far to near
            if not visible[i]:
                continue
            atom = mol.atoms[i]
            r = float(self._radii_px[i])
            c = QtCore.QPointF(screen[i, 0], screen[i, 1])
            base = self._fog(colors[i], (depth[i] - zmin) / span)
            if _is_mpmc_dummy_site(atom):
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(base.red(), base.green(), base.blue(), 150))
                painter.drawEllipse(c, max(r, 1.5), max(r, 1.5))
                continue
            painter.setBrush(self._shaded(c, r, base))
            painter.setPen(outline if r > 4 else QtCore.Qt.NoPen)
            painter.drawEllipse(c, r, r)

    def _paint_cell(self, painter, mol) -> None:
        corners = self._project(self.cell_corners())[0]
        color = self._text_color()
        color.setAlpha(110)
        painter.setPen(QtGui.QPen(color, 1.2, QtCore.Qt.DashLine))
        for i, j in mol.pbc.edges():
            painter.drawLine(QtCore.QPointF(*corners[i]), QtCore.QPointF(*corners[j]))

    def _paint_bonds(self, painter, pos, screen, depth, persp, zmin, span, colors,
                     visible) -> None:
        segments = []  # (depth, p, q, atom index, perspective)
        for k, (i, j) in enumerate(self._bonds):
            if not (visible[i] and visible[j]):
                continue
            if self._periodic[k]:  # stubs toward each atom's nearest image
                half = self._d_min[k] / 2
                ends = self._project(np.array([pos[i] + half, pos[j] - half]))[0]
                segments.append((depth[i], screen[i], ends[0], i))
                segments.append((depth[j], screen[j], ends[1], j))
            else:
                mid = (screen[i] + screen[j]) / 2
                segments.append((depth[i], screen[i], mid, i))
                segments.append((depth[j], screen[j], mid, j))
        segments.sort(key=lambda s: s[0])
        for z, p, q, idx in segments:
            width = max(1.5, 2 * self.bond_radius * self._zoom * float(persp[idx]))
            color = self._fog(colors[idx], (z - zmin) / span).darker(115)
            painter.setPen(QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(*p), QtCore.QPointF(*q))

    def _paint_ribbon(self, painter, chains, screen) -> None:
        palette = [(50, 130, 255), (255, 100, 80), (80, 220, 100), (255, 200, 50),
                   (200, 100, 255), (100, 220, 220), (255, 150, 200), (180, 180, 100)]
        width = max(2.0, 0.5 * self._zoom)
        for k, chain in enumerate(chains):
            pts = screen[chain]
            path = QtGui.QPainterPath(QtCore.QPointF(*pts[0]))
            for a in range(len(pts) - 1):  # Catmull-Rom through the CA positions
                p0, p1 = pts[max(a - 1, 0)], pts[a]
                p2, p3 = pts[a + 1], pts[min(a + 2, len(pts) - 1)]
                c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
                path.cubicTo(QtCore.QPointF(*c1), QtCore.QPointF(*c2), QtCore.QPointF(*p2))
            painter.setPen(QtGui.QPen(QtGui.QColor(*palette[k % len(palette)]), width,
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                      QtCore.Qt.RoundJoin))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)

    def _paint_polyhedra(self, painter, mol) -> None:
        polys = self._cached("polyhedra", lambda: coordination_polyhedra(mol.atoms, mol.pbc))
        colors = self.atom_colors()
        for ci, nbrs in polys:
            pts = self._project(nbrs)[0]
            c = colors[ci]
            fill = QtGui.QColor(c.red(), c.green(), c.blue(), 40)
            edge = QtGui.QColor(c.red() // 2, c.green() // 2, c.blue() // 2, 200)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(QtGui.QPolygonF(_convex_hull(pts)))
            painter.setPen(QtGui.QPen(edge, 1.0))
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    painter.drawLine(QtCore.QPointF(*pts[a]), QtCore.QPointF(*pts[b]))

    def _paint_density(self, painter) -> None:
        positions, values = self.density
        if not len(values):
            return
        pts, depth, persp = self._project(positions)
        vmax = float(values.max()) or 1.0
        for i in np.argsort(depth):
            t = float(values[i] / vmax)
            color = _mix(QtGui.QColor(60, 120, 255), QtGui.QColor(255, 70, 60), t)
            color.setAlpha(int(40 + 140 * t))
            r = (2.0 + 6.0 * t) * float(persp[i])
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(*pts[i]), r, r)

    def _paint_marks(self, painter, mol, screen) -> None:
        rings = [(i, self.palette().color(QtGui.QPalette.Highlight))
                 for i in ([self.selected] if self.selected is not None else [])]
        marked = sorted(self.highlighted | set(self.picked))
        rings += [(i, QtGui.QColor(255, 210, 60)) for i in marked]
        painter.setBrush(QtCore.Qt.NoBrush)
        for i, color in rings:
            if i < len(mol.atoms):
                painter.setPen(QtGui.QPen(color, 2.5))
                r = float(self._radii_px[i]) + 4
                painter.drawEllipse(QtCore.QPointF(screen[i, 0], screen[i, 1]), r, r)

    def _paint_numbers(self, painter, screen, depth, visible) -> None:
        painter.setPen(self._text_color())
        order = [i for i in np.argsort(depth) if visible[i]][-3000:]  # nearest 3000
        for i in order:
            painter.drawText(QtCore.QPointF(screen[i, 0] + 0.6 * self._radii_px[i],
                                            screen[i, 1] - 0.6 * self._radii_px[i]), str(i + 1))

    def _paint_measurements(self, painter, mol, screen) -> None:
        if not self.measurements:
            return
        accent = QtGui.QColor(255, 210, 60)
        for idx, _ in self.measurements:
            if any(i >= len(mol.atoms) for i in idx):
                continue
            label = f"{self.distance(*idx):.3f} Å" if len(idx) == 2 else ""
            pts = [QtCore.QPointF(*screen[i]) for i in idx]
            painter.setPen(QtGui.QPen(accent, 1.5, QtCore.Qt.DashLine))
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a, b)
            mid = sum((p for p in pts), QtCore.QPointF(0, 0)) / len(pts)
            painter.setPen(accent)
            painter.drawText(mid + QtCore.QPointF(6, -6), label)

    def _paint_axes(self, painter, mol) -> None:
        if not self.has_cell():
            return
        origin = QtCore.QPointF(42, self.height() - 42)
        colors = [QtGui.QColor(230, 80, 80), QtGui.QColor(90, 190, 90), QtGui.QColor(90, 140, 240)]
        for k, (name, vec) in enumerate(zip("abc", mol.pbc.basis_matrix)):
            d = (vec / np.linalg.norm(vec)) @ self._rot.T
            tip = origin + QtCore.QPointF(26 * d[0], -26 * d[1])
            painter.setPen(QtGui.QPen(colors[k], 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(origin, tip)
            painter.drawText(tip + QtCore.QPointF(3 * np.sign(d[0]) - 3, 4), name)

    # ---- measuring ----------------------------------------------------------------------

    def _vec(self, i: int, j: int) -> np.ndarray:
        d = self.mol.atoms[i].x - self.mol.atoms[j].x
        return self.mol.pbc.wrap(d) if self.mol.pbc is not None else d

    def distance(self, i: int, j: int) -> float:
        d = self.mol.atoms[i].x - self.mol.atoms[j].x
        if self.mol.pbc is not None:
            return self.mol.pbc.min_image(d)
        return float(np.linalg.norm(d))

    def angle(self, i: int, j: int, k: int) -> float:
        v1, v2 = self._vec(i, j), self._vec(k, j)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def clear_marks(self) -> None:
        """Forget selection, picks, highlights and pinned measurements (after an edit the
        indices may point at other atoms, or past the end)."""
        self.selected = None
        self.picked = []
        self.highlighted = set()
        self.measurements = []

    def pick(self, idx: int, shift: bool = False) -> str:
        """Measure-mode / shift-click picking, as in the terminal viewer. Returns the
        message to show."""
        n = len(self.mol.atoms) if self.mol is not None else 0
        self.picked = [i for i in self.picked if i < n]
        if not 0 <= idx < n:
            return ""
        a = self.mol.atoms[idx]
        if self.measure_mode:
            if idx not in self.picked:
                self.picked.append(idx)
            if len(self.picked) == 1:
                msg = f"Atom {idx + 1}: {a.element.symbol} — click another to measure"
            else:
                i, j = self.picked[:2]
                d = self.distance(i, j)
                self.measurements.append(((i, j), f"{d:.3f} Å"))
                msg = f"Distance {i + 1}-{j + 1} = {d:.4f} Å (pinned)"
                self.picked = []
        else:
            if idx not in self.picked:
                self.picked.append(idx)
            del self.picked[:-3]
            p = self.picked
            if len(p) == 1:
                msg = f"Atom {p[0] + 1}: {a.element.symbol} ({a.name})"
            elif len(p) == 2:
                msg = f"Distance {p[0] + 1}-{p[1] + 1} = {self.distance(*p):.4f} Å"
            else:
                msg = f"Angle {p[0] + 1}-{p[1] + 1}-{p[2] + 1} = {self.angle(*p):.2f}°"
        self.update()
        return msg

    # ---- input --------------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = event.position()
        self._drag_button = event.button()
        self._pressed_at = event.position()
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        p = event.position()
        if self._drag_pos is not None:
            dx, dy = p.x() - self._drag_pos.x(), p.y() - self._drag_pos.y()
            if self._drag_button == QtCore.Qt.LeftButton and not self.pan_mode:
                self.rotate(dx * 0.01, dy * 0.01)
            else:
                self._pan += QtCore.QPointF(dx, dy)
                self.update()
            self._drag_pos = p
        elif self.on_hover is not None:
            self.on_hover(self.atom_at(p.x(), p.y()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        p = event.position()
        moved = (p - getattr(self, "_pressed_at", p)).manhattanLength()
        if moved < 4 and event.button() == QtCore.Qt.LeftButton:
            idx = self.atom_at(p.x(), p.y())
            shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
            if idx is not None and (self.measure_mode or shift):
                msg = self.pick(idx, shift)
                if self.on_message is not None:
                    self.on_message(msg)
            else:
                if not shift:
                    self.picked = []
                self.selected = idx
                if self.on_select is not None:
                    self.on_select(self.selected)
            self.update()
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.zoom(1.12 if event.angleDelta().y() > 0 else 1 / 1.12)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key, text = event.key(), event.text()
        step = 0.1
        moves = {QtCore.Qt.Key_Left: (-step, 0), QtCore.Qt.Key_Right: (step, 0),
                 QtCore.Qt.Key_Up: (0, -step), QtCore.Qt.Key_Down: (0, step),
                 QtCore.Qt.Key_H: (-step, 0), QtCore.Qt.Key_L: (step, 0),
                 QtCore.Qt.Key_K: (0, -step), QtCore.Qt.Key_J: (0, step)}
        toggles = {"b": "show_bonds", "#": "show_atom_numbers", "w": "hide_water",
                   "P": "show_polyhedra", "O": "orthographic", "t": "pan_mode"}
        if key in moves and text not in ("K", "J"):
            dx, dy = moves[key]
            if self.pan_mode:
                self._pan += QtCore.QPointF(-dx * 300, -dy * 300)
                self.update()
            else:
                self.rotate(dx, dy)
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal) or text == "K":
            self.zoom(1.12)
        elif key == QtCore.Qt.Key_Minus or text == "J":
            self.zoom(1 / 1.12)
        elif text == ",":
            self.roll(0.1)
        elif text == ".":
            self.roll(-0.1)
        elif text == "r":
            self.reset_view()
        elif text == "c":
            self.center_view()
        elif text == "v":
            self.cycle_style()
        elif text == "C":
            self.cycle_color_mode()
        elif text == "i":
            self.background = "light" if self._background().lightness() < 128 else "dark"
            self.update()
        elif text == "m":
            self.set_measure_mode(not self.measure_mode)
        elif text in toggles:
            setattr(self, toggles[text], not getattr(self, toggles[text]))
            self.invalidate()
        else:
            super().keyPressEvent(event)
            return
        if self.on_option is not None:
            self.on_option()

    def resizeEvent(self, event) -> None:  # noqa: N802
        old, new = event.oldSize(), event.size()
        if old.width() > 50 and old.height() > 50 and not self._fit_pending:
            self._zoom *= min(new.width(), new.height()) / max(min(old.width(), old.height()), 1)
        super().resizeEvent(event)


def _qt_alive(obj) -> bool:
    """False once Qt has deleted the C++ side of a widget (closed with WA_DeleteOnClose)."""
    try:
        obj.isVisible()
    except RuntimeError:
        return False
    return True


def _cell_problem(a: float, b: float, c: float, alpha: float, beta: float,
                  gamma: float) -> Optional[str]:
    """Why these cell parameters are unusable, or None."""
    values = (a, b, c, alpha, beta, gamma)
    if not all(np.isfinite(v) for v in values):
        return "Cell parameters must be finite numbers"
    if min(a, b, c) <= 0:
        return "Cell lengths must be positive"
    if not all(0 < x < 180 for x in (alpha, beta, gamma)):
        return "Cell angles must be between 0 and 180 degrees"
    ca, cb, cg = (np.cos(np.radians(x)) for x in (alpha, beta, gamma))
    if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg <= 1e-10:
        return "These angles do not make a cell (zero or negative volume)"
    return None


def _convex_hull(pts: np.ndarray) -> list:
    """Screen-space convex hull (monotone chain) as QPointFs, for polyhedron shading."""
    p = sorted(map(tuple, np.asarray(pts, float)))
    if len(p) < 3:
        return [QtCore.QPointF(*q) for q in p]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return [QtCore.QPointF(*q) for q in lower[:-1] + upper[:-1]]


# ======================================================================
# Atom table
# ======================================================================

class AtomTableModel(_QTableModel):
    HEADERS = ["#", "Element", "Name", "Molecule", "x", "y", "z", "q (e)", "α (Å³)",
               "ε (K)", "σ (Å)"]

    def __init__(self) -> None:
        super().__init__()
        self.mol: Optional[Molecule] = None

    def set_molecule(self, mol: Optional[Molecule]) -> None:
        self.beginResetModel()
        self.mol = mol
        self.endResetModel()

    def rowCount(self, parent=None) -> int:  # noqa: N802
        return 0 if self.mol is None else len(self.mol.atoms)

    def columnCount(self, parent=None) -> int:  # noqa: N802
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=None):  # noqa: N802
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=None):
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.TextAlignmentRole) or self.mol is None:
            return None
        col = index.column()
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter) if col >= 4 else None
        a = self.mol.atoms[index.row()]
        if col == 0:
            return index.row() + 1
        if col == 1:
            return a.element.symbol
        if col == 2:
            return a.name
        if col == 3:
            return f"M {a.mpmc_mol} ({a.mpmc_res})" if a.mpmc_flag == "M" else "framework"
        values = [a.x[0], a.x[1], a.x[2], a.charge, a.alpha, a.epsilon, a.sigma]
        return f"{values[col - 4]:.4f}" if col in (4, 5, 6, 7) else f"{values[col - 4]:.3f}"


# ======================================================================
# Documents (tabs)
# ======================================================================

class GuiDocument:
    """One open structure or trajectory: what a tab holds."""

    def __init__(self, mol: Optional[Molecule] = None, path: Optional[str] = None,
                 frames: Optional[list] = None, name: str = "") -> None:
        self.mol = mol
        self.path = path
        self.frames = frames
        self.frame_idx = 0
        self.undo: list = []
        self.name = name or (Path(path).name if path else "untitled")
        self.view_state: Optional[tuple] = None
        self.density: Optional[tuple] = None
        self.measurements: list = []
        self.loading = False  # a trajectory is still being read into this tab

    @property
    def empty(self) -> bool:
        return self.mol is None or not self.mol.atoms


def _doc_attr(name: str):
    return property(lambda self: getattr(self.doc, name),
                    lambda self, value: setattr(self.doc, name, value))


# ======================================================================
# Main window
# ======================================================================

class MainWindow(_QMainWindow):
    """PDB Wizard's desktop window."""

    mol = _doc_attr("mol")
    path = _doc_attr("path")
    frames = _doc_attr("frames")
    _undo = _doc_attr("undo")

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__()
        self.docs: List[GuiDocument] = [GuiDocument()]
        self.doc_index = 0
        self.dialogs: list = []  # keeps open tool windows alive
        self.load_task: Optional[GuiTask] = None
        self.task: Optional[GuiTask] = None
        self._db_cache: Optional[str] = None
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self.view = StructureView()
        self.view.on_hover = self._hovered
        self.view.on_select = self._selected
        self.view.on_message = lambda msg: self.statusBar().showMessage(msg, 8000)
        self.view.on_option = self._sync_view_controls
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setDocumentMode(True)
        self.tab_bar.setExpanding(False)
        self.tab_bar.addTab("untitled")
        self.tab_bar.currentChanged.connect(self._switch_doc)
        self.tab_bar.tabCloseRequested.connect(self.close_tab)
        self.frame_bar = GuiFrameBar(self.set_frame)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.tab_bar)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.frame_bar)
        self.setCentralWidget(central)
        self._build_info_dock()
        self._build_tools_dock()
        self._build_table_dock()
        self._build_geometry_dock()
        self._build_actions()
        self.statusBar().showMessage("Ready")
        self._update_title()
        self._update_supercell_preview()
        if path:
            self.open_path(path)

    @property
    def doc(self) -> GuiDocument:
        return self.docs[self.doc_index]

    # ---- layout -------------------------------------------------------------------------

    def _build_info_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Structure", self)
        dock.setObjectName("structure")
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        self.info_form = QtWidgets.QFormLayout()
        self.info_labels = {}
        for key in ("File", "Formula", "Atoms", "Sorbates", "a, b, c (Å)",
                    "α, β, γ (°)", "Volume (Å³)", "Density (g/cm³)",
                    "Net charge (e)", "Frame"):
            label = QtWidgets.QLabel("—")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setWordWrap(True)
            self.info_labels[key] = label
            self.info_form.addRow(f"<b>{key}</b>", label)
        lay.addLayout(self.info_form)
        lay.addWidget(QtWidgets.QLabel("<b>Elements</b>"))
        self.legend = QtWidgets.QListWidget()
        self.legend.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.legend, 1)
        self.selection_label = QtWidgets.QLabel("Click an atom to inspect it.")
        self.selection_label.setWordWrap(True)
        lay.addWidget(self.selection_label)
        dock.setWidget(body)
        dock.setMinimumWidth(250)
        self.info_dock = dock
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_tools_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Tools", self)
        dock.setObjectName("tools")
        tools = QtWidgets.QTabWidget()
        tools.setDocumentMode(True)
        tools.addTab(self._view_page(), "View")

        # supercell
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.extend_spins = []
        for axis in "abc":
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 20)
            spin.setToolTip("Unit cells along this axis in the result (1 leaves it as it is)")
            spin.valueChanged.connect(lambda _: self._update_supercell_preview())
            self.extend_spins.append(spin)
            form.addRow(f"Cells along {axis}", spin)
        self.supercell_preview = QtWidgets.QLabel()
        form.addRow(self.supercell_preview)
        btn = QtWidgets.QPushButton("Build supercell")
        btn.clicked.connect(self.build_supercell)
        form.addRow(btn)
        wrap_row = QtWidgets.QHBoxLayout()
        for label, forward in (("Wrap (centred)", False), ("Wrap (0–1)", True)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, f=forward: self.wrap(f))
            wrap_row.addWidget(b)
        form.addRow(wrap_row)
        reduce_btn = QtWidgets.QPushButton("Reduce supercell…")
        reduce_btn.clicked.connect(lambda: self.reduce_supercell())
        form.addRow(reduce_btn)
        tools.addTab(page, "Supercell")

        # sorbates
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.sorbate_combo = QtWidgets.QComboBox()
        for name, model in SORBATE_MODELS.items():
            self.sorbate_combo.addItem(f"{model.description}", name)
        self.sorbate_combo.addItem("From an MPMC .pqr file…", "__file__")
        idx = self.sorbate_combo.findData("h2_bss")
        self.sorbate_combo.setCurrentIndex(max(idx, 0))
        form.addRow("Model", self.sorbate_combo)
        self.sorbate_count = QtWidgets.QSpinBox()
        self.sorbate_count.setRange(1, 100000)
        self.sorbate_count.setValue(10)
        self.sorbate_fill = QtWidgets.QCheckBox("Fill the cell")
        self.sorbate_fill.toggled.connect(lambda on: self.sorbate_count.setEnabled(not on))
        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(self.sorbate_count)
        count_row.addWidget(self.sorbate_fill)
        form.addRow("Molecules", count_row)
        self.sorbate_scale = QtWidgets.QDoubleSpinBox()
        self.sorbate_scale.setRange(0.5, 1.2)
        self.sorbate_scale.setSingleStep(0.05)
        self.sorbate_scale.setValue(0.85)
        self.sorbate_scale.setToolTip("Closest allowed approach, as a fraction of the LJ σ")
        form.addRow("Overlap scale", self.sorbate_scale)
        btn = QtWidgets.QPushButton("Pack sorbates")
        btn.clicked.connect(lambda: self.pack_sorbates())
        form.addRow(btn)
        hint = QtWidgets.QLabel("Random positions and orientations without overlaps. Relax "
                                "them in MPMC (NVT) before starting μVT.")
        hint.setWordWrap(True)
        form.addRow(hint)
        tools.addTab(page, "Sorbates")

        # force field and charges
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.ff_combo = QtWidgets.QComboBox()
        for label, idx in _FORCE_FIELDS:
            self.ff_combo.addItem(label, idx)
        self.ff_combo.setCurrentIndex(1)
        ff_btn = QtWidgets.QPushButton("Apply force field")
        ff_btn.clicked.connect(self.apply_force_field)
        form.addRow("Force field", self.ff_combo)
        form.addRow(ff_btn)
        q_btn = QtWidgets.QPushButton("Load charges file…")
        q_btn.clicked.connect(lambda: self.load_charges())
        qeq_btn = QtWidgets.QPushButton("QEq charges")
        qeq_btn.clicked.connect(self.qeq_charges)
        form.addRow(q_btn)
        form.addRow(qeq_btn)
        note = QtWidgets.QLabel("Only framework atoms are changed; packed sorbates keep their "
                                "model parameters.")
        note.setWordWrap(True)
        form.addRow(note)
        tools.addTab(page, "Charges && FF")

        self.tools = tools
        dock.setWidget(tools)
        dock.setMinimumWidth(300)
        self.tools_dock = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _view_page(self) -> "QtWidgets.QWidget":
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        v = self.view
        self.style_combo = QtWidgets.QComboBox()
        for key, label in STYLES:
            self.style_combo.addItem(label, key)
        self.style_combo.currentIndexChanged.connect(
            lambda _: self.set_style(self.style_combo.currentData()))
        form.addRow("Style", self.style_combo)
        self.color_combo = QtWidgets.QComboBox()
        for key, label in COLOR_MODES:
            self.color_combo.addItem(label, key)
        self.color_combo.currentIndexChanged.connect(
            lambda _: self._set_view("color_mode", self.color_combo.currentData(), True))
        form.addRow("Colour by", self.color_combo)
        self.bg_combo = QtWidgets.QComboBox()
        for key, label in BACKGROUNDS:
            self.bg_combo.addItem(label, key)
        self.bg_combo.currentIndexChanged.connect(
            lambda _: self._set_view("background", self.bg_combo.currentData()))
        form.addRow("Background", self.bg_combo)
        self.view_checks = {}
        for attr, label in (("show_bonds", "Bonds"), ("show_cell", "Unit cell"),
                            ("show_atom_numbers", "Atom numbers"), ("hide_water", "Hide water"),
                            ("show_polyhedra", "Polyhedra"), ("orthographic", "Orthographic"),
                            ("depth_cue", "Depth cueing"), ("measure_mode", "Measure mode")):
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(getattr(v, attr))
            chk.toggled.connect(lambda on, a=attr: self._toggle_view(a, on))
            self.view_checks[attr] = chk
            form.addRow(chk)
        # kept for scripts and tests written against the first version
        self.chk_bonds = self.view_checks["show_bonds"]
        self.chk_cell = self.view_checks["show_cell"]
        self.chk_fog = self.view_checks["depth_cue"]
        self.sliders = {}
        for attr, label, lo, hi, scale in (("atom_scale", "Atom size", 0.3, 2.0, 100),
                                           ("bond_radius", "Bond radius (Å)", 0.02, 0.3, 100),
                                           ("ambient", "Ambient", 0.0, 1.0, 100),
                                           ("diffuse", "Diffuse", 0.0, 1.0, 100),
                                           ("specular", "Specular", 0.0, 1.0, 100),
                                           ("shininess", "Shininess", 1.0, 128.0, 1)):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(int(lo * scale), int(hi * scale))
            s.setValue(int(round(getattr(v, attr) * scale)))
            s.valueChanged.connect(lambda val, a=attr, k=scale: self._set_view(a, val / k))
            self.sliders[attr] = s
            form.addRow(label, s)
        self.size_slider = self.sliders["atom_scale"]
        row = QtWidgets.QHBoxLayout()
        for label, slot in (("Reset view", self.view.reset_view),
                            ("Clear measurements", self.view.clear_measurements)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        form.addRow(row)
        page.setWidget(body)
        return page

    def _build_table_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Atoms", self)
        dock.setObjectName("atoms")
        self.table_model = AtomTableModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(self.table.fontMetrics().height() + 8)
        self.table.clicked.connect(lambda idx: self._selected(idx.row(), from_table=True))
        dock.setWidget(self.table)
        self.table_dock = dock
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [190], QtCore.Qt.Vertical)

    def _build_geometry_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Geometry", self)
        dock.setObjectName("geometry")
        self.geometry = GuiGeometryPanel(self._highlight, self.edit_coord)
        dock.setWidget(self.geometry)
        self.geometry_dock = dock
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.tabifyDockWidget(self.table_dock, dock)
        self.table_dock.raise_()

    def _build_actions(self) -> None:
        style = self.style()
        bar = self.addToolBar("Main")
        bar.setObjectName("main")
        bar.setMovable(False)
        menus = {name: self.menuBar().addMenu(name) for name in
                 ("&File", "&Edit", "&View", "&Tools", "&Analysis", "&Help")}
        self.actions_by_name = {}

        def action(menu, text, slot, shortcut=None, icon=None, toolbar=False, checkable=False):
            act = QtGui.QAction(text, self)
            if icon is not None:
                act.setIcon(style.standardIcon(icon))
            if shortcut:
                act.setShortcut(QtGui.QKeySequence(shortcut))
            act.setCheckable(checkable)
            act.triggered.connect(slot)
            menus[menu].addAction(act)
            if toolbar:
                bar.addAction(act)
            self.actions_by_name[text.split("\t")[0].replace("&", "").rstrip("…")] = act
            return act

        sp = QtWidgets.QStyle
        f, e, v, t, a = "&File", "&Edit", "&View", "&Tools", "&Analysis"
        action(f, "&Open…", self.open_dialog, "Ctrl+O", sp.SP_DialogOpenButton, True)
        action(f, "Save &as…", self.save_dialog, "Ctrl+S", sp.SP_DialogSaveButton, True)
        menus[f].addSeparator()
        action(f, "Open frame in new &tab", self.frame_to_tab)
        action(f, "Fetch from &database…", lambda: self.open_tool("db_search"), "Ctrl+D")
        menus[f].addSeparator()
        action(f, "Export &image…", lambda: self.export_image(), "Ctrl+E")
        action(f, "Export rotation &GIF…", lambda: self.export_gif())
        menus[f].addSeparator()
        action(f, "&Close tab", lambda: self.close_tab(self.doc_index), "Ctrl+W")
        action(f, "&Quit", self.close, "Ctrl+Q")

        self.undo_action = action(e, "&Undo", self.undo, "Ctrl+Z", sp.SP_ArrowBack, True)
        self.undo_action.setEnabled(False)
        menus[e].addSeparator()
        action(e, "&Build supercell…", self._show_supercell_tab)
        action(e, "&Reduce supercell…", lambda: self.reduce_supercell())
        action(e, "Wrap (&centred)", lambda: self.wrap(False))
        action(e, "Wrap (&forward)", lambda: self.wrap(True))
        action(e, "Update unit c&ell…", lambda: self.update_cell())
        menus[e].addSeparator()
        action(e, "&Sort atoms", self.sort_atoms)
        action(e, "Substitute ele&ment…", lambda: self.substitute_element())
        action(e, "Delete &lone atoms", self.delete_lone_atoms)
        action(e, "Edit &H distances…", lambda: self.edit_h_distances())
        bar.addSeparator()

        action(v, "&Reset view", self.view.reset_view, "R", sp.SP_BrowserReload, True)
        action(v, "C&entre\tC", self.view.center_view)
        action(v, "Zoom &in", lambda: self.view.zoom(1.2), "Ctrl++")
        action(v, "Zoom &out", lambda: self.view.zoom(1 / 1.2), "Ctrl+-")
        menus[v].addSeparator()
        action(v, "Cycle &style\tV", lambda: self._key_option(self.view.cycle_style))
        action(v, "Cycle &colour mode\tShift+C",
               lambda: self._key_option(self.view.cycle_color_mode))
        menus[v].addSeparator()
        # The same options as the View tab; the keys work while the 3D view has the focus.
        self.view_actions = {}
        for attr, label in (("show_bonds", "&Bonds\tB"), ("show_cell", "&Unit cell"),
                            ("show_atom_numbers", "Atom &numbers\t#"),
                            ("hide_water", "Hide &water\tW"),
                            ("show_polyhedra", "&Polyhedra\tShift+P"),
                            ("orthographic", "Ortho&graphic\tShift+O"),
                            ("depth_cue", "Dept&h cueing"),
                            ("measure_mode", "&Measure mode\tM")):
            act = action(v, label, lambda on, a=attr: self._menu_toggle_view(a, on),
                         checkable=True)
            act.setChecked(bool(getattr(self.view, attr)))
            self.view_actions[attr] = act
        action(v, "Clear measurements", self.view.clear_measurements)
        menus[v].addSeparator()
        self.density_action = action(v, "Show 3D &density", lambda: None, checkable=True)
        self.density_action.setChecked(True)
        self.density_action.toggled.connect(self._toggle_density)
        self.density_action.setEnabled(False)
        action(v, "Clear 3D density", lambda: self.set_density(None, None))
        menus[v].addSeparator()
        for dock in (self.info_dock, self.tools_dock, self.table_dock, self.geometry_dock):
            menus[v].addAction(dock.toggleViewAction())
        self.geometry_dock.toggleViewAction().setShortcut(QtGui.QKeySequence("Ctrl+G"))

        action(t, "&Pack sorbates…", lambda: self._show_tools_tab(2))
        action(t, "&Force field and charges…", lambda: self._show_tools_tab(3))
        action(t, "Load &charges…", lambda: self.load_charges())
        action(t, "Generate &QEq charges", self.qeq_charges)
        menus[t].addSeparator()
        action(t, "&Input generator…", lambda: self.open_tool("input_generator"), "Ctrl+I")

        for key, label in (("info", "System &info"), ("void_vol", "&Void volume"),
                           ("surface_area", "&Surface area"),
                           ("pore_size", "&Pore size distribution"), (None, None),
                           ("rdf", "&Radial distribution"),
                           ("coordination", "&Coordination number"),
                           ("hbonds", "&Hydrogen bonds"), ("msd", "&Mean squared displacement"),
                           ("rmsd", "RMSD vs &time"), ("gyration", "Radius of &gyration"),
                           ("density", "&Density profile"), ("density3d", "&3D density map"),
                           ("pxrd", "Powder &XRD"), (None, None),
                           ("energy_plot", "&Energy plot (energy.dat)…"),
                           ("isotherm_plot", "Plot is&otherm results…")):
            if key is None:
                menus[a].addSeparator()
            else:
                action(a, label, lambda _=False, k=key: self.open_tool(k))
        action("&Help", "&Keyboard and mouse", self.show_controls, "F1")
        action("&Help", "&About PDB Wizard", self.about)

        # the terminal viewer's trajectory keys
        for keys, slot in (("]", lambda: self.frame_bar.go(self.frame_bar.index + 1)),
                           ("[", lambda: self.frame_bar.go(self.frame_bar.index - 1)),
                           ("Home", lambda: self.frame_bar.go(0)),
                           ("End", lambda: self.frame_bar.go(self.frame_bar.n - 1)),
                           ("Space", self.frame_bar.toggle_play)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            sc.activated.connect(slot)

    # ---- documents ----------------------------------------------------------------------

    def _store_view(self) -> None:
        d = self.doc
        d.view_state = self.view.view_state()
        d.density = self.view.density
        d.measurements = list(self.view.measurements)

    def _switch_doc(self, index: int) -> None:
        if index < 0 or index >= len(self.docs) or index == self.doc_index:
            return
        self._store_view()
        self.frame_bar.stop(notify=False)
        self.doc_index = index
        self._restore_doc()

    def _restore_doc(self) -> None:
        """Show the current document with its saved camera, measurements and density."""
        d = self.doc
        self.view.set_molecule(d.mol, reset_view=d.view_state is None, redetect=False)
        if d.view_state is not None:
            self.view.set_view_state(d.view_state)
        self.view.measurements = list(d.measurements)
        self.view.set_density(*(d.density if d.density else (None, None)))
        self.density_action.setEnabled(d.density is not None)
        self.frame_bar.set_count(len(d.frames) if d.frames else 1, d.frame_idx)
        self._refresh_panels()
        self.undo_action.setEnabled(bool(d.undo))

    def new_tab(self, mol: Optional[Molecule], path: Optional[str] = None,
                name: str = "", frames: Optional[list] = None) -> GuiDocument:
        """Open ``mol`` in a tab: the current one if it is empty, else a new one."""
        d = GuiDocument(mol, path, frames, name)
        if self.doc.empty and not self.doc.loading:
            self.docs[self.doc_index] = d
            self.tab_bar.setTabText(self.doc_index, d.name)
            self._show_doc()
        else:
            self._store_view()
            self.docs.append(d)
            self.tab_bar.addTab(d.name)
            self.tab_bar.setCurrentIndex(len(self.docs) - 1)  # -> _switch_doc
        return d

    def _show_doc(self) -> None:
        d = self.doc
        self.view.set_molecule(d.mol, reset_view=True, redetect=False)
        self.view.set_density(None, None)
        self.density_action.setEnabled(False)
        self.frame_bar.set_count(len(d.frames) if d.frames else 1, d.frame_idx)
        self._refresh_panels()
        self.undo_action.setEnabled(bool(d.undo))

    def close_tab(self, index: int) -> None:
        if not 0 <= index < len(self.docs):
            return
        self.frame_bar.stop(notify=False)
        self._close_windows_of(self.docs[index])
        if len(self.docs) == 1:
            self.docs[0] = GuiDocument()
            self.tab_bar.setTabText(0, "untitled")
            self._show_doc()
            return
        current = self.doc_index
        if index != current:
            self._store_view()
        del self.docs[index]
        target = current - 1 if index < current else min(current, len(self.docs) - 1)
        self.tab_bar.blockSignals(True)
        self.tab_bar.removeTab(index)
        self.tab_bar.setCurrentIndex(target)
        self.tab_bar.blockSignals(False)
        self.doc_index = target
        if index == current:
            self._restore_doc()

    # ---- state --------------------------------------------------------------------------

    def set_molecule(self, mol: Optional[Molecule], reset_view: bool = True) -> None:
        self.mol = mol
        if self.frames and mol is not None:
            self.frames[self.doc.frame_idx] = mol
        self.view.set_molecule(mol, reset_view=reset_view)
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        self.table_model.set_molecule(self.mol)
        self.geometry.set_molecule(self.mol)
        self._refresh_info()
        self._update_title()
        self._update_supercell_preview()
        for d in self._live_windows():  # tool windows work on their tab's current atoms
            if getattr(d, "doc", None) is self.doc and hasattr(d, "set_structure"):
                d.set_structure(self.doc.mol, self.doc.frames)

    def _changed(self, message: str) -> None:
        """After an edit: redraw everything and say what happened."""
        self.view.clear_marks()
        self._selected(None)
        self.view.refresh_bonds()
        self.view.update()
        self._refresh_panels()
        self.statusBar().showMessage(message, 8000)

    def _checkpoint(self) -> None:
        # remember which frame was edited: undo must restore that frame, not the shown one
        self._undo.append((self.doc.frame_idx, copy.deepcopy(self.mol)))
        del self._undo[:-20]
        self.undo_action.setEnabled(True)

    def _discard_checkpoint(self) -> None:
        if self._undo:
            self._undo.pop()
        self.undo_action.setEnabled(bool(self._undo))

    def undo(self) -> None:
        if not self._undo:
            return
        frame, mol = self._undo.pop()
        if self.frames and frame < len(self.frames):
            self.frame_bar.stop(notify=False)
            self.doc.frame_idx = frame
            self.frame_bar.set_count(len(self.frames), frame)
        self.set_molecule(mol, reset_view=False)
        self.view.clear_marks()
        self.undo_action.setEnabled(bool(self._undo))
        self.statusBar().showMessage("Undone", 4000)

    def _need_structure(self, what: str, cell: bool = False) -> bool:
        if self.mol is None or not self.mol.atoms:
            self._warn(f"{what} needs a structure. Open one first.")
            return False
        if cell and self.mol.pbc is None:
            self._warn(f"{what} needs a unit cell (PBC). This structure has none — "
                       "set one with Edit > Update unit cell first.")
            return False
        return True

    def _set_view(self, attr: str, value, invalidate: bool = False) -> None:
        setattr(self.view, attr, value)
        if invalidate:
            self.view.invalidate()
        else:
            self.view.update()

    def _toggle_view(self, attr: str, on: bool) -> None:
        if attr == "measure_mode":
            self.view.set_measure_mode(on)
            if on:
                self.statusBar().showMessage(
                    "Measure mode: click two atoms to pin their distance", 6000)
        else:
            self._set_view(attr, on, invalidate=True)

    def _menu_toggle_view(self, attr: str, on: bool) -> None:
        self._toggle_view(attr, on)
        self._sync_view_controls()

    def set_style(self, name: str) -> None:
        self.view.set_style(name)
        self._sync_view_controls()
        if name == "ribbon" and self.mol is not None and self.mol.atoms:
            if not _ca_chains(self.mol):
                self.statusBar().showMessage(
                    "No protein backbone (CA atoms) here; showing ball and stick", 6000)

    def _key_option(self, fn: Callable) -> None:
        fn()
        self._sync_view_controls()

    def _sync_view_controls(self) -> None:
        """Make the View tab match the view after a key or menu changed it."""
        v = self.view
        for combo, value in ((self.style_combo, v.style_name), (self.color_combo, v.color_mode),
                             (self.bg_combo, v.background)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        for attr, chk in self.view_checks.items():
            for box in (chk, getattr(self, "view_actions", {}).get(attr)):
                if box is not None:
                    box.blockSignals(True)
                    box.setChecked(bool(getattr(v, attr)))
                    box.blockSignals(False)
        s = self.sliders["bond_radius"]
        s.blockSignals(True)
        s.setValue(int(round(v.bond_radius * 100)))
        s.blockSignals(False)

    def _update_title(self) -> None:
        name = self.doc.name if not self.doc.empty or self.doc.path else "no structure"
        frames = self.frames
        extra = (f" (frame {self.doc.frame_idx + 1}/{len(frames)})"
                 if frames and len(frames) > 1 else "")
        self.setWindowTitle(f"PDB Wizard — {name}{extra}")
        self.tab_bar.setTabText(self.doc_index, self.doc.name)

    def _refresh_info(self) -> None:
        labels = self.info_labels
        self.legend.clear()
        if self.mol is None:
            for label in labels.values():
                label.setText("—")
            return
        info = system_info(self.mol, self.path or "")
        labels["File"].setText(Path(self.path).name if self.path else f"({self.doc.name})")
        counts: dict = {}
        for a in self.mol.atoms:
            if not _is_mpmc_dummy_site(a):
                counts[a.element.symbol] = counts.get(a.element.symbol, 0) + 1
        labels["Formula"].setText(" ".join(f"{el}<sub>{n}</sub>" for el, n in counts.items()))
        labels["Atoms"].setText(str(info["n_atoms"]))
        labels["Sorbates"].setText(str(mpmc_movable_count(self.mol)))
        labels["Net charge (e)"].setText(f"{sum(a.charge for a in self.mol.atoms):+.4f}")
        frames = self.frames
        labels["Frame"].setText(f"{self.doc.frame_idx + 1} / {len(frames)}"
                                if frames and len(frames) > 1 else "—")
        if "cell" in info:
            c = info["cell"]
            labels["a, b, c (Å)"].setText(f"{c['a']:.3f}, {c['b']:.3f}, {c['c']:.3f}")
            labels["α, β, γ (°)"].setText(
                f"{c['alpha']:.2f}, {c['beta']:.2f}, {c['gamma']:.2f}")
            labels["Volume (Å³)"].setText(f"{info['volume']:.2f}")
            labels["Density (g/cm³)"].setText(f"{info['density']:.4f}")
        else:
            for key in ("a, b, c (Å)", "α, β, γ (°)", "Volume (Å³)", "Density (g/cm³)"):
                labels[key].setText("no cell")
        counts = {}
        for a in self.mol.atoms:
            key = "model sites" if _is_mpmc_dummy_site(a) else a.element.symbol
            counts[key] = counts.get(key, 0) + 1
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pix = QtGui.QPixmap(14, 14)
            pix.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pix)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            color = (QtGui.QColor(255, 20, 147) if key == "model sites"
                     else QtGui.QColor(*next(a for a in self.mol.atoms
                                             if a.element.symbol == key).element.cpk_color))
            p.setBrush(color)
            p.setPen(QtGui.QPen(color.darker(160)))
            p.drawEllipse(1, 1, 12, 12)
            p.end()
            self.legend.addItem(QtWidgets.QListWidgetItem(QtGui.QIcon(pix), f"{key}   {n}"))

    def _hovered(self, idx: Optional[int]) -> None:
        if idx is None or self.mol is None:
            self.statusBar().clearMessage()
            return
        self.statusBar().showMessage(self._describe(idx))

    def _selected(self, idx: Optional[int], from_table: bool = False) -> None:
        self.view.selected = idx
        self.view.update()
        if idx is None or self.mol is None:
            self.selection_label.setText("Click an atom to inspect it.")
            return
        self.selection_label.setText(self._describe(idx))
        if not from_table:
            self.table.selectRow(idx)
            self.table.scrollTo(self.table_model.index(idx, 0))

    def _highlight(self, indices: tuple) -> None:
        self.view.highlighted = set(indices)
        self.view.update()

    def _describe(self, idx: int) -> str:
        a = self.mol.atoms[idx]
        where = f"sorbate {a.mpmc_mol}" if a.mpmc_flag == "M" else "framework"
        return (f"#{idx + 1} {a.name} ({a.element.symbol}, {where})  "
                f"x={a.x[0]:.3f} y={a.x[1]:.3f} z={a.x[2]:.3f}  q={a.charge:+.4f} e  "
                f"ε={a.epsilon:.2f} K  σ={a.sigma:.3f} Å")

    def _warn(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "PDB Wizard", text)

    def _live_windows(self) -> list:
        self.dialogs = [d for d in self.dialogs if _qt_alive(d)]
        return list(self.dialogs)

    def _show_window(self, dialog, doc: Optional[GuiDocument] = None) -> object:
        """Show a tool window without blocking. It is freed when closed; ``doc`` ties it to
        a tab (it follows that tab's edits and closes with it)."""
        self._live_windows()
        dialog.doc = doc
        self.dialogs.append(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.show()
        return dialog

    def _close_windows_of(self, doc: GuiDocument) -> None:
        for d in self._live_windows():
            if getattr(d, "doc", None) is doc:
                d.close()

    def report(self, title: str, text: str) -> object:
        return self._show_window(TextReport(title, text, self))

    def _busy(self, message: str, fn: Callable, done: Callable) -> GuiTask:
        """Run ``fn`` in the background with a wait cursor and a status message."""
        self.statusBar().showMessage(message)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        def finish(result):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
            done(result)

        def failed(e):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
            self._warn(f"{message.rstrip('.…')} failed:\n{e}")

        self.task = GuiTask(fn, finish, failed)
        return self.task

    def _snapshot(self) -> Optional[Molecule]:
        """A copy of the shown structure for background work (edits may change the live one)."""
        return copy.deepcopy(self.mol) if self.mol is not None else None

    # ---- files --------------------------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open structure", "", _OPEN_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        """Open a structure (or a trajectory, loaded in the background) in a tab."""
        if not os.path.isfile(path):
            self._warn(f"No such file: {path}")
            return False
        try:
            trajectory = is_trajectory_file(path)
        except Exception:  # an unreadable file: let read_file report it below
            trajectory = False
        if trajectory:
            return self._open_trajectory(path)
        messages: list = []
        try:
            mol = read_file(path, on_message=messages.append)
        except Exception as e:  # any reader error goes to the user, not the console
            self._warn(f"Could not read {Path(path).name}:\n{e}")
            return False
        self.new_tab(mol, path)
        n = len(messages)
        note = (f" (deleted {n} overlapping atom{'s' if n != 1 else ''}, closer than "
                f"{OVERLAP_CUTOFF} Å)" if n else "")
        self.statusBar().showMessage(f"Opened {Path(path).name}: {len(mol.atoms)} atoms{note}",
                                     8000)
        return True

    def _open_trajectory(self, path: str) -> bool:
        d = self.new_tab(Molecule(atoms=[]), path, name=f"{Path(path).name} (loading)")
        d.loading = True
        progress = [0.0]

        def read():
            frames = read_trajectory(path, lambda f: progress.__setitem__(0, f))
            if frames:
                frames[0].detect_bonds()
            return frames

        def finished() -> bool:
            timer.stop()
            d.loading = False
            self.statusBar().clearMessage()
            return d in self.docs  # False if its tab was closed meanwhile

        def done(frames):
            if not finished():
                return
            d.frames, d.frame_idx = frames, 0
            d.mol = frames[0] if frames else Molecule(atoms=[])
            d.name = Path(path).name
            d.view_state = None  # fit the camera to the frames, not the empty placeholder
            if d is self.doc:
                self._show_doc()
            self.tab_bar.setTabText(self.docs.index(d), d.name)
            self.statusBar().showMessage(f"Loaded {len(frames)} frames", 6000)
            for note in trajectory_caveats(frames):
                self._warn(note)

        def failed(e):
            if finished():
                self.close_tab(self.docs.index(d))  # no empty "(loading)" tab left behind
            self._warn(f"Could not read {Path(path).name}:\n{e}")

        timer = QtCore.QTimer(self)
        timer.setInterval(100)
        timer.timeout.connect(lambda: self.statusBar().showMessage(
            f"Loading trajectory... {int(progress[0] * 100)}%"))
        timer.start()
        self.load_task = GuiTask(read, done, failed)
        return True

    def set_frame(self, idx: int) -> None:
        frames = self.frames
        if not frames or len(frames) <= 1:
            return
        idx = max(0, min(idx, len(frames) - 1))
        playing = self.frame_bar.playing
        if idx == self.doc.frame_idx and not playing and self.mol is frames[idx]:
            self._refresh_panels()
            return
        self.doc.frame_idx = idx
        frame_bonds(frames, idx)
        self.mol = frames[idx]
        self.view.set_molecule(self.mol, reset_view=False, redetect=False)
        if not playing:  # tables are expensive; refresh them when playback stops
            self._refresh_panels()
        else:
            self._update_title()

    def frame_to_tab(self) -> Optional[GuiDocument]:
        """Open the current trajectory frame as a standalone structure in a new tab."""
        if not self.frames or self.mol is None or not self.mol.atoms:
            self.statusBar().showMessage("No trajectory frame to open", 4000)
            return None
        stem = Path(self.path).stem if self.path else "frame"
        name = f"{stem}_frame{self.doc.frame_idx + 1}"
        d = self.new_tab(copy.deepcopy(self.mol), None, name)
        self.statusBar().showMessage(f"Opened frame as {name}", 4000)
        return d

    def save_dialog(self) -> None:
        if not self._need_structure("Saving"):
            return
        start = str(Path(self.path).with_suffix(".pqr")) if self.path else "structure.pqr"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save structure", start, _SAVE_FILTER)
        if path:
            self.save_path(path)

    def save_path(self, path: str) -> bool:
        try:
            save_structure(self.mol, path)
        except Exception as e:
            self._warn(f"Could not save {Path(path).name}:\n{e}")
            return False
        self.statusBar().showMessage(f"Saved {path}", 8000)
        return True

    def export_image(self, path: Optional[str] = None) -> Optional[str]:
        if not path:
            start = str(Path(self.path).with_suffix(".png")) if self.path else "structure.png"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export image", start,
                                                            "PNG (*.png)")
        if not path:
            return None
        if not self.view.grab().save(path):
            self._warn(f"Could not write {path} (check the folder and the extension)")
            return None
        self.statusBar().showMessage(f"Saved image {path}", 6000)
        return path

    def export_gif(self, path: Optional[str] = None, n_frames: int = 36) -> Optional[str]:
        """A full turn about the vertical axis, 80 ms per frame (as the terminal viewer)."""
        if not self._need_structure("A GIF"):
            return None
        try:
            from PIL import Image
        except ImportError:
            self._warn("Pillow required: pip install Pillow")
            return None
        if not path:
            start = str(Path(self.path).with_suffix(".gif")) if self.path else "structure.gif"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export rotation GIF", start,
                                                            "GIF (*.gif)")
        if not path:
            return None
        n_frames = max(1, int(n_frames))
        base = self.view._rot.copy()
        images = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for k in range(n_frames):
                self.view._rot = _rotation(1, 2 * np.pi * k / n_frames) @ base
                buf = QtCore.QBuffer()
                buf.open(QtCore.QIODevice.WriteOnly)
                self.view.grab().save(buf, "PNG")
                images.append(Image.open(_io.BytesIO(bytes(buf.data()))).convert("RGB"))
        finally:
            self.view._rot = base
            self.view.update()
            QtWidgets.QApplication.restoreOverrideCursor()
        try:
            images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0)
        except (OSError, ValueError) as e:
            self._warn(f"Could not write {path}:\n{e}")
            return None
        self.statusBar().showMessage(f"Saved {path} ({n_frames} frames)", 6000)
        return path

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            self.open_path(url.toLocalFile())

    def closeEvent(self, event) -> None:  # noqa: N802
        self.frame_bar.stop(notify=False)
        for d in self._live_windows():
            d.close()
        if self._db_cache and os.path.isdir(self._db_cache):
            shutil.rmtree(self._db_cache, ignore_errors=True)
        super().closeEvent(event)

    # ---- edits --------------------------------------------------------------------------

    def _show_tools_tab(self, index: int) -> None:
        self.tools_dock.show()
        self.tools_dock.raise_()
        self.tools.setCurrentIndex(index)

    def _show_supercell_tab(self) -> None:
        self._show_tools_tab(1)

    def build_supercell(self) -> None:
        if not self._need_structure("A supercell", cell=True):
            return
        counts = [spin.value() for spin in self.extend_spins]
        if all(n == 1 for n in counts):
            self.statusBar().showMessage("Set at least one axis above 1", 5000)
            return
        self._checkpoint()
        for axis, n in enumerate(counts):
            if n > 1:
                extend_axis(self.mol, axis, n - 1)  # extend_axis counts the extra copies
        self.view.reset_view()
        size = "×".join(str(n) for n in counts)
        for spin in self.extend_spins:
            spin.setValue(1)
        self._changed(f"Built a {size} supercell: {len(self.mol.atoms)} atoms")

    def _update_supercell_preview(self) -> None:
        counts = [spin.value() for spin in self.extend_spins]
        mol = self.mol
        if mol is None or not mol.atoms or not self.view.has_cell():
            self.supercell_preview.setText("Needs a structure with a unit cell")
            return
        n = len(mol.atoms) * counts[0] * counts[1] * counts[2]
        size = " × ".join(str(c) for c in counts)
        note = " (no change)" if n == len(mol.atoms) else ""
        self.supercell_preview.setText(f"Result: {size} cells, {n:,} atoms{note}")

    def wrap(self, forward: bool) -> None:
        if not self._need_structure("Wrapping atoms", cell=True):
            return
        self._checkpoint()
        wrap_atoms(self.mol, forward=forward)
        self._changed("Wrapped atoms into " + ("0–1" if forward else "−½–½"))

    def sort_atoms(self) -> None:
        if not self._need_structure("Sorting atoms into molecules", cell=True):
            return
        self._checkpoint()
        sort_system(self.mol, False)
        self._changed("Sorted atoms")

    def delete_lone_atoms(self) -> int:
        if not self._need_structure("Finding lone atoms", cell=True):
            return 0
        self._checkpoint()
        _, deleted = remove_lone_atoms(self.mol)
        if not deleted:
            self._discard_checkpoint()
            self.statusBar().showMessage("No lone atoms found", 5000)
            return 0
        n = len(deleted)
        self._changed(f"Deleted {n} lone atom{'s' if n != 1 else ''}")
        return n

    def edit_h_distances(self, element: Optional[str] = None,
                         distance: Optional[float] = None) -> int:
        if not self._need_structure("Editing H distances", cell=True):
            return 0
        if element is None or distance is None:
            others = [e for e in elements_of(self.mol) if e != "H"] or ["O"]
            element, ok = QtWidgets.QInputDialog.getItem(
                self, "Edit H distances", "Set every bonded X–H distance where X is:", others,
                others.index("O") if "O" in others else 0, False)
            if not ok:
                return 0
            distance, ok = QtWidgets.QInputDialog.getDouble(
                self, "Edit H distances", f"{element}–H distance (Å):", 1.0, 0.5, 3.0, 3)
            if not ok:
                return 0
        element = element.strip().capitalize()
        self._checkpoint()
        _, messages = adjust_h_distances(self.mol, element, distance)
        if not messages:
            self._discard_checkpoint()
            self.statusBar().showMessage(f"No bonded {element}-H pairs found; nothing changed",
                                         6000)
            return 0
        n = len(messages)
        self._changed(f"Set {n} {element}-H distance{'s' if n != 1 else ''} to {distance} A")
        return n

    def update_cell(self, values="ask") -> bool:
        """Set the cell to (a, b, c, alpha, beta, gamma), or remove it with None."""
        if not self._need_structure("Updating the cell"):
            return False
        if values == "ask":
            dlg = CellDialog(self.mol.pbc, self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return False
            values = dlg.values()
        if values is None:
            if self.mol.pbc is None:
                self.statusBar().showMessage("This system already has no PBC", 4000)
                return False
            self._checkpoint()
            self.mol.pbc = None
            self._changed("PBC removed")
            return True
        a, b, c, alpha, beta, gamma = (float(v) for v in values)
        problem = _cell_problem(a, b, c, alpha, beta, gamma)
        if problem:
            self._warn(problem)
            return False
        self._checkpoint()
        if self.mol.pbc is not None:
            self.mol.pbc.update(a, b, c, alpha, beta, gamma)
        else:
            self.mol.pbc = PBC(a, b, c, alpha, beta, gamma)
        self._changed(f"Cell: {a:.3f} {b:.3f} {c:.3f} {alpha:.1f} {beta:.1f} {gamma:.1f}")
        return True

    def substitute_element(self, old: Optional[str] = None, new: Optional[str] = None) -> int:
        if not self._need_structure("Substituting"):
            return 0
        if old is None or new is None:
            present = elements_of(self.mol)
            old, ok = QtWidgets.QInputDialog.getItem(self, "Substitute element",
                                                     "Replace every:", present, 0, False)
            if not ok:
                return 0
            new, ok = QtWidgets.QInputDialog.getText(self, "Substitute element",
                                                     f"Replace {old} with (element symbol):")
            if not ok or not new.strip():
                return 0
        new_elem = get_element(new)
        if new_elem.symbol == "X":
            self._warn(f"Unknown element: {new}")
            return 0
        self._checkpoint()
        count = 0
        for atom in self.mol.atoms:
            if atom.element.symbol == old:
                atom._element = new_elem
                atom.name = new_elem.symbol
                count += 1
        if count == 0:
            self._discard_checkpoint()
            self.statusBar().showMessage(f"No {old} atoms found", 4000)
            return 0
        self._changed(f"Substituted {count} {old} -> {new_elem.symbol}")
        return count

    def reduce_supercell(self, dialog: Optional[ReduceCellDialog] = None) -> bool:
        """Reduce Supercell. Pass an already-filled dialog to skip showing it (tests)."""
        if not self._need_structure("Supercell reduction", cell=True):
            return False
        if dialog is None:
            dialog = ReduceCellDialog(self.mol, self)
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return False
        result = dialog.result_mol
        if result is None:
            return False
        kept, total = len(dialog.keep()), len(dialog.sorbates)
        self._checkpoint()
        self.set_molecule(result, reset_view=True)
        self.statusBar().showMessage(
            f"Reduced to primitive cell: {len(result.atoms)} atoms, "
            f"kept {kept}/{total} sorbate molecules", 8000)
        return True

    def edit_coord(self, idx: int, xyz: tuple) -> None:
        if self.mol is None or not 0 <= idx < len(self.mol.atoms):
            return
        if len(xyz) != 3 or not np.all(np.isfinite(xyz)):
            self._warn("Coordinates must be three finite numbers")
            return
        self._checkpoint()
        self.mol.atoms[idx].x = np.array(xyz, dtype=float)
        self._changed(f"Updated atom {idx + 1} position")

    def pack_sorbates(self, model_path: Optional[str] = None) -> int:
        if not self._need_structure("Packing sorbates", cell=True):
            return 0
        key = self.sorbate_combo.currentData()
        if key == "__file__":
            if not model_path:
                model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Sorbate template (MPMC .pqr)", "", "MPMC (*.pqr *.pdb)")
            if not model_path:
                return 0
            try:
                model = read_sorbate_template(model_path)
            except (OSError, ValueError) as e:
                self._warn(str(e))
                return 0
        else:
            model = SORBATE_MODELS[key]
        n = None if self.sorbate_fill.isChecked() else self.sorbate_count.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            placed = insert_sorbates(self.mol, model, n, scale=self.sorbate_scale.value())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not placed:
            self._warn("No room for even one molecule with these settings.")
            return 0
        self._checkpoint()
        first_id = next_mpmc_mol_id(self.mol)
        self.mol.atoms.extend(sorbate_atoms(model, placed, first_id))
        full = "" if n is None or len(placed) == n else f" (asked for {n}; the cell is full)"
        self._changed(f"Packed {len(placed)} {model.label}{full}. Save as .pqr for MPMC.")
        return len(placed)

    def apply_force_field(self) -> None:
        if not self._need_structure("A force field"):
            return
        idx = self.ff_combo.currentData()
        if idx is None:
            return
        self._checkpoint()
        _, missing = apply_ff_to_system_report(self.mol.atoms, get_forcefield(idx))
        self._changed(f"Applied {self.ff_combo.currentText()} to the framework")
        if missing:
            self._warn("\n".join(m.strip("! ") for m in missing_ff_messages(missing)))

    def load_charges(self, path: Optional[str] = None) -> bool:
        if not self._need_structure("Charges"):
            return False
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Charges (.resp or one per line)")
        if not path:
            return False
        try:
            charges = read_charges_file(path)
        except (OSError, ValueError) as e:
            self._warn(f"Could not read charges:\n{e}")
            return False
        self._checkpoint()
        problem = assign_framework_charges(self.mol, charges)
        if problem:
            self._discard_checkpoint()
            self._warn(problem)
            return False
        self._changed(f"Loaded {len(charges)} charges onto the framework")
        return True

    def qeq_charges(self) -> None:
        if not self._need_structure("QEq charges"):
            return
        self._checkpoint()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            apply_qeq_charges(self.mol)
        except Exception as e:  # e.g. an element without QEq parameters
            self._discard_checkpoint()
            self._warn(f"QEq failed: {e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._changed("QEq charges assigned to the framework")
        self.report("QEq Charges", charge_report(self.mol))

    # ---- analysis and tools -------------------------------------------------------------

    def open_tool(self, key: str):
        """Open an Analysis-menu entry. Returns its window (or the background task)."""
        why = unavailable(key, self.mol, self.frames)
        if why:
            self.statusBar().showMessage(why, 6000)
            self._warn(why)
            return None
        mol, frames, doc = self.mol, self.frames, self.doc
        if key == "info":
            return self.report("System Info", system_info_text(mol, self.path or ""))
        if key in ("void_vol", "surface_area"):
            mol = self._snapshot()
        if key == "void_vol":
            return self._busy("Calculating void volume (100k samples)...",
                              lambda: void_volume_text(mol),
                              lambda text: self.report("Void Volume", text))
        if key == "surface_area":
            return self._busy("Calculating surface area (N2 probe)...",
                              lambda: surface_area_text(mol),
                              lambda text: self.report("Surface Area", text))
        if key == "density3d":
            return self.density_3d()
        if key == "energy_plot":
            return self.energy_plot()
        if key == "input_generator":
            return self._show_window(InputGeneratorDialog(mol, self.path or "", self), doc)
        if key == "db_search":
            if self._db_cache is None:
                self._db_cache = tempfile.mkdtemp(prefix="pdb_wizard_db_")
            return self._show_window(DatabaseDialog(self._db_cache, self._open_fetched, self))
        dialog = AnalysisDialog(key, mol, frames, self)
        self._show_window(dialog, doc if key != "isotherm_plot" else None)
        if not dialog.spec.fields:  # nothing to choose: compute straight away
            dialog.run()
        return dialog

    def _open_fetched(self, path: str, label: str, rid: str) -> None:
        messages: list = []
        try:
            mol = read_file(path, True, messages.append)
        except Exception as e:  # a download that is not a readable structure
            self._warn(f"Could not read {label} {rid}:\n{e}")
            return
        self.new_tab(mol, path)
        self.statusBar().showMessage(f"Opened {label} {rid} ({len(mol.atoms)} atoms)", 6000)

    def energy_plot(self, path: Optional[str] = None):
        """MPMC energy.dat next to the structure, or one you pick."""
        path = path or find_energy_file(self.path or "")
        if not path:
            start = str(Path(self.path).parent / "energy.dat") if self.path else "energy.dat"
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open energy file", start,
                                                            "Energy (*.dat *.csv);;All files (*)")
        if not path:
            return None
        data = read_energy_dat(path)
        if not data:
            self._warn(f"Could not read {path}")
            return None
        return self._show_window(EnergyDialog(data, path, self))

    def density_3d(self, element: Optional[str] = None) -> Optional[GuiTask]:
        if element is None:
            items = elements_of(self.mol)
            element, ok = QtWidgets.QInputDialog.getItem(self, "3D density map", "Element:",
                                                         items, 0, False)
            if not ok:
                return None
        doc = self.doc
        frames = list(self.frames)
        current = next((k for k, f in enumerate(frames) if f is self.mol), -1)
        if current >= 0:  # the shown frame may be edited meanwhile; the rest cannot
            frames[current] = self._snapshot()

        def done(res):
            if "positions" not in res.extra:  # no atoms of that element
                self.statusBar().showMessage(res.summary, 6000)
                return
            if doc not in self.docs:
                return
            doc.density = (np.asarray(res.extra["positions"]),
                           np.asarray(res.extra["values"], float))
            if doc is self.doc:  # the map belongs to the tab it was computed for
                self.view.set_density(*doc.density)
                self.density_action.setEnabled(True)
            self.statusBar().showMessage(f"{res.summary} (View > Show 3D density)", 8000)

        return self._busy(f"Computing 3D density for {element}...",
                          lambda: run_density_3d(frames, element, 15), done)

    def set_density(self, positions, values) -> None:
        self.view.set_density(positions, values)
        self.doc.density = None if positions is None else self.view.density
        self.density_action.setEnabled(positions is not None)

    def _toggle_density(self, on: bool) -> None:
        self.view.show_density = on
        self.view.update()

    def show_controls(self):
        """Help > Keyboard and mouse: everything the 3D view responds to."""
        return self.report("Keyboard and mouse", CONTROLS_HELP)

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, "About PDB Wizard",
            "<h3>PDB Wizard</h3><p>Structure handling for porous-materials simulations.</p>"
            "<p>Space group, NC State University. GPL-3.0-or-later.</p>")


# ======================================================================
# Start-up
# ======================================================================

def _apply_theme(app) -> None:
    """Use the desktop's own style (Breeze on KDE); otherwise a dark Fusion theme."""
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    if "kde" in desktop or os.environ.get("KDE_FULL_SESSION"):
        return
    app.setStyle("Fusion")
    pal, c, P = QtGui.QPalette(), QtGui.QColor, QtGui.QPalette
    text, panel = c(225, 228, 232), c(48, 52, 60)
    colors = {
        P.Window: c(37, 40, 46), P.WindowText: text, P.Base: c(28, 31, 36),
        P.AlternateBase: c(42, 46, 53), P.Text: text, P.Button: panel, P.ButtonText: text,
        P.Highlight: c(61, 174, 233), P.HighlightedText: c(255, 255, 255),
        P.ToolTipBase: panel, P.ToolTipText: text, P.PlaceholderText: c(140, 146, 155),
    }
    for role, color in colors.items():
        pal.setColor(role, color)
    for role in (P.Text, P.ButtonText, P.WindowText):
        pal.setColor(P.Disabled, role, c(120, 124, 130))
    app.setPalette(pal)


def run_gui(path: Optional[str] = None) -> int:
    """Open the desktop window (blocking). Returns the exit code."""
    if not _HAS_QT:
        print(GUI_INSTALL_HINT)
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["pdb_wizard"])
    app.setApplicationName("PDB Wizard")
    _apply_theme(app)
    window = MainWindow(path)
    window.show()
    return app.exec()
