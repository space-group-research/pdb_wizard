"""Desktop GUI for PDB Wizard (Qt / PySide6).

``pdb_wizard --gui [file]`` opens a window with a 3D ball-and-stick view, the structure's
details, an atom table and tool panels for supercells, sorbate packing, force fields and
charges. The view is drawn with QPainter (depth-sorted, shaded spheres), not OpenGL, so it
works on any display, over SSH X-forwarding, and headless (QT_QPA_PLATFORM=offscreen).

PySide6 is optional (``pip install "pdb-wizard[gui]"``). Without it this module still
imports, so the single-file build keeps working; run_gui() then explains what to install.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.charges import apply_qeq_charges
from pdb_wizard.forcefields import (
    apply_ff_to_system_report,
    get_forcefield,
    missing_ff_messages,
)
from pdb_wizard.formats import write_cif, write_lammps_data, write_poscar
from pdb_wizard.io import (
    _is_mpmc_dummy_site,
    assign_framework_charges,
    mpmc_movable_count,
    read_charges_file,
    read_file,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis, system_info, wrap_atoms
from pdb_wizard.renderer import classify_periodic_bonds
from pdb_wizard.sorbates import (
    SORBATE_MODELS,
    insert_sorbates,
    read_sorbate_template,
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
                "*.lmp *.data *.sdf *.mol2 *.log *.com *.gjf *.zmat);;All files (*)")
_FORCE_FIELDS = [("None", None), ("OPLS-AA / UFF", 0), ("PHAHST", 1)]
_DUMMY_RADIUS = 0.15  # A, drawn size of massless model sites (water M/LP, H2 centres)


def sorbate_atoms(model, placements, first_mol_id: int) -> List[Atom]:
    """Atoms for packed sorbate molecules, flagged movable so MPMC export keeps them so."""
    atoms = []
    for k, xyz in enumerate(placements):
        for site, pos in zip(model.sites, xyz):
            atom = Atom(pos[0], pos[1], pos[2], site.atomtype)
            atom.charge, atom.alpha = site.charge, site.polarizability
            atom.epsilon, atom.sigma = site.epsilon, site.sigma
            atom.c6, atom.c8, atom.c10 = site.c6, site.c8, site.c10
            atom.mpmc_flag, atom.mpmc_mol = "M", first_mol_id + k
            atom.mpmc_res, atom.file_mass = model.label, site.mass
            atoms.append(atom)
    return atoms


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
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class StructureView(_QWidget):
    """Ball-and-stick view drawn with QPainter.

    Left drag rotates, right or middle drag pans, the wheel zooms, a double click resets
    the view and a click selects the atom under the cursor.
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
        self.color_by_charge = False
        self.atom_scale = 1.0
        self.selected: Optional[int] = None
        self.on_hover: Optional[Callable[[Optional[int]], None]] = None
        self.on_select: Optional[Callable[[Optional[int]], None]] = None
        self._rot = np.eye(3)
        self._zoom = 20.0
        self._pan = QtCore.QPointF(0, 0)
        self._center = np.zeros(3)
        self._fit_pending = False
        self._drag_pos: Optional[QtCore.QPointF] = None
        self._drag_button = None
        self._screen = np.zeros((0, 2))
        self._radii_px = np.zeros(0)
        self._depth = np.zeros(0)
        self._bonds: list = []
        self._periodic = np.zeros(0, dtype=bool)
        self._d_min = np.zeros((0, 3))

    # ---- data ---------------------------------------------------------------------------

    def set_molecule(self, mol: Optional[Molecule], reset_view: bool = True) -> None:
        self.mol = mol
        self.selected = None
        self.refresh_bonds()
        if reset_view:
            self.reset_view()
        self.update()

    def refresh_bonds(self) -> None:
        mol = self.mol
        if mol is None or not mol.atoms:
            self._bonds, self._periodic, self._d_min = [], np.zeros(0, dtype=bool), np.zeros((0, 3))
            return
        mol.detect_bonds()
        dummy = {i for i, a in enumerate(mol.atoms) if _is_mpmc_dummy_site(a)}
        self._bonds = [(i, j) for i, j in mol.bonds if i not in dummy and j not in dummy]
        pos = np.array([a.x for a in mol.atoms])
        self._periodic, self._d_min = classify_periodic_bonds(pos, self._bonds, mol.pbc)

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
        if mol.pbc is not None:
            corners = self.cell_corners()
            self._center = corners.mean(axis=0)
            pts = np.vstack([corners, pts])
        else:
            self._center = pts.mean(axis=0)
        radius = max(float(np.linalg.norm(pts - self._center, axis=1).max()), 2.0)
        if self.width() < 50 or self.height() < 50:
            self._fit_pending = True  # not laid out yet: fit on the first paint
            return
        self._fit_pending = False
        self._zoom = 0.47 * min(self.width(), self.height()) / radius
        self.update()

    def rotate(self, d_yaw: float, d_pitch: float) -> None:
        self._rot = _rotation(0, d_pitch) @ _rotation(1, d_yaw) @ self._rot
        self.update()

    def zoom(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 500.0))
        self.update()

    # ---- projection ---------------------------------------------------------------------

    def _project(self, xyz: np.ndarray) -> tuple:
        p = (np.asarray(xyz, dtype=float).reshape(-1, 3) - self._center) @ self._rot.T
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        return np.column_stack([cx + p[:, 0] * self._zoom, cy - p[:, 1] * self._zoom]), p[:, 2]

    def atom_radius(self, atom: Atom) -> float:
        """Drawn radius in A."""
        if _is_mpmc_dummy_site(atom):
            return _DUMMY_RADIUS
        return 0.28 * atom.element.vdw_radius * self.atom_scale

    def atom_at(self, x: float, y: float) -> Optional[int]:
        """Index of the front-most atom drawn under the point, or None."""
        if not len(self._screen):
            return None
        d = np.hypot(self._screen[:, 0] - x, self._screen[:, 1] - y)
        hit = np.where(d <= np.maximum(self._radii_px, 3.0))[0]
        if not len(hit):
            return None
        return int(hit[np.argmax(self._depth[hit])])

    # ---- colours ------------------------------------------------------------------------

    def atom_color(self, atom: Atom) -> "QtGui.QColor":
        if self.color_by_charge:
            q = float(np.clip(atom.charge, -1.0, 1.0))
            if q >= 0:  # white -> blue for positive, white -> red for negative
                return QtGui.QColor(int(255 - 175 * q), int(255 - 145 * q), 255)
            return QtGui.QColor(255, int(255 + 175 * q), int(255 + 175 * q))
        return QtGui.QColor(*atom.element.cpk_color)

    def _background(self) -> "QtGui.QColor":
        return self.palette().color(QtGui.QPalette.Base)

    def _fog(self, color: "QtGui.QColor", t: float) -> "QtGui.QColor":
        """Blend far objects (t near 0) toward the background."""
        if not self.depth_cue:
            return color
        bg = self._background()
        f = 0.55 * (1.0 - t)
        return QtGui.QColor(
            int(color.red() * (1 - f) + bg.red() * f),
            int(color.green() * (1 - f) + bg.green() * f),
            int(color.blue() * (1 - f) + bg.blue() * f),
        )

    # ---- painting -----------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QtGui.QPainter(self)
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
        screen, depth = self._project(pos)
        radii = np.array([self.atom_radius(a) for a in mol.atoms])
        self._screen, self._depth, self._radii_px = screen, depth, radii * self._zoom
        zmin, zmax = float(depth.min()), float(depth.max())
        span = max(zmax - zmin, 1e-6)

        if self.show_cell and mol.pbc is not None:
            self._paint_cell(painter, mol)
        if self.show_bonds and self._bonds:
            self._paint_bonds(painter, mol, pos, screen, depth, zmin, span)

        outline = QtGui.QPen(QtGui.QColor(0, 0, 0, 90), 0.8)
        for i in np.argsort(depth):  # far to near
            atom = mol.atoms[i]
            r = float(self._radii_px[i])
            c = QtCore.QPointF(screen[i, 0], screen[i, 1])
            base = self._fog(self.atom_color(atom), (depth[i] - zmin) / span)
            if _is_mpmc_dummy_site(atom):
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(base.red(), base.green(), base.blue(), 150))
                painter.drawEllipse(c, max(r, 1.5), max(r, 1.5))
                continue
            g = QtGui.QRadialGradient(c, r, c + QtCore.QPointF(-0.35 * r, -0.4 * r))
            g.setColorAt(0.0, base.lighter(175))
            g.setColorAt(0.45, base)
            g.setColorAt(1.0, base.darker(185))
            painter.setBrush(g)
            painter.setPen(outline if r > 4 else QtCore.Qt.NoPen)
            painter.drawEllipse(c, r, r)
        if self.selected is not None and self.selected < len(mol.atoms):
            i = self.selected
            accent = self.palette().color(QtGui.QPalette.Highlight)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(accent, 2.5))
            r = float(self._radii_px[i]) + 4
            painter.drawEllipse(QtCore.QPointF(screen[i, 0], screen[i, 1]), r, r)
        self._paint_axes(painter, mol)

    def _paint_cell(self, painter, mol) -> None:
        corners, _ = self._project(self.cell_corners())
        color = self.palette().color(QtGui.QPalette.Text)
        color.setAlpha(110)
        pen = QtGui.QPen(color, 1.2, QtCore.Qt.DashLine)
        painter.setPen(pen)
        for i, j in mol.pbc.edges():
            painter.drawLine(QtCore.QPointF(*corners[i]), QtCore.QPointF(*corners[j]))

    def _paint_bonds(self, painter, mol, pos, screen, depth, zmin, span) -> None:
        width = max(1.5, 0.22 * self._zoom * self.atom_scale)
        segments = []  # (depth, p, q, color index)
        for k, (i, j) in enumerate(self._bonds):
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
            color = self._fog(self.atom_color(mol.atoms[idx]), (z - zmin) / span).darker(115)
            painter.setPen(QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(*p), QtCore.QPointF(*q))

    def _paint_axes(self, painter, mol) -> None:
        if mol.pbc is None:
            return
        origin = QtCore.QPointF(42, self.height() - 42)
        colors = [QtGui.QColor(230, 80, 80), QtGui.QColor(90, 190, 90), QtGui.QColor(90, 140, 240)]
        for k, (name, vec) in enumerate(zip("abc", mol.pbc.basis_matrix)):
            d = (vec / np.linalg.norm(vec)) @ self._rot.T
            tip = origin + QtCore.QPointF(26 * d[0], -26 * d[1])
            painter.setPen(QtGui.QPen(colors[k], 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(origin, tip)
            painter.drawText(tip + QtCore.QPointF(3 * np.sign(d[0]) - 3, 4), name)

    # ---- input --------------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = event.position()
        self._drag_button = event.button()
        self._pressed_at = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        p = event.position()
        if self._drag_pos is not None:
            dx, dy = p.x() - self._drag_pos.x(), p.y() - self._drag_pos.y()
            if self._drag_button == QtCore.Qt.LeftButton:
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
            self.selected = self.atom_at(p.x(), p.y())
            if self.on_select is not None:
                self.on_select(self.selected)
            self.update()
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.zoom(1.12 if event.angleDelta().y() > 0 else 1 / 1.12)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        steps = {QtCore.Qt.Key_Left: (-0.1, 0), QtCore.Qt.Key_Right: (0.1, 0),
                 QtCore.Qt.Key_Up: (0, -0.1), QtCore.Qt.Key_Down: (0, 0.1)}
        if key in steps:
            self.rotate(*steps[key])
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.zoom(1.12)
        elif key == QtCore.Qt.Key_Minus:
            self.zoom(1 / 1.12)
        elif key == QtCore.Qt.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        old, new = event.oldSize(), event.size()
        if old.width() > 50 and old.height() > 50 and not self._fit_pending:
            self._zoom *= min(new.width(), new.height()) / max(min(old.width(), old.height()), 1)
        super().resizeEvent(event)


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
# Main window
# ======================================================================

class MainWindow(_QMainWindow):
    """PDB Wizard's desktop window."""

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__()
        self.mol: Optional[Molecule] = None
        self.path: Optional[str] = None
        self._undo: list = []
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self.view = StructureView()
        self.view.on_hover = self._hovered
        self.view.on_select = self._selected
        self.setCentralWidget(self.view)
        self._build_info_dock()
        self._build_tools_dock()
        self._build_table_dock()
        self._build_actions()
        self.statusBar().showMessage("Ready")
        self._update_title()
        if path:
            self.open_path(path)

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
                    "Net charge (e)"):
            label = QtWidgets.QLabel("—")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setWordWrap(True)
            self.info_labels[key] = label
            self.info_form.addRow(f"<b>{key}</b>", label)
        lay.addLayout(self.info_form)
        title = QtWidgets.QLabel("<b>Elements</b>")
        lay.addWidget(title)
        self.legend = QtWidgets.QListWidget()
        self.legend.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.legend, 1)
        self.selection_label = QtWidgets.QLabel("Click an atom to inspect it.")
        self.selection_label.setWordWrap(True)
        lay.addWidget(self.selection_label)
        dock.setWidget(body)
        dock.setMinimumWidth(250)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_tools_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Tools", self)
        dock.setObjectName("tools")
        tools = QtWidgets.QTabWidget()
        tools.setDocumentMode(True)

        # view
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.chk_bonds = QtWidgets.QCheckBox("Bonds")
        self.chk_cell = QtWidgets.QCheckBox("Unit cell")
        self.chk_fog = QtWidgets.QCheckBox("Depth cueing")
        self.chk_charge = QtWidgets.QCheckBox("Colour by charge")
        for chk, attr in ((self.chk_bonds, "show_bonds"), (self.chk_cell, "show_cell"),
                          (self.chk_fog, "depth_cue"), (self.chk_charge, "color_by_charge")):
            chk.setChecked(getattr(self.view, attr))
            chk.toggled.connect(lambda on, a=attr: self._set_view(a, on))
            lay.addWidget(chk)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Atom size"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(40, 200)
        self.size_slider.setValue(100)
        self.size_slider.valueChanged.connect(lambda v: self._set_view("atom_scale", v / 100))
        size_row.addWidget(self.size_slider)
        lay.addLayout(size_row)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.view.reset_view)
        lay.addWidget(reset)
        lay.addStretch(1)
        tools.addTab(page, "View")

        # supercell
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.extend_spins = []
        for axis in "abc":
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 19)
            spin.setToolTip("Extra copies along this axis (1 doubles it, as in the classic menu)")
            self.extend_spins.append(spin)
            form.addRow(f"Extra copies along {axis}", spin)
        btn = QtWidgets.QPushButton("Build supercell")
        btn.clicked.connect(self.build_supercell)
        form.addRow(btn)
        wrap_row = QtWidgets.QHBoxLayout()
        for label, forward in (("Wrap (centred)", False), ("Wrap (0–1)", True)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, f=forward: self.wrap(f))
            wrap_row.addWidget(b)
        form.addRow(wrap_row)
        tools.addTab(page, "Supercell")

        # sorbates
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.sorbate_combo = QtWidgets.QComboBox()
        for name, model in SORBATE_MODELS.items():
            self.sorbate_combo.addItem(f"{model.description}", name)
        self.sorbate_combo.addItem("From an MPMC .pqr file…", "__file__")
        idx = self.sorbate_combo.findData("h2o_dc97")
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
        btn.clicked.connect(self.pack_sorbates)
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
        q_btn.clicked.connect(self.load_charges)
        qeq_btn = QtWidgets.QPushButton("QEq charges")
        qeq_btn.clicked.connect(self.qeq_charges)
        form.addRow(q_btn)
        form.addRow(qeq_btn)
        note = QtWidgets.QLabel("Only framework atoms are changed; packed sorbates keep their "
                                "model parameters.")
        note.setWordWrap(True)
        form.addRow(note)
        tools.addTab(page, "Charges && FF")

        tools.setCurrentIndex(2)
        self.tools = tools
        dock.setWidget(tools)
        dock.setMinimumWidth(290)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

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
        self.table.clicked.connect(lambda idx: self._selected(idx.row(), from_table=True))
        dock.setWidget(self.table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [190], QtCore.Qt.Vertical)

    def _build_actions(self) -> None:
        style = self.style()
        bar = self.addToolBar("Main")
        bar.setObjectName("main")
        bar.setMovable(False)
        menus = {name: self.menuBar().addMenu(name) for name in
                 ("&File", "&Edit", "&View", "&Tools", "&Help")}

        def action(menu, text, slot, shortcut=None, icon=None, toolbar=False):
            act = QtGui.QAction(text, self)
            if icon is not None:
                act.setIcon(style.standardIcon(icon))
            if shortcut:
                act.setShortcut(QtGui.QKeySequence(shortcut))
            act.triggered.connect(slot)
            menus[menu].addAction(act)
            if toolbar:
                bar.addAction(act)
            return act

        sp = QtWidgets.QStyle
        action("&File", "&Open…", self.open_dialog, "Ctrl+O", sp.SP_DialogOpenButton, True)
        action("&File", "Save &as…", self.save_dialog, "Ctrl+S", sp.SP_DialogSaveButton, True)
        action("&File", "Export &image…", self.export_image, "Ctrl+E")
        menus["&File"].addSeparator()
        action("&File", "&Quit", self.close, "Ctrl+Q")
        self.undo_action = action("&Edit", "&Undo", self.undo, "Ctrl+Z", sp.SP_ArrowBack, True)
        self.undo_action.setEnabled(False)
        bar.addSeparator()
        action("&View", "&Reset view", self.view.reset_view, "R", sp.SP_BrowserReload, True)
        action("&View", "Zoom &in", lambda: self.view.zoom(1.2), "Ctrl++")
        action("&View", "Zoom &out", lambda: self.view.zoom(1 / 1.2), "Ctrl+-")
        action("&Tools", "Build &supercell", self.build_supercell)
        action("&Tools", "&Pack sorbates", self.pack_sorbates)
        action("&Tools", "Apply &force field", self.apply_force_field)
        action("&Tools", "Load &charges…", self.load_charges)
        action("&Tools", "&QEq charges", self.qeq_charges)
        action("&Help", "&About PDB Wizard", self.about)

    # ---- state --------------------------------------------------------------------------

    def set_molecule(self, mol: Optional[Molecule], reset_view: bool = True) -> None:
        self.mol = mol
        self.view.set_molecule(mol, reset_view=reset_view)
        self.table_model.set_molecule(mol)
        self._refresh_info()
        self._update_title()

    def _changed(self, message: str) -> None:
        """After an edit: redraw everything and say what happened."""
        self.view.refresh_bonds()
        self.view.update()
        self.table_model.set_molecule(self.mol)
        self._refresh_info()
        self.statusBar().showMessage(message, 8000)

    def _checkpoint(self) -> None:
        self._undo.append(copy.deepcopy(self.mol))
        del self._undo[:-20]
        self.undo_action.setEnabled(True)

    def undo(self) -> None:
        if not self._undo:
            return
        self.set_molecule(self._undo.pop(), reset_view=False)
        self.undo_action.setEnabled(bool(self._undo))
        self.statusBar().showMessage("Undone", 4000)

    def _need_structure(self, what: str, cell: bool = False) -> bool:
        if self.mol is None or not self.mol.atoms:
            self._warn(f"{what} needs a structure. Open one first.")
            return False
        if cell and self.mol.pbc is None:
            self._warn(f"{what} needs a unit cell, and this structure has none.")
            return False
        return True

    def _set_view(self, attr: str, value) -> None:
        setattr(self.view, attr, value)
        self.view.update()

    def _update_title(self) -> None:
        name = Path(self.path).name if self.path else "no structure"
        self.setWindowTitle(f"PDB Wizard — {name}")

    def _refresh_info(self) -> None:
        labels = self.info_labels
        self.legend.clear()
        if self.mol is None:
            for label in labels.values():
                label.setText("—")
            return
        info = system_info(self.mol, self.path or "")
        labels["File"].setText(Path(self.path).name if self.path else "(unsaved)")
        counts: dict = {}
        for a in self.mol.atoms:
            if not _is_mpmc_dummy_site(a):
                counts[a.element.symbol] = counts.get(a.element.symbol, 0) + 1
        labels["Formula"].setText(" ".join(f"{el}<sub>{n}</sub>" for el, n in counts.items()))
        labels["Atoms"].setText(str(info["n_atoms"]))
        labels["Sorbates"].setText(str(mpmc_movable_count(self.mol)))
        labels["Net charge (e)"].setText(f"{sum(a.charge for a in self.mol.atoms):+.4f}")
        if "cell" in info:
            c = info["cell"]
            labels["a, b, c (Å)"].setText(f"{c['a']:.3f}, {c['b']:.3f}, {c['c']:.3f}")
            labels["α, β, γ (°)"].setText(
                f"{c['alpha']:.2f}, {c['beta']:.2f}, {c['gamma']:.2f}")
            labels["Volume (Å³)"].setText(f"{info['volume']:.2f}")
            labels["Density (g/cm³)"].setText(f"{info['density']:.4f}")
        else:
            for key in ("a, b, c (Å)", "α, β, γ (°)", "Volume (Å³)",
                        "Density (g/cm³)"):
                labels[key].setText("no cell")
        counts: dict = {}
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

    def _describe(self, idx: int) -> str:
        a = self.mol.atoms[idx]
        where = f"sorbate {a.mpmc_mol}" if a.mpmc_flag == "M" else "framework"
        return (f"#{idx + 1} {a.name} ({a.element.symbol}, {where})  "
                f"x={a.x[0]:.3f} y={a.x[1]:.3f} z={a.x[2]:.3f}  q={a.charge:+.4f} e  "
                f"ε={a.epsilon:.2f} K  σ={a.sigma:.3f} Å")

    def _warn(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "PDB Wizard", text)

    # ---- files --------------------------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open structure", "", _OPEN_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        messages: list = []
        try:
            mol = read_file(path, on_message=messages.append)
        except Exception as e:  # any reader error goes to the user, not the console
            self._warn(f"Could not read {Path(path).name}:\n{e}")
            return False
        self.path = path
        self._undo.clear()
        self.undo_action.setEnabled(False)
        self.set_molecule(mol)
        note = f" ({messages[0]})" if messages else ""
        self.statusBar().showMessage(
            f"Opened {Path(path).name}: {len(mol.atoms)} atoms{note}", 8000)
        return True

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
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export image", "structure.png",
                                                            "PNG (*.png)")
        if not path:
            return None
        self.view.grab().save(path)
        self.statusBar().showMessage(f"Saved image {path}", 6000)
        return path

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    # ---- tools --------------------------------------------------------------------------

    def build_supercell(self) -> None:
        if not self._need_structure("A supercell", cell=True):
            return
        counts = [spin.value() for spin in self.extend_spins]
        if not any(counts):
            self.statusBar().showMessage("Set at least one axis above 0", 5000)
            return
        self._checkpoint()
        for axis, n in enumerate(counts):
            if n:
                extend_axis(self.mol, axis, n)
        self.view.reset_view()
        size = "×".join(str(n + 1) for n in counts)
        self._changed(f"Built a {size} supercell: {len(self.mol.atoms)} atoms")

    def wrap(self, forward: bool) -> None:
        if not self._need_structure("Wrapping", cell=True):
            return
        self._checkpoint()
        wrap_atoms(self.mol, forward=forward)
        self._changed("Wrapped atoms into " + ("0–1" if forward else "−½–½"))

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
        first_id = 2 + mpmc_movable_count(self.mol)
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
            self._undo.pop()
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
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._changed("QEq charges assigned to the framework")

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
