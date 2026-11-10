"""Desktop GUI tools: plots, background tasks, analysis windows, edit dialogs, the input
generator, database search, the geometry tables and the trajectory bar.

Every computation goes through the same shared modules as the terminal viewer
(``analyses``, ``inputgen``, ``database``, ``trajectory``), so the two front ends offer the
same features and produce the same numbers. Like gui.py, this imports without PySide6.
"""

from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from pdb_wizard.analyses import (
    ALL_ATOMS,
    PXRD_SOURCES,
    AnalysisResult,
    elements_of,
    energy_columns,
    energy_n_steps,
    names_of,
    read_experimental_pxrd,
    run_coordination,
    run_density_profile,
    run_energy,
    run_gyration,
    run_hbonds,
    run_isotherm,
    run_msd,
    run_pore_size,
    run_pxrd,
    run_rdf,
    run_rmsd,
    write_csv,
    write_energy_csv,
)
from pdb_wizard.database import BACKENDS, backends_for, modes_for, placeholder_for, search_backends
from pdb_wizard.geometry import get_close_contacts, get_lone_atoms
from pdb_wizard.inputgen import (
    check_engine,
    default_engine_outdir,
    engine_template_text,
    generate_engine_inputs,
    parse_form_value,
)
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.reduce_cell import identify_framework_and_sorbates, reduce_to_primitive
from pdb_wizard.templates import ENGINES

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    _HAS_QT_TOOLS = True
except ImportError:  # optional extra, see gui.py
    _HAS_QT_TOOLS = False

if _HAS_QT_TOOLS:
    _QWidgetT = QtWidgets.QWidget
    _QDialogT = QtWidgets.QDialog
else:  # keeps the module importable without Qt
    _QWidgetT = _QDialogT = object

_PLOT_COLORS = [(61, 174, 233), (255, 130, 60)]


def _mono_font():
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)


# ======================================================================
# Background work
# ======================================================================

class GuiTask:
    """Run ``fn`` on a worker thread; ``on_done`` / ``on_error`` run on the GUI thread.

    A QTimer polls the future, so no Qt object crosses threads. ``wait()`` blocks until the
    task has finished and its callback has run (for scripts and tests).
    """

    _pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    _live: set = set()  # running tasks stay referenced, so a second task never drops the first

    def __init__(self, fn: Callable, on_done: Callable, on_error: Callable) -> None:
        self._future = self._pool.submit(fn)
        self._on_done, self._on_error = on_done, on_error
        self.finished = False
        GuiTask._live.add(self)
        self._timer = QtCore.QTimer()
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def detach(self) -> None:
        """Drop the callbacks (their window is closing); the work finishes unseen."""
        self._on_done = self._on_error = lambda *_: None

    def _poll(self) -> None:
        if self.finished or not self._future.done():
            return
        self._timer.stop()
        self.finished = True
        GuiTask._live.discard(self)
        try:
            result = self._future.result()
        except Exception as e:  # reported to the user by on_error
            callback, arg = self._on_error, e
        else:
            callback, arg = self._on_done, result
        try:
            callback(arg)
        except RuntimeError as e:
            if "already deleted" not in str(e):  # its window was destroyed meanwhile: fine
                raise

    def wait(self, timeout: float = 600.0) -> None:
        concurrent.futures.wait([self._future], timeout=timeout)
        self._poll()


# ======================================================================
# Plot
# ======================================================================

def _nice_ticks(lo: float, hi: float, n: int = 6) -> List[float]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), raw)
    start = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(start, hi + step * 1e-3, step)]


class QtPlot(_QWidgetT):
    """Line plot drawn with QPainter, like the terminal viewer's plot.

    The wheel (or +/-) zooms the x axis around the cursor, a double click (or r) resets
    it, and hovering shows the nearest point. One overlay curve can be added (e.g. an
    experimental PXRD pattern).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.x = np.array([])
        self.y = np.array([])
        self.overlay: Optional[tuple] = None
        self.title = self.x_label = ""
        self.y_min: Optional[float] = 0.0
        self.y_max: Optional[float] = None
        self.x_range: Optional[tuple] = None  # zoom window, None = everything
        self._hover: Optional[int] = None
        self._frame = QtCore.QRectF()

    def set_data(self, x, y, title: str = "", x_label: str = "",
                 y_min: Optional[float] = 0.0, y_max: Optional[float] = None) -> None:
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.title, self.x_label = title, x_label
        self.y_min, self.y_max = y_min, y_max
        self.x_range = None
        self.overlay = None
        self._hover = None
        self.update()

    def set_result(self, res: AnalysisResult) -> None:
        self.set_data(res.x, res.y, res.title, res.x_label, res.y_min, res.y_max)

    def set_overlay(self, x, y, label: str = "") -> None:
        self.overlay = (np.asarray(x, float), np.asarray(y, float), label)
        self.update()

    def clear(self) -> None:
        self.set_data([], [])

    # ---- geometry -----------------------------------------------------------------------

    def limits(self) -> tuple:
        xs = [a for a in (self.x, self.overlay[0] if self.overlay else None)
              if a is not None and len(a)]
        if not xs:
            return 0.0, 1.0, 0.0, 1.0
        x0 = min(float(np.nanmin(a)) for a in xs)
        x1 = max(float(np.nanmax(a)) for a in xs)
        if self.x_range is not None:
            x0, x1 = self.x_range
        ys = [a for a in (self.y, self.overlay[1] if self.overlay else None)
              if a is not None and len(a)]
        y0 = self.y_min if self.y_min is not None else min(float(np.nanmin(a)) for a in ys)
        y1 = self.y_max if self.y_max is not None else max(float(np.nanmax(a)) for a in ys)
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 <= y0:
            y1 = y0 + 1.0
        return x0, x1, y0, y1

    def zoom_x(self, factor: float, center: Optional[float] = None) -> None:
        if not len(self.x):
            return
        x0, x1, _, _ = self.limits()
        c = (x0 + x1) / 2 if center is None else center
        lo, hi = c - (c - x0) / factor, c + (x1 - c) / factor
        xs = [self.x] + ([self.overlay[0]] if self.overlay is not None else [])
        full0 = min(float(np.nanmin(a)) for a in xs if len(a))
        full1 = max(float(np.nanmax(a)) for a in xs if len(a))
        self.x_range = None if hi - lo >= full1 - full0 else (max(lo, full0), min(hi, full1))
        self.update()

    def reset_zoom(self) -> None:
        self.x_range = None
        self.update()

    # ---- painting -----------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        p = QtGui.QPainter(self)
        try:  # an exception with the painter still active crashes Qt on the next paint
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p) -> None:
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QtGui.QPalette.Base))
        text = pal.color(QtGui.QPalette.Text)
        fm = p.fontMetrics()
        left, right, top, bottom = 72, 18, 30, 46
        frame = QtCore.QRectF(left, top, max(self.width() - left - right, 10),
                              max(self.height() - top - bottom, 10))
        self._frame = frame
        x0, x1, y0, y1 = self.limits()

        def px(x, y):
            return (frame.left() + (x - x0) / (x1 - x0) * frame.width(),
                    frame.bottom() - (y - y0) / (y1 - y0) * frame.height())

        grid = QtGui.QColor(text)
        grid.setAlpha(38)
        xt, yt = _nice_ticks(x0, x1), _nice_ticks(y0, y1)
        p.setPen(QtGui.QPen(grid, 1))
        for t in xt:
            xx = px(t, y0)[0]
            p.drawLine(QtCore.QPointF(xx, frame.top()), QtCore.QPointF(xx, frame.bottom()))
        for t in yt:
            yy = px(x0, t)[1]
            p.drawLine(QtCore.QPointF(frame.left(), yy), QtCore.QPointF(frame.right(), yy))
        p.setPen(QtGui.QPen(text, 1))
        p.drawRect(frame)
        for t in xt:
            label = f"{t:g}"
            p.drawText(QtCore.QPointF(px(t, y0)[0] - fm.horizontalAdvance(label) / 2,
                                      frame.bottom() + 16), label)
        for t in yt:
            label = f"{t:.4g}"
            p.drawText(QtCore.QPointF(frame.left() - 8 - fm.horizontalAdvance(label),
                                      px(x0, t)[1] + 4), label)
        p.drawText(QtCore.QRectF(0, 6, self.width(), 20), QtCore.Qt.AlignCenter, self.title)
        p.drawText(QtCore.QRectF(left, self.height() - 22, frame.width(), 18),
                   QtCore.Qt.AlignCenter, self.x_label)
        if not len(self.x):
            p.drawText(frame, QtCore.Qt.AlignCenter, "Set the options and press Compute")
            return

        p.setClipRect(frame)
        curves = [(self.x, self.y, QtGui.QColor(*_PLOT_COLORS[0]))]
        if self.overlay is not None:
            curves.append((self.overlay[0], self.overlay[1], QtGui.QColor(*_PLOT_COLORS[1])))
        for xs, ys, color in curves:
            path = QtGui.QPainterPath()
            for i, (xi, yi) in enumerate(zip(xs, ys)):
                a, b = px(xi, yi)
                if i == 0:
                    path.moveTo(a, b)
                else:
                    path.lineTo(a, b)
            p.setPen(QtGui.QPen(color, 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPath(path)
            if len(xs) == 1:
                p.setBrush(color)
                p.drawEllipse(QtCore.QPointF(*px(xs[0], ys[0])), 4, 4)
        p.setClipping(False)
        if self.overlay is not None and self.overlay[2]:
            p.setPen(QtGui.QColor(*_PLOT_COLORS[1]))
            p.drawText(QtCore.QPointF(frame.right() - 8 - fm.horizontalAdvance(self.overlay[2]),
                                      frame.top() + 16), self.overlay[2])
        if self._hover is not None and self._hover < len(self.x):
            i = self._hover
            hx, _ = px(self.x[i], self.y[i])
            p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Highlight), 1, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(hx, frame.top()), QtCore.QPointF(hx, frame.bottom()))
            p.setPen(text)
            p.drawText(QtCore.QPointF(frame.left() + 6, frame.top() + 16),
                       f"x = {self.x[i]:.4g}   y = {self.y[i]:.4g}")

    # ---- input --------------------------------------------------------------------------

    def _x_at(self, pos) -> float:
        x0, x1, _, _ = self.limits()
        return x0 + (pos.x() - self._frame.left()) / max(self._frame.width(), 1) * (x1 - x0)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if not len(self.x) or not self._frame.contains(pos):
            self._hover = None
        else:
            dist = np.abs(self.x - self._x_at(pos))
            self._hover = int(np.nanargmin(dist)) if np.isfinite(dist).any() else None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.zoom_x(factor, self._x_at(event.position()))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_zoom()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.zoom_x(1.25)
        elif key == QtCore.Qt.Key_Minus:
            self.zoom_x(1 / 1.25)
        elif key == QtCore.Qt.Key_R:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)


# ======================================================================
# Analysis windows
# ======================================================================

@dataclass
class Field:
    """One option of an analysis window.

    kind: 'element' (default 'first' or 'last'), 'element_all', 'name:<element field>',
    'int', 'float', 'choice' (``choices`` = ((label, value), ...)) or 'dir'.
    """

    key: str
    label: str
    kind: str
    default: object = None
    lo: float = 0
    hi: float = 0
    step: float = 1
    choices: tuple = ()


@dataclass
class AnalysisSpec:
    title: str
    fields: list
    run: Callable  # (mol, frames, params, progress) -> AnalysisResult
    frame_range: bool = False  # Start/End boxes for trajectories
    traj_default: Optional[bool] = None  # a "Use trajectory" box with this default


def _rng(p: Dict) -> tuple:
    return p.get("start", 1), p.get("end")


ANALYSES: Dict[str, AnalysisSpec] = {
    "rdf": AnalysisSpec(
        "Radial Distribution",
        [Field("el1", "Type 1 element", "element", "first"),
         Field("name1", "Type 1 name", "name:el1", ALL_ATOMS),
         Field("el2", "Type 2 element", "element", "last"),
         Field("name2", "Type 2 name", "name:el2", ALL_ATOMS),
         Field("xmax", "r max (Å)", "int", 13, 2, 50),
         Field("bins", "Bins", "int", 100, 10, 1000, 10),
         Field("ymax", "Y max (display)", "int", 5, 1, 100)],
        lambda mol, fr, p, prog: run_rdf(mol, p["el1"], p["el2"], p["name1"], p["name2"],
                                         float(p["xmax"]), p["bins"], fr, *_rng(p),
                                         p.get("traj", False), prog),
        frame_range=True, traj_default=True),
    "coordination": AnalysisSpec(
        "Coordination Number",
        [Field("center", "Center", "element", "first"),
         Field("neighbor", "Neighbor", "element", "last"),
         Field("rmax", "r max (Å)", "int", 5, 2, 20)],
        lambda mol, fr, p, prog: run_coordination(mol, p["center"], p["neighbor"],
                                                  float(p["rmax"]), fr, *_rng(p),
                                                  p.get("traj", False)),
        frame_range=True, traj_default=False),
    "hbonds": AnalysisSpec(
        "Hydrogen Bonds",
        [Field("cutoff", "D–A cutoff (Å)", "float", 3.5, 2.0, 5.0, 0.1),
         Field("angle", "Min D–H···A angle (°)", "int", 120, 90, 170)],
        lambda mol, fr, p, prog: run_hbonds(mol, p["cutoff"], float(p["angle"]), fr, *_rng(p)),
        frame_range=True),
    "msd": AnalysisSpec(
        "Mean Squared Displacement",
        [Field("element", "Element", "element", "last"),
         Field("name", "Name", "name:element", ALL_ATOMS),
         Field("dt", "dt (ps per frame)", "int", 1, 1, 10000)],
        lambda mol, fr, p, prog: run_msd(fr, p["element"], p["name"], float(p["dt"]), *_rng(p)),
        frame_range=True),
    "rmsd": AnalysisSpec(
        "RMSD vs Time",
        [Field("element", "Element", "element_all", ALL_ATOMS),
         Field("name", "Name", "name:element", ALL_ATOMS),
         Field("dt", "dt (ps per frame)", "int", 1, 1, 10000)],
        lambda mol, fr, p, prog: run_rmsd(fr, p["element"], p["name"], float(p["dt"]),
                                          *_rng(p)),
        frame_range=True),
    "gyration": AnalysisSpec(
        "Radius of Gyration",
        [Field("element", "Element", "element_all", ALL_ATOMS),
         Field("dt", "dt (ps per frame)", "int", 1, 1, 10000)],
        lambda mol, fr, p, prog: run_gyration(fr, p["element"], float(p["dt"]), *_rng(p)),
        frame_range=True),
    "density": AnalysisSpec(
        "Density Profile",
        [Field("element", "Element", "element", "first"),
         Field("name", "Name", "name:element", ALL_ATOMS),
         Field("axis", "Axis", "choice", 2, choices=(("a (x)", 0), ("b (y)", 1), ("c (z)", 2)))],
        lambda mol, fr, p, prog: run_density_profile(mol, p["element"], p["name"], p["axis"],
                                                     fr, *_rng(p), p.get("traj", False)),
        frame_range=True, traj_default=True),
    "pxrd": AnalysisSpec(
        "Powder XRD",
        [Field("wavelength", "Source", "choice", 1.5406, choices=PXRD_SOURCES),
         Field("tt_max", "2θ max (°)", "int", 50, 10, 90),
         Field("width", "Peak width (0.01°)", "int", 10, 1, 50)],
        lambda mol, fr, p, prog: run_pxrd(mol, p["wavelength"], float(p["tt_max"]),
                                          p["width"] / 100.0, prog)),
    "pore_size": AnalysisSpec(
        "Pore Size Distribution", [],
        lambda mol, fr, p, prog: run_pore_size(mol, 50000, 50, 20.0)),
    "isotherm_plot": AnalysisSpec(
        "Isotherm Results", [Field("directory", "Isotherm folder", "dir", "isotherm")],
        lambda mol, fr, p, prog: run_isotherm(p["directory"])),
}


def _couple_range(start, end) -> None:
    """Keep a Start spin box at or below its End box, so a frame range is never empty."""
    start.valueChanged.connect(end.setMinimum)
    end.valueChanged.connect(start.setMaximum)
    end.setMinimum(start.value())
    start.setMaximum(end.value())


class AnalysisDialog(_QDialogT):
    """Options on top, Compute / CSV / Save-plot buttons, the plot and a status line.

    The same windows as the terminal viewer's analysis screens, driven by ``ANALYSES``.
    """

    def __init__(self, key: str, mol: Optional[Molecule], frames: Optional[list] = None,
                 parent=None) -> None:
        super().__init__(parent)
        spec = ANALYSES[key]
        self.key, self.spec, self.mol, self.frames = key, spec, mol, frames
        self.doc = None  # the window's document; MainWindow keeps mol/frames in step with it
        self._overlay: Optional[tuple] = None
        self.result: Optional[AnalysisResult] = None
        self.task: Optional[GuiTask] = None
        self._progress = [0.0]
        self.setWindowTitle(spec.title)
        self.resize(940, 600)
        lay = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        self.inputs: Dict[str, QtWidgets.QWidget] = {}
        cells = []
        for f in spec.fields:
            w = self._make_input(f)
            self.inputs[f.key] = w
            cells.append((f.label, w))
        n_frames = len(frames) if frames else 0
        self.traj_box = None
        if n_frames > 1 and spec.traj_default is not None:
            self.traj_box = QtWidgets.QCheckBox("Use trajectory")
            self.traj_box.setChecked(spec.traj_default)
            cells.append(("", self.traj_box))
        self.start_spin = self.end_spin = None
        if n_frames > 1 and spec.frame_range:
            self.start_spin, self.end_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
            for spin, value in ((self.start_spin, 1), (self.end_spin, n_frames)):
                spin.setRange(1, n_frames)
                spin.setValue(value)
            _couple_range(self.start_spin, self.end_spin)
            cells += [("Start frame", self.start_spin), ("End frame", self.end_spin)]
        for k, (label, w) in enumerate(cells):
            row, col = divmod(k, 3)
            if label:
                grid.addWidget(QtWidgets.QLabel(label), row, 2 * col)
            grid.addWidget(w, row, 2 * col + 1)
        lay.addLayout(grid)
        buttons = QtWidgets.QHBoxLayout()
        self.compute_button = QtWidgets.QPushButton("Compute")
        self.compute_button.setDefault(True)
        self.compute_button.clicked.connect(self.run)
        self.csv_button = QtWidgets.QPushButton("CSV…")
        self.csv_button.clicked.connect(lambda: self.export_csv())
        self.png_button = QtWidgets.QPushButton("Save plot…")
        self.png_button.clicked.connect(lambda: self.export_png())
        buttons.addWidget(self.compute_button)
        if key == "pxrd":
            self.exp_button = QtWidgets.QPushButton("Load experimental…")
            self.exp_button.clicked.connect(lambda: self.load_experimental())
            buttons.addWidget(self.exp_button)
        buttons.addStretch(1)
        buttons.addWidget(self.csv_button)
        buttons.addWidget(self.png_button)
        lay.addLayout(buttons)
        self.plot = QtPlot()
        lay.addWidget(self.plot, 1)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(_mono_font())
        self.text.setMaximumHeight(140)
        self.text.hide()
        lay.addWidget(self.text)
        self.status = QtWidgets.QLabel("Set the options and press Compute")
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        self._progress_timer = QtCore.QTimer(self)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(
            lambda: self.status.setText(f"Computing… {int(self._progress[0] * 100)}%"))
        if "ymax" in self.inputs:
            self.inputs["ymax"].valueChanged.connect(lambda _: self._apply_ymax())
        self._set_export_enabled(False)

    # ---- inputs -------------------------------------------------------------------------

    def _make_input(self, f: Field):
        mol = self.mol
        if f.kind in ("element", "element_all") or f.kind.startswith("name:"):
            w = QtWidgets.QComboBox()
            if f.kind.startswith("name:"):
                el_box = self.inputs[f.kind.split(":", 1)[1]]
                w.addItems([ALL_ATOMS] + (names_of(mol, el_box.currentText()) if mol else []))
                el_box.currentTextChanged.connect(lambda el, box=w: self._refill_names(box, el))
            else:
                items = elements_of(mol) if mol is not None else []
                if f.kind == "element_all":
                    items = [ALL_ATOMS] + items
                w.addItems(items or ["X"])
                if f.default == "last":
                    w.setCurrentIndex(w.count() - 1)
                elif f.default not in ("first", None) and f.default in items:
                    w.setCurrentText(f.default)
            return w
        if f.kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(f.lo), int(f.hi))
            w.setSingleStep(int(f.step))
            w.setValue(int(f.default))
            return w
        if f.kind == "float":
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(f.lo, f.hi)
            w.setSingleStep(f.step)
            w.setDecimals(2)
            w.setValue(float(f.default))
            return w
        if f.kind == "choice":
            w = QtWidgets.QComboBox()
            for label, value in f.choices:
                w.addItem(label, value)
            w.setCurrentIndex(max(w.findData(f.default), 0))
            return w
        w = QtWidgets.QLineEdit(str(f.default or ""))  # 'dir'
        act = w.addAction(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon),
                          QtWidgets.QLineEdit.TrailingPosition)
        act.triggered.connect(lambda _=False, edit=w: self._browse_dir(edit))
        return w

    def _refill_names(self, box, element: str) -> None:
        keep = box.currentText()
        box.clear()
        box.addItems([ALL_ATOMS] + (names_of(self.mol, element) if self.mol is not None else []))
        if box.findText(keep) >= 0:
            box.setCurrentText(keep)

    def set_structure(self, mol: Optional[Molecule], frames: Optional[list]) -> None:
        """Follow the document after an edit, undo or frame change: new atoms, same options."""
        self.mol, self.frames = mol, frames
        elements = elements_of(mol) if mol is not None else []
        for f in self.spec.fields:
            if f.kind not in ("element", "element_all"):
                continue
            box = self.inputs[f.key]
            keep = box.currentText()
            items = ([ALL_ATOMS] if f.kind == "element_all" else []) + elements
            box.blockSignals(True)
            box.clear()
            box.addItems(items or ["X"])
            if box.findText(keep) >= 0:
                box.setCurrentText(keep)
            elif f.default == "last":
                box.setCurrentIndex(box.count() - 1)
            box.blockSignals(False)
            box.currentTextChanged.emit(box.currentText())  # refresh linked name boxes
        n = len(frames) if frames else 0
        if self.end_spin is not None and n > 1:
            for spin in (self.start_spin, self.end_spin):
                spin.setMaximum(n)

    def _snapshot(self) -> tuple:
        """Copies the worker can read while edits change the live structure. Only the
        frame being shown can be edited, so the other frames are shared, not copied."""
        mol = copy.deepcopy(self.mol) if self.mol is not None else None
        frames = self.frames
        if frames:
            frames = [mol if f is self.mol else f for f in frames]
        return mol, frames

    def _browse_dir(self, edit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder", edit.text())
        if path:
            edit.setText(path)

    def params(self) -> Dict:
        out: Dict = {}
        for f in self.spec.fields:
            w = self.inputs[f.key]
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                out[f.key] = w.value()
            elif isinstance(w, QtWidgets.QComboBox):
                out[f.key] = w.currentData() if f.kind == "choice" else w.currentText()
            else:
                out[f.key] = w.text().strip()
        out["traj"] = bool(self.traj_box is not None and self.traj_box.isChecked())
        if self.start_spin is not None:
            out["start"], out["end"] = self.start_spin.value(), self.end_spin.value()
        return out

    # ---- running ------------------------------------------------------------------------

    def run(self) -> GuiTask:
        params = self.params()
        mol, frames = self._snapshot()
        fn = self.spec.run
        self._progress[0] = 0.0
        self.compute_button.setEnabled(False)
        self.status.setText("Computing…")
        self._progress_timer.start()

        def progress(frac: float) -> None:
            self._progress[0] = frac

        self.task = GuiTask(lambda: fn(mol, frames, params, progress), self._done, self._failed)
        return self.task

    def _done(self, res: AnalysisResult) -> None:
        self._progress_timer.stop()
        self.compute_button.setEnabled(True)
        self.result = res
        self.status.setText(res.summary)
        self.text.setPlainText(res.text)
        self.text.setVisible(bool(res.text))
        if res.empty:
            self.plot.clear()
            self._set_export_enabled(False)
            return
        self.plot.set_result(res)
        if self._overlay is not None:  # keep a loaded experimental pattern across recomputes
            self.plot.set_overlay(*self._overlay)
        self._apply_ymax()
        self._set_export_enabled(True)

    def _failed(self, error: Exception) -> None:
        self._progress_timer.stop()
        self.compute_button.setEnabled(True)
        self.status.setText(f"Error: {error}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.task is not None:
            self.task.detach()
        self._progress_timer.stop()
        super().closeEvent(event)

    def _apply_ymax(self) -> None:
        if "ymax" in self.inputs and len(self.plot.x):
            self.plot.y_max = float(self.inputs["ymax"].value())
            self.plot.update()

    def _set_export_enabled(self, on: bool) -> None:
        self.csv_button.setEnabled(on)
        self.png_button.setEnabled(on)

    # ---- export -------------------------------------------------------------------------

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        if self.result is None or self.result.empty:
            return None
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save CSV", f"{self.key}.csv", "CSV (*.csv)")
        if not path:
            return None
        try:
            write_csv(path, self.result.csv_header, *self.result.csv_arrays())
        except OSError as e:
            self.status.setText(f"Could not write {path}: {e}")
            return None
        self.status.setText(f"Wrote {path}")
        return path

    def export_png(self, path: Optional[str] = None) -> Optional[str]:
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save plot", f"{self.key}.png", "PNG (*.png)")
        if not path:
            return None
        if not self.plot.grab().save(path):
            self.status.setText(f"Could not write {path}")
            return None
        self.status.setText(f"Wrote {path}")
        return path

    def load_experimental(self, path: Optional[str] = None) -> bool:
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Experimental PXRD (2θ, intensity)", "",
                "Patterns (*.xy *.csv *.dat *.txt);;All files (*)")
        if not path:
            return False
        try:
            tt, inten = read_experimental_pxrd(path)
        except Exception as e:  # malformed file
            self.status.setText(f"Error loading: {e}")
            return False
        self._overlay = (tt, inten, "experimental")
        self.plot.set_overlay(*self._overlay)
        self.status.setText(f"Experimental data loaded ({len(tt)} points) — "
                            "blue = simulated, orange = experimental")
        return True


class EnergyDialog(_QDialogT):
    """MPMC energy.dat plot: pick a column and a step range; CSV writes every column."""

    def __init__(self, data: Dict, path: str, parent=None) -> None:
        super().__init__(parent)
        self.data, self.path = data, path
        self.result: Optional[AnalysisResult] = None
        self.setWindowTitle("Energy Plot")
        self.resize(900, 560)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        cols, default = energy_columns(data)
        self.column = QtWidgets.QComboBox()
        self.column.addItems(cols)
        self.column.setCurrentText(default)
        n = energy_n_steps(data)
        self.start, self.end = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin, value in ((self.start, 1), (self.end, n)):
            spin.setRange(1, max(n, 1))
            spin.setValue(value)
        _couple_range(self.start, self.end)
        for label, w in (("Column", self.column), ("Start", self.start), ("End", self.end)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(w)
        row.addStretch(1)
        self.csv_button = QtWidgets.QPushButton("CSV…")
        self.csv_button.clicked.connect(lambda: self.export_csv())
        row.addWidget(self.csv_button)
        lay.addLayout(row)
        self.plot = QtPlot()
        lay.addWidget(self.plot, 1)
        self.status = QtWidgets.QLabel(f"Loaded {path}")
        lay.addWidget(self.status)
        self.column.currentTextChanged.connect(lambda _: self.replot())
        self.start.valueChanged.connect(lambda _: self.replot())
        self.end.valueChanged.connect(lambda _: self.replot())
        self.replot()

    def replot(self) -> None:
        res = run_energy(self.data, self.column.currentText(), self.start.value(),
                         self.end.value())
        self.result = res
        self.status.setText(res.summary)
        if res.empty:
            self.plot.clear()
        else:
            self.plot.set_result(res)

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        if self.result is None or self.result.empty:
            return None
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "energy.csv",
                                                            "CSV (*.csv)")
        if not path:
            return None
        sl = self.result.extra["slice"]
        try:
            write_energy_csv(self.data, path, sl)
        except OSError as e:
            self.status.setText(f"Error: {e}")
            return None
        self.status.setText(f"Exported steps {sl.start + 1}-{sl.stop} to {path}")
        return path


class TextReport(_QDialogT):
    """A read-only monospace report (System Info, Void Volume, QEq summary, ...)."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(580, 400)
        lay = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit(text)
        self.text.setReadOnly(True)
        self.text.setFont(_mono_font())
        lay.addWidget(self.text)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        copy = buttons.addButton("Copy", QtWidgets.QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text))
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


# ======================================================================
# Edit dialogs
# ======================================================================

class CellDialog(_QDialogT):
    """Update the cell (a b c α β γ) or remove it."""

    def __init__(self, pbc, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Update Unit Cell")
        form = QtWidgets.QFormLayout(self)
        values = ((pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma) if pbc is not None
                  else (10.0, 10.0, 10.0, 90.0, 90.0, 90.0))
        labels = ("a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)")
        self.spins = []
        self._original = tuple(float(v) for v in values)
        self._edited = [False] * 6
        for k, (label, v) in enumerate(zip(labels, values)):
            spin = QtWidgets.QDoubleSpinBox()
            if k < 3:
                spin.setRange(0.0001, 1.0e6)
            else:
                spin.setRange(0.001, 179.999)
            spin.setDecimals(6)
            spin.setValue(v)
            spin.valueChanged.connect(lambda _v, i=k: self._edited.__setitem__(i, True))
            self.spins.append(spin)
            form.addRow(label, spin)
        self.remove = QtWidgets.QCheckBox("Remove the cell (no periodic boundaries)")
        self.remove.setEnabled(pbc is not None)
        self.remove.toggled.connect(lambda on: [s.setEnabled(not on) for s in self.spins])
        form.addRow(self.remove)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Optional[tuple]:
        """(a, b, c, alpha, beta, gamma), or None to remove the cell."""
        if self.remove.isChecked():
            return None
        # untouched fields return the cell's own values, not the spin boxes' rounding
        return tuple(s.value() if edited else orig
                     for s, edited, orig in zip(self.spins, self._edited, self._original))


class ReduceCellDialog(_QDialogT):
    """Reduce a supercell: set its size, Analyze, untick sorbates to drop, Reduce."""

    def __init__(self, mol: Molecule, parent=None) -> None:
        super().__init__(parent)
        self.mol = mol
        self.result_mol: Optional[Molecule] = None
        self.framework: list = []
        self.sorbates: list = []
        self.setWindowTitle("Reduce Supercell")
        self.resize(660, 460)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Supercell:"))
        self.spins = []
        for k in range(3):
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 8)
            spin.setValue(2)
            spin.valueChanged.connect(lambda _v: self._sizes_changed())
            self.spins.append(spin)
            row.addWidget(spin)
            if k < 2:
                row.addWidget(QtWidgets.QLabel("×"))
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        self.reduce_button = QtWidgets.QPushButton("Reduce")
        self.reduce_button.clicked.connect(self.reduce)
        self.reduce_button.setEnabled(False)
        row.addStretch(1)
        row.addWidget(self.analyze_button)
        row.addWidget(self.reduce_button)
        lay.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Keep", "#", "Atoms", "Elements", "Center (Å)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        self.status = QtWidgets.QLabel("Set the supercell dimensions and click Analyze")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def sizes(self) -> tuple:
        return tuple(s.value() for s in self.spins)

    def _sizes_changed(self) -> None:
        if self.reduce_button.isEnabled():  # the analysis was for another size
            self.reduce_button.setEnabled(False)
            self.status.setText("Supercell size changed: click Analyze again")

    def analyze(self) -> None:
        na, nb, nc = self.sizes()
        if self.mol.pbc is None:
            self.status.setText("Reducing a supercell needs a periodic cell")
            self.reduce_button.setEnabled(False)
            return
        self.framework, self.sorbates = identify_framework_and_sorbates(self.mol, na, nb, nc)
        self.table.setRowCount(len(self.sorbates))
        for i, idx in enumerate(self.sorbates):
            counts: dict = {}
            for j in idx:
                el = self.mol.atoms[j].element.symbol
                counts[el] = counts.get(el, 0) + 1
            center = np.mean([self.mol.atoms[j].x for j in idx], axis=0) if idx else np.zeros(3)
            keep = QtWidgets.QTableWidgetItem()
            keep.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            keep.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(i, 0, keep)
            cells = (str(i + 1), str(len(idx)),
                     " ".join(f"{e}{n}" for e, n in sorted(counts.items())),
                     f"{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}")
            for col, text in enumerate(cells, 1):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(text))
        self.status.setText(
            f"Framework: {len(self.framework)} unique sites "
            f"({len(self.framework) * na * nb * nc} total atoms)  |  "
            f"Sorbate molecules: {len(self.sorbates)}  |  Untick sorbates to drop, then Reduce")
        self.reduce_button.setEnabled(True)

    def keep(self) -> List[int]:
        return [i for i in range(self.table.rowCount())
                if self.table.item(i, 0).checkState() == QtCore.Qt.Checked]

    def reduce(self) -> Optional[Molecule]:
        if not self.reduce_button.isEnabled() or (not self.framework and not self.sorbates):
            self.status.setText("Click Analyze first")
            return None
        self.result_mol = reduce_to_primitive(self.mol, *self.sizes(), self.keep())
        self.accept()
        return self.result_mol


# ======================================================================
# Input generator
# ======================================================================

class InputGeneratorDialog(_QDialogT):
    """MPMC isotherm, CP2K and OpenMM inputs from the shared engine templates."""

    def __init__(self, mol: Molecule, filepath: str = "", parent=None,
                 engine: str = "mpmc_isotherm") -> None:
        super().__init__(parent)
        self.mol, self.filepath = mol, filepath
        self.engine = engine
        self._values_by_engine: Dict[str, Dict] = {}
        self.widgets: Dict[str, QtWidgets.QWidget] = {}
        self.setWindowTitle("Input Generator")
        self.resize(780, 660)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.engine_box = QtWidgets.QComboBox()
        for key, spec in ENGINES.items():
            self.engine_box.addItem(spec.label, key)
        self.engine_box.setCurrentIndex(self.engine_box.findData(engine))
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        top.addWidget(QtWidgets.QLabel("Engine"))
        top.addWidget(self.engine_box)
        top.addWidget(self.description, 1)
        lay.addLayout(top)
        out = QtWidgets.QHBoxLayout()
        self.outdir = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        out.addWidget(QtWidgets.QLabel("Output folder"))
        out.addWidget(self.outdir, 1)
        out.addWidget(browse)
        lay.addLayout(out)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        lay.addWidget(self.form_host)
        buttons = QtWidgets.QHBoxLayout()
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.template_button = QtWidgets.QPushButton("Show template")
        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.setDefault(True)
        self.validate_button.clicked.connect(self.validate)
        self.template_button.clicked.connect(self.show_template)
        self.generate_button.clicked.connect(self.generate)
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.template_button)
        buttons.addStretch(1)
        buttons.addWidget(self.generate_button)
        lay.addLayout(buttons)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(_mono_font())
        lay.addWidget(self.preview, 1)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.engine_box.currentIndexChanged.connect(
            lambda _: self.set_engine(self.engine_box.currentData()))
        self._load_engine()

    def _browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder",
                                                          self.outdir.text())
        if path:
            self.outdir.setText(path)

    def set_engine(self, key: str) -> None:
        if key == self.engine:
            return
        self._values_by_engine[self.engine] = self.values(strict=False)
        self.engine = key
        self.engine_box.blockSignals(True)
        self.engine_box.setCurrentIndex(self.engine_box.findData(key))
        self.engine_box.blockSignals(False)
        self._load_engine()

    def _load_engine(self) -> None:
        spec = ENGINES[self.engine]
        self.description.setText(spec.description)
        self.outdir.setText(default_engine_outdir(self.filepath, self.engine))
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets.clear()
        saved = self._values_by_engine.get(self.engine, {})
        for var in spec.user_vars:
            value = saved.get(var.name, var.default)
            if var.kind == "select":
                w = QtWidgets.QComboBox()
                w.addItems(list(var.choices or ()))
                if value in (var.choices or ()):
                    w.setCurrentText(value)
            elif var.kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(int(var.min_val) if var.min_val is not None else 0,
                           int(var.max_val) if var.max_val is not None else 99)
                w.setSingleStep(int(var.step) if var.step is not None else 1)
                w.setValue(int(value))
            else:  # float, str and csv_floats are typed in, as in the terminal viewer
                if isinstance(value, list):
                    value = ", ".join(repr(float(v)) for v in value)  # exact, not rounded
                w = QtWidgets.QLineEdit(str(value))
            if var.help:
                w.setToolTip(var.help)
            self.widgets[var.name] = w
            self.form.addRow(var.label, w)
        if self.engine == "mpmc_isotherm":
            self.validate()
        else:
            self.status.setText("Edit values, then Generate.")
            self.preview.setPlainText(
                f"Engine: {self.engine}\nOutput file: {spec.output_filename}\n\n"
                "Click Show template to inspect the source template, "
                "or Generate to write the rendered files.")

    def values(self, strict: bool = True) -> Dict:
        out: Dict = {}
        for var in ENGINES[self.engine].user_vars:
            w = self.widgets[var.name]
            if isinstance(w, QtWidgets.QSpinBox):
                raw = w.value()
            elif isinstance(w, QtWidgets.QComboBox):
                raw = w.currentText()
            else:
                raw = w.text()
            try:
                out[var.name] = parse_form_value(var, raw)
            except ValueError:
                if strict:
                    raise
        return out

    def set_structure(self, mol: Molecule, frames: Optional[list] = None) -> None:
        """Follow the document after edits (validation and generation use the new atoms)."""
        self.mol = mol

    def _show(self, outcome) -> None:
        if outcome.preview is not None:
            self.preview.setPlainText(outcome.preview)
        self.status.setText(outcome.status)

    def validate(self) -> None:
        self._show(check_engine(self.mol, self.engine, self.values(strict=False)))

    def show_template(self) -> None:
        self._show(engine_template_text(self.engine))

    def generate(self):
        try:
            values = self.values(strict=True)
        except (ValueError, KeyError) as e:
            self.status.setText(f"Invalid input: {e}")
            return None
        outcome = generate_engine_inputs(self.mol, self.engine, values,
                                         self.outdir.text().strip())
        self._show(outcome)
        return outcome


# ======================================================================
# Database search
# ======================================================================

class DatabaseDialog(_QDialogT):
    """COD, RCSB PDB, PubChem and Materials Project: search, then fetch into a new tab.

    ``on_fetched(path, label, rid)`` is called with the downloaded file.
    """

    def __init__(self, cache_dir: str, on_fetched: Callable, parent=None) -> None:
        super().__init__(parent)
        self.cache_dir, self.on_fetched = cache_dir, on_fetched
        self.results: list = []
        self.task: Optional[GuiTask] = None
        self.setWindowTitle("Fetch from Database")
        self.resize(820, 520)
        lay = QtWidgets.QVBoxLayout(self)
        row1 = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.addItem("All databases", "all")
        for code, spec in BACKENDS.items():
            self.source.addItem(spec["label"], code)
        self.mode = QtWidgets.QComboBox()
        row1.addWidget(QtWidgets.QLabel("Database"))
        row1.addWidget(self.source)
        row1.addWidget(QtWidgets.QLabel("Search by"))
        row1.addWidget(self.mode)
        row1.addStretch(1)
        lay.addLayout(row1)
        row2 = QtWidgets.QHBoxLayout()
        self.query = QtWidgets.QLineEdit()
        self.search_button = QtWidgets.QPushButton("Search")
        self.fetch_id_button = QtWidgets.QPushButton("Fetch by ID")
        self.fetch_button = QtWidgets.QPushButton("Fetch selected")
        row2.addWidget(self.query, 1)
        for b in (self.search_button, self.fetch_id_button, self.fetch_button):
            row2.addWidget(b)
        lay.addLayout(row2)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source", "ID", "Formula", "Title"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        self.preview = QtWidgets.QLabel("")
        self.status = QtWidgets.QLabel(
            "Pick a source + mode → type a query → Search. Or 'Fetch by ID' to skip search.")
        lay.addWidget(self.preview)
        lay.addWidget(self.status)
        self.source.currentIndexChanged.connect(lambda _: self._source_changed())
        self.mode.currentIndexChanged.connect(lambda _: self._update_placeholder())
        self.search_button.clicked.connect(self.search)
        self.query.returnPressed.connect(self.search)
        self.fetch_id_button.clicked.connect(self.fetch_by_id)
        self.fetch_button.clicked.connect(self.fetch_selected)
        self.table.doubleClicked.connect(lambda _: self.fetch_selected())
        self.table.currentCellChanged.connect(lambda r, *_: self._preview(r))
        self._source_changed()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.task is not None:
            self.task.detach()
        super().closeEvent(event)

    def _source_changed(self) -> None:
        current = self.mode.currentData()
        self.mode.blockSignals(True)
        self.mode.clear()
        for m in modes_for(self.source.currentData()):
            self.mode.addItem(m.capitalize(), m)
        self.mode.setCurrentIndex(max(self.mode.findData(current), 0))
        self.mode.blockSignals(False)
        self._update_placeholder()

    def _update_placeholder(self) -> None:
        self.query.setPlaceholderText(
            placeholder_for(self.source.currentData(), self.mode.currentData()))

    def _preview(self, row: int) -> None:
        if not 0 <= row < len(self.results):
            return
        r = self.results[row]
        label = BACKENDS.get(r.get("source", ""), {}).get("label", r.get("source", ""))
        bits = [f"<b>{label}</b> {r.get('id') or ''}"]
        if r.get("formula"):
            bits.append(f"formula: {r['formula']}")
        if r.get("title"):
            bits.append(f"title: {r['title']}")
        self.preview.setText("  •  ".join(bits))

    def search(self) -> Optional[GuiTask]:
        source, mode = self.source.currentData(), self.mode.currentData()
        query = self.query.text().strip()
        if mode == "id":  # direct-ID mode skips the search
            if not query:
                self.status.setText("Enter an ID then click 'Fetch by ID'")
                return None
            return self._fetch(source, query)
        if not query:
            return None
        labels = ", ".join(BACKENDS[c]["label"] for c in backends_for(source, mode))
        self.status.setText(f"Searching {labels} ({mode})...")
        self.task = GuiTask(lambda: search_backends(source, mode, query), self._show_results,
                            lambda e: self.status.setText(f"Error: {e}"))
        return self.task

    def _show_results(self, out: tuple) -> None:
        self.results, msg = out
        self.table.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            label = BACKENDS.get(r.get("source", ""), {}).get("label", r.get("source", ""))
            cells = (label, str(r.get("id") or ""), str(r.get("formula") or "")[:25],
                     str(r.get("title") or "")[:60])
            for j, text in enumerate(cells):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.status.setText(msg)
        self.preview.setText("")

    def fetch_selected(self) -> Optional[GuiTask]:
        row = self.table.currentRow()
        if not 0 <= row < len(self.results):
            self.status.setText("Select a result first")
            return None
        r = self.results[row]
        if not r.get("id"):
            self.status.setText("This result has no ID to fetch")
            return None
        return self._fetch(r.get("source", ""), str(r["id"]))

    def fetch_by_id(self) -> Optional[GuiTask]:
        source = self.source.currentData()
        if source == "all":
            self.status.setText("Pick a specific database first (not 'All')")
            return None
        rid = self.query.text().strip()
        if not rid:
            self.status.setText("Enter an ID in the query field")
            return None
        return self._fetch(source, rid)

    def _fetch(self, source: str, rid: str) -> Optional[GuiTask]:
        spec = BACKENDS.get(source)
        if spec is None:
            self.status.setText(f"Unknown source: {source!r}")
            return None
        if source == "all":
            self.status.setText("Pick a specific database first (not 'All')")
            return None
        self.status.setText(f"Fetching {spec['label']} {rid}...")
        cache = self.cache_dir

        def done(path: str) -> None:
            try:
                self.on_fetched(path, spec["label"], rid)
            except Exception as e:  # e.g. a download that is not a readable structure
                self.status.setText(f"Error opening {rid}: {e}")
                return
            self.accept()

        self.task = GuiTask(lambda: spec["fetch"](rid, cache), done,
                            lambda e: self.status.setText(f"Error: {e}"))
        return self.task


# ======================================================================
# Geometry tables
# ======================================================================

GEOMETRY_TABS = ["Bonds", "Angles", "Dihedrals", "Contacts", "Mols", "Lone", "Coords"]
_SORTABLE = {"Bonds", "Angles", "Dihedrals"}


def _label(mol: Molecule, i: int) -> str:
    return f"{i + 1}:{mol.atoms[i].element.symbol}"


def geometry_rows(mol: Optional[Molecule], tab: str, ascending: bool = False) -> tuple:
    """(column headers, [(cells, atom indices)]) for one geometry tab.

    The same tables as the terminal viewer's geometry panel.
    """
    if mol is None or not mol.atoms:
        return [], []
    if tab in _SORTABLE and not mol.bonds:
        mol.detect_bonds()
    if tab == "Bonds":
        data = mol.get_bond_lengths()
        if ascending:
            data.sort(key=lambda x: x[2])
        return (["Atom 1", "Atom 2", "Length (Å)"],
                [((_label(mol, i), _label(mol, j), f"{d:.4f}"), (i, j)) for i, j, d in data])
    if tab == "Angles":
        data = mol.get_angles()
        if ascending:
            data.sort(key=lambda x: x[3])
        return (["Atom 1", "Vertex", "Atom 3", "Angle (°)"],
                [((_label(mol, i), _label(mol, j), _label(mol, k), f"{a:.3f}"), (i, j, k))
                 for i, j, k, a in data])
    if tab == "Dihedrals":
        data = mol.get_dihedrals()
        if ascending:
            data.sort(key=lambda x: x[4])
        return (["Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle (°)"],
                [((_label(mol, i), _label(mol, j), _label(mol, k), _label(mol, m), f"{a:.3f}"),
                  (i, j, k, m)) for i, j, k, m, a in data])
    if tab == "Contacts":
        set_atom_ids(mol.atoms)
        rows = []
        for msg in get_close_contacts(mol):
            parts = msg.split()
            el1, el2 = parts[0].split("-")
            id1, id2 = int(parts[1]), int(parts[2])
            rows.append(((f"{id1}:{el1}", f"{id2}:{el2}", parts[-1]), (id1 - 1, id2 - 1)))
        return ["Atom 1", "Atom 2", "Distance (Å)"], rows
    if tab == "Mols":
        index = {id(a): k for k, a in enumerate(mol.atoms)}
        rows = []
        for mi, sm in enumerate(mol.find_molecules()):
            atoms = sorted(sm.atoms, key=lambda a: a.atomic_number, reverse=True)
            counts: dict = {}
            for a in atoms:
                counts[a.element.symbol] = counts.get(a.element.symbol, 0) + 1
            formula = "".join(f"{el}{n}" if n > 1 else el for el, n in counts.items())
            rows.append(((str(mi + 1), str(len(atoms)), formula),
                         tuple(index[id(a)] for a in atoms if id(a) in index)))
        return ["#", "Atoms", "Formula"], rows
    if tab == "Lone":
        set_atom_ids(mol.atoms)
        index = {id(a): k for k, a in enumerate(mol.atoms)}
        return (["#", "Element", "x", "y", "z"],
                [((str(a.id), a.element.symbol, f"{a.x[0]:.4f}", f"{a.x[1]:.4f}",
                   f"{a.x[2]:.4f}"), (index[id(a)],)) for a in get_lone_atoms(mol)])
    if tab == "Coords":
        return (["#", "El", "x", "y", "z"],
                [((str(k + 1), a.element.symbol, f"{a.x[0]:.4f}", f"{a.x[1]:.4f}",
                   f"{a.x[2]:.4f}"), (k,)) for k, a in enumerate(mol.atoms)])
    raise ValueError(f"unknown geometry tab {tab!r}")


class GuiGeometryPanel(_QWidgetT):
    """Bonds, angles, dihedrals, close contacts, molecules, lone atoms and coordinates.

    Selecting a row highlights its atoms in the view; editing a Coords cell moves the atom.
    A table is filled only when its tab is shown.
    """

    def __init__(self, on_highlight: Callable[[tuple], None],
                 on_edit_coord: Callable[[int, tuple], None], parent=None) -> None:
        super().__init__(parent)
        self.on_highlight, self.on_edit_coord = on_highlight, on_edit_coord
        self.mol: Optional[Molecule] = None
        self._stale = set(GEOMETRY_TABS)
        self._ascending: Dict[str, bool] = {}
        self._indices: Dict[str, list] = {}
        self._filling = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.tables: Dict[str, QtWidgets.QTableWidget] = {}
        for name in GEOMETRY_TABS:
            t = QtWidgets.QTableWidget(0, 0)
            t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            t.verticalHeader().setVisible(False)
            t.horizontalHeader().setStretchLastSection(True)
            t.currentCellChanged.connect(lambda r, *_, n=name: self._row_selected(n, r))
            t.itemChanged.connect(lambda item, n=name: self._item_changed(n, item))
            self.tables[name] = t
            self.tabs.addTab(t, name)
        self.tabs.currentChanged.connect(lambda _: self.refresh_visible())
        lay.addWidget(self.tabs)
        row = QtWidgets.QHBoxLayout()
        self.sort_button = QtWidgets.QPushButton("Sort by value")
        self.sort_button.setCheckable(True)
        self.sort_button.toggled.connect(self._toggle_sort)
        self.hint = QtWidgets.QLabel("")
        row.addWidget(self.sort_button)
        row.addWidget(self.hint, 1)
        lay.addLayout(row)

    def current_tab(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())

    def set_molecule(self, mol: Optional[Molecule]) -> None:
        self.mol = mol
        self._stale = set(GEOMETRY_TABS)
        self.refresh_visible()

    def refresh_visible(self, force: bool = False) -> None:
        name = self.current_tab()
        self.sort_button.setEnabled(name in _SORTABLE)
        self.sort_button.blockSignals(True)
        self.sort_button.setChecked(self._ascending.get(name, False))
        self.sort_button.blockSignals(False)
        self.hint.setText("Double-click x, y or z to move an atom" if name == "Coords" else "")
        if not self.isVisible() or (name not in self._stale and not force):
            return
        headers, rows = geometry_rows(self.mol, name, self._ascending.get(name, False))
        t = self.tables[name]
        self._filling = True
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(rows))
        for r, (cells, _) in enumerate(rows):
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if name != "Coords" or c < 2:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                t.setItem(r, c, item)
        self._indices[name] = [idx for _, idx in rows]
        self._filling = False
        self._stale.discard(name)

    def _toggle_sort(self, on: bool) -> None:
        self._ascending[self.current_tab()] = on
        self.refresh_visible(force=True)

    def _row_selected(self, name: str, row: int) -> None:
        idx = self._indices.get(name, [])
        if not self._filling and 0 <= row < len(idx):
            self.on_highlight(idx[row])

    def _item_changed(self, name: str, item) -> None:
        if self._filling or name != "Coords" or self.mol is None:
            return
        row, t = item.row(), self.tables[name]
        try:
            xyz = tuple(float(t.item(row, c).text()) for c in (2, 3, 4))
        except (ValueError, AttributeError):
            xyz = ()
        if len(xyz) != 3 or not np.all(np.isfinite(xyz)):
            self.refresh_visible(force=True)  # not a usable number: put the old value back
            return
        self.on_edit_coord(self._indices[name][row][0], xyz)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh_visible()


# ======================================================================
# Trajectory bar
# ======================================================================

class GuiFrameBar(_QWidgetT):
    """First / previous / play / next / last, a slider, the frame count and the speed."""

    BASE_INTERVAL_MS = 100

    def __init__(self, on_frame: Callable[[int], None], parent=None) -> None:
        super().__init__(parent)
        self.on_frame = on_frame
        self.n = 1
        self.speed = 1.0
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        sp = QtWidgets.QStyle
        self.buttons: Dict[str, QtWidgets.QToolButton] = {}
        for name, icon, tip in (("first", sp.SP_MediaSkipBackward, "First frame (Home)"),
                                ("prev", sp.SP_MediaSeekBackward, "Previous frame ([)"),
                                ("play", sp.SP_MediaPlay, "Play / pause (Space)"),
                                ("next", sp.SP_MediaSeekForward, "Next frame (])"),
                                ("last", sp.SP_MediaSkipForward, "Last frame (End)")):
            b = QtWidgets.QToolButton()
            b.setIcon(self.style().standardIcon(icon))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            self.buttons[name] = b
            lay.addWidget(b)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        lay.addWidget(self.slider, 1)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(80)
        lay.addWidget(self.label)
        self.buttons["slower"] = QtWidgets.QToolButton()
        self.buttons["slower"].setText("−")
        self.buttons["slower"].setToolTip("Slower")
        self.speed_label = QtWidgets.QLabel("1x")
        self.buttons["faster"] = QtWidgets.QToolButton()
        self.buttons["faster"].setText("+")
        self.buttons["faster"].setToolTip("Faster")
        for w in (self.buttons["slower"], self.speed_label, self.buttons["faster"]):
            lay.addWidget(w)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.BASE_INTERVAL_MS)
        self.timer.timeout.connect(self.step)
        self.buttons["first"].clicked.connect(lambda: self.go(0))
        self.buttons["last"].clicked.connect(lambda: self.go(self.n - 1))
        self.buttons["prev"].clicked.connect(lambda: self.go(self.slider.value() - 1))
        self.buttons["next"].clicked.connect(lambda: self.go(self.slider.value() + 1))
        self.buttons["play"].clicked.connect(self.toggle_play)
        self.buttons["slower"].clicked.connect(lambda: self.set_speed(self.speed / 2))
        self.buttons["faster"].clicked.connect(lambda: self.set_speed(self.speed * 2))
        self.slider.valueChanged.connect(self._changed)
        self.setVisible(False)

    @property
    def index(self) -> int:
        return self.slider.value()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def set_count(self, n: int, current: int = 0) -> None:
        self.stop(notify=False)
        self.n = max(n, 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n - 1)
        self.slider.setValue(min(current, self.n - 1))
        self.slider.blockSignals(False)
        self._label()
        self.setVisible(self.n > 1)

    def go(self, idx: int) -> None:
        self.slider.setValue(max(0, min(idx, self.n - 1)))

    def _label(self) -> None:
        self.label.setText(f"{self.slider.value() + 1}/{self.n}")

    def _changed(self, value: int) -> None:
        self._label()
        self.on_frame(value)

    def set_speed(self, speed: float) -> None:
        self.speed = float(min(16.0, max(0.125, speed)))
        self.speed_label.setText(f"{self.speed:g}x")
        self.timer.setInterval(max(1, int(self.BASE_INTERVAL_MS / self.speed)))

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.stop()
        elif self.n > 1:
            self.timer.start()
            self.buttons["play"].setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))

    def stop(self, notify: bool = True) -> None:
        was = self.timer.isActive()
        self.timer.stop()
        self.buttons["play"].setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        if was and notify:  # listeners refresh what they skip while playing
            self.on_frame(self.slider.value())

    def step(self) -> None:
        self.slider.setValue((self.slider.value() + 1) % self.n)
