"""PDB Wizard TUI viewer (adapted from moltui, MO code removed)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

if not hasattr(asyncio, "to_thread"):
    # Python 3.8 backport of asyncio.to_thread (added in 3.9). pdb_wizard
    # targets >=3.8 and offloads analysis compute via to_thread throughout.
    import contextvars
    import functools

    async def _to_thread(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        call = functools.partial(ctx.run, func, *args, **kwargs)
        return await loop.run_in_executor(None, call)

    asyncio.to_thread = _to_thread

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    TabbedContent,
    TextArea,
)

from pdb_wizard.molecule import Molecule
from pdb_wizard.panels import GeometryPanel, OperationsPanel, VisualPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.renderer import render_scene, rotation_matrix

_BRAILLE_MAP = np.array(
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint8,
)


class InputModal(ModalScreen[Optional[str]]):
    """Modal dialog that prompts for a text value with OK/Cancel."""

    # No 'q' binding — the focused Input would capture it as text. Escape only.
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    InputModal {
        align: center middle;
    }
    InputModal > Vertical {
        width: 60%;
        min-width: 40;
        max-width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    InputModal Label {
        margin-bottom: 1;
    }
    InputModal Input {
        margin-bottom: 1;
    }
    InputModal .hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    InputModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    InputModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self, prompt: str, default: str = "", hint: str = "") -> None:
        super().__init__()
        self._prompt = prompt
        self._default = default
        self._hint = hint

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._prompt)
            yield Input(value=self._default, id="modal-input")
            if self._hint:
                yield Label(self._hint, classes="hint")
            with Horizontal():
                yield Button("OK", id="input-ok", variant="primary")
                yield Button("Cancel", id="input-cancel")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "input-ok":
            self.dismiss(self.query_one("#modal-input", Input).value)
        else:
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


class InfoModal(ModalScreen[None]):
    """Modal dialog showing copyable text with an OK button."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close"),
        Binding("enter", "dismiss_modal", "OK"),
    ]
    DEFAULT_CSS = """
    InfoModal {
        align: center middle;
    }
    InfoModal > Vertical {
        width: 70%;
        min-width: 40;
        max-width: 90;
        height: 70%;
        min-height: 12;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    InfoModal Label {
        height: 1;
        margin-bottom: 1;
        text-style: bold;
    }
    InfoModal TextArea {
        height: 1fr;
        margin-bottom: 1;
    }
    InfoModal Button {
        dock: bottom;
        width: auto;
        min-width: 10;
        margin-top: 1;
        align-horizontal: center;
    }
    """

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self._title = title
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            yield TextArea(self._content, read_only=True, id="info-text")
            yield Button("OK", id="info-ok", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#info-ok", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Modal OK/Cancel confirmation dialog."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "OK"),
    ]
    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal > Vertical {
        width: 60%;
        min-width: 36;
        max-width: 70;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ConfirmModal Label {
        margin-bottom: 1;
        text-style: bold;
    }
    ConfirmModal .description {
        margin-bottom: 1;
        text-style: none;
    }
    ConfirmModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    ConfirmModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self, title: str, description: str, yes_no: bool = False) -> None:
        super().__init__()
        self._title = title
        self._description = description
        self._yes_no = yes_no

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            yield Label(self._description, classes="description")
            with Horizontal():
                ok_label = "Yes" if self._yes_no else "OK"
                cancel_label = "No" if self._yes_no else "Cancel"
                yield Button(ok_label, id="confirm-ok", variant="primary")
                yield Button(cancel_label, id="confirm-cancel")

    def on_mount(self) -> None:
        self.query_one("#confirm-ok", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-ok")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class FileSaveModal(ModalScreen[Optional[str]]):
    """Modal file browser for choosing a save location."""

    # No 'q' binding — the focused Input/DirectoryTree would capture it. Escape only.
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    DEFAULT_CSS = """
    FileSaveModal {
        align: center middle;
    }
    FileSaveModal > Vertical {
        width: 80%;
        min-width: 45;
        max-width: 120;
        height: 80%;
        min-height: 14;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    FileSaveModal DirectoryTree {
        height: 1fr;
        margin-bottom: 1;
    }
    FileSaveModal Input {
        margin-bottom: 1;
    }
    FileSaveModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    FileSaveModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self, prompt: str, default: str = "", button_label: str = "Save") -> None:
        super().__init__()
        self._prompt = prompt
        self._default = default
        self._button_label = button_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._prompt)
            yield DirectoryTree(str(Path(self._default).parent or "."), id="file-tree")
            yield Input(value=self._default, id="file-input")
            with Horizontal():
                yield Button(self._button_label, id="file-save", variant="primary")
                yield Button("Cancel", id="file-cancel")

    def on_mount(self) -> None:
        self.query_one("#file-input", Input).focus()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        inp = self.query_one("#file-input", Input)
        inp.value = str(event.path)
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-save":
            self.dismiss(self.query_one("#file-input", Input).value)
        else:
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


class ForceFieldModal(ModalScreen[Optional[str]]):
    """Modal to select a force field with a live parameter preview."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Close"),
    ]
    DEFAULT_CSS = """
    ForceFieldModal {
        align: center middle;
    }
    ForceFieldModal > Vertical {
        width: 75%;
        min-width: 45;
        max-width: 100;
        height: 75%;
        min-height: 16;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ForceFieldModal Label {
        margin-bottom: 1;
    }
    ForceFieldModal .title {
        text-style: bold;
    }
    ForceFieldModal RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    ForceFieldModal TextArea {
        height: 1fr;
        margin-bottom: 1;
    }
    ForceFieldModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    ForceFieldModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self, molecule: "Molecule") -> None:
        super().__init__()
        self._molecule = molecule

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Select Force Field", classes="title")
            with RadioSet(id="ff-radio"):
                yield RadioButton("None (skip)", value=True, id="ff-none")
                yield RadioButton("OPLS-AA / UFF", id="ff-oplsaa")
                yield RadioButton("PHAHST", id="ff-phahst")
            yield Label("Parameter preview:")
            yield TextArea("", read_only=True, id="ff-preview")
            with Horizontal():
                yield Button("OK", id="ff-apply", variant="primary")
                yield Button("Cancel", id="ff-cancel")

    def on_mount(self) -> None:
        self._update_preview(-1)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        idx_map = {"ff-none": -1, "ff-oplsaa": 0, "ff-phahst": 1}
        self._update_preview(idx_map.get(event.pressed.id, -1))

    def _update_preview(self, ff_idx: int) -> None:
        from collections import Counter
        elements = Counter(a.element.symbol for a in self._molecule.atoms)

        if ff_idx < 0:
            lines = ["No force field selected.", "", "Parameters will not be written."]
            self.query_one("#ff-preview", TextArea).text = "\n".join(lines)
            return

        from pdb_wizard.forcefields import get_forcefield
        ff = get_forcefield(ff_idx)
        label = ["OPLS-AA/UFF", "PHAHST"][ff_idx]
        is_phahst = ff_idx == 1

        lines = [f"{label} -- {len(ff)} element types available", ""]
        if is_phahst:
            lines.append(" El     #      rho     beta    alpha         C6         C8          C10")
            lines.append("               (A)    (1/A)    (A^3)     (a.u.)     (a.u.)       (a.u.)")
            lines.append("-" * 72)
            for el, count in sorted(elements.items()):
                if el in ff:
                    p = ff[el]
                    sig, eps, alp = p.get("sigma", 0), p.get("epsilon", 0), p.get("alpha", 0)
                    c6, c8, c10 = p.get("c6", 0), p.get("c8", 0), p.get("c10", 0)
                    lines.append(
                        f"{el:>3} {count:>5} {sig:>8.4f} {eps:>8.4f} {alp:>8.5f}"
                        f" {c6:>10.4f} {c8:>10.4f} {c10:>12.4f}"
                    )
                else:
                    lines.append(f"{el:>3} {count:>5}      ---      ---      ---        ---        ---          ---")
        else:
            lines.append(" El     #    sigma      eps    alpha")
            lines.append("               (A)      (K)    (A^3)")
            lines.append("-" * 38)
            for el, count in sorted(elements.items()):
                if el in ff:
                    p = ff[el]
                    sig, eps, alp = p.get("sigma", 0), p.get("epsilon", 0), p.get("alpha", 0)
                    lines.append(
                        f"{el:>3} {count:>5} {sig:>8.4f} {eps:>8.2f} {alp:>8.5f}"
                    )
                else:
                    lines.append(f"{el:>3} {count:>5}      ---      ---      ---")

        self.query_one("#ff-preview", TextArea).text = "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ff-apply":
            if self.query_one("#ff-phahst", RadioButton).value:
                self.dismiss("PHAHST")
            elif self.query_one("#ff-oplsaa", RadioButton).value:
                self.dismiss("OPLSAA")
            else:
                self.dismiss("")  # None selected: skip FF but continue the flow
        else:
            self.dismiss(None)  # Cancel: abort the whole flow

    def action_cancel(self) -> None:
        self.dismiss(None)  # Cancel: abort the whole flow


class TrackSlider(Widget):
    """Clickable/draggable horizontal slider for trajectory scrubbing."""

    DEFAULT_CSS = """
    TrackSlider {
        height: 1;
        width: 1fr;
    }
    """

    class Seeked(Message):
        def __init__(self, fraction: float) -> None:
            super().__init__()
            self.fraction = fraction

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fraction = 0.0
        self._total = 1
        self._current = 0
        self._dragging = False

    def set_position(self, current: int, total: int) -> None:
        self._current = current
        self._total = max(1, total)
        self._fraction = current / self._total
        self.refresh()

    def render_line(self, y: int) -> Strip:
        w = self.size.width
        if w < 1:
            return Strip.blank(w)
        filled = int(self._fraction * w)
        filled = max(0, min(w, filled))
        empty = w - filled
        segments = []
        if filled > 0:
            segments.append(Segment(" " * filled, Style(bgcolor="rgb(0,180,100)")))
        if empty > 0:
            segments.append(Segment(" " * empty, Style(bgcolor="rgb(40,40,45)")))
        return Strip(segments, w)

    def _seek_to_x(self, x: int) -> None:
        w = self.size.width
        if w <= 0:
            return
        frac = max(0.0, min(1.0, x / w))
        self._fraction = frac
        self.refresh()
        self.post_message(self.Seeked(frac))

    def on_mouse_down(self, event) -> None:
        self._dragging = True
        self.capture_mouse()
        self._seek_to_x(event.x)

    def on_mouse_up(self, event) -> None:
        self._dragging = False
        self.release_mouse()

    def on_mouse_move(self, event) -> None:
        if self._dragging:
            self._seek_to_x(event.x)

    def on_click(self, event) -> None:
        self._seek_to_x(event.x)


class SpinBox(Widget, can_focus=True):
    """Integer input with +/- buttons and mouse wheel support."""

    DEFAULT_CSS = """
    SpinBox {
        height: 3;
        layout: horizontal;
    }
    SpinBox Button {
        width: 4;
        min-width: 4;
        height: 3;
    }
    SpinBox .spin-value {
        width: 1fr;
        height: 3;
        content-align: center middle;
        text-style: bold;
        border: tall $accent;
    }
    SpinBox:focus .spin-value {
        border: tall $accent-lighten-2;
    }
    """

    def __init__(self, value: int = 1, min_val: int = 1, max_val: int = 99, step: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value = max(min_val, min(max_val, value))
        self._min = min_val
        self._max = max_val
        self._step = step

    @property
    def value(self) -> int:
        return self._value

    def compose(self) -> ComposeResult:
        yield Button("-", id="spin-dec")
        yield Label(str(self._value), id="spin-display", classes="spin-value")
        yield Button("+", id="spin-inc")

    class Changed(Message):
        def __init__(self, spin_box: "SpinBox", value: int) -> None:
            super().__init__()
            self.spin_box = spin_box
            self.value = value

    def _update(self, new: int) -> None:
        new = max(self._min, min(self._max, new))
        if new != self._value:
            self._value = new
            self.query_one("#spin-display", Label).update(str(self._value))
            self.post_message(self.Changed(self, self._value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "spin-inc":
            self._update(self._value + self._step)
        elif event.button.id == "spin-dec":
            self._update(self._value - self._step)

    def on_mouse_scroll_up(self, event) -> None:
        self._update(self._value + self._step)

    def on_mouse_scroll_down(self, event) -> None:
        self._update(self._value - self._step)

    def on_key(self, event: Key) -> None:
        if event.key == "up":
            self._update(self._value + self._step)
            event.stop()
        elif event.key == "down":
            self._update(self._value - self._step)
            event.stop()


class ExtendAxisModal(ModalScreen[tuple]):
    """Modal to extend all three axes at once. Returns (na, nb, nc), the number
    of extra copies per axis, matching the classic menu's "How many times"."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("q", "cancel", "Close")]
    DEFAULT_CSS = """
    ExtendAxisModal {
        align: center middle;
    }
    ExtendAxisModal > Vertical {
        width: 60%;
        min-width: 40;
        max-width: 60;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ExtendAxisModal Label {
        margin-bottom: 0;
    }
    ExtendAxisModal .title {
        text-style: bold;
        margin-bottom: 1;
    }
    ExtendAxisModal SpinBox {
        width: 24;
        height: 3;
        margin: 0 1;
    }
    ExtendAxisModal .axis-label {
        width: 4;
        height: 3;
        content-align: right middle;
        text-style: bold;
    }
    ExtendAxisModal .axis-row {
        height: auto;
        margin-bottom: 0;
        align-horizontal: center;
    }
    ExtendAxisModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    ExtendAxisModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self, pbc) -> None:
        super().__init__()
        self._pbc = pbc

    def compose(self) -> ComposeResult:
        pbc = self._pbc
        if pbc:
            sub = f"a={pbc.a:.2f}  b={pbc.b:.2f}  c={pbc.c:.2f} A"
        else:
            sub = ""
        with Vertical():
            yield Label("Extend Supercell", classes="title")
            if sub:
                yield Label(sub)
            yield Label("Times to extend each axis (1 doubles it, as in the classic menu):")
            with Horizontal(classes="axis-row"):
                yield Label("a:", classes="axis-label")
                yield SpinBox(value=0, min_val=0, max_val=19, id="extend-na")
            with Horizontal(classes="axis-row"):
                yield Label("b:", classes="axis-label")
                yield SpinBox(value=0, min_val=0, max_val=19, id="extend-nb")
            with Horizontal(classes="axis-row"):
                yield Label("c:", classes="axis-label")
                yield SpinBox(value=0, min_val=0, max_val=19, id="extend-nc")
            with Horizontal():
                yield Button("Extend", id="extend-ok", variant="primary")
                yield Button("Cancel", id="extend-cancel")

    def on_mount(self) -> None:
        self.query_one("#extend-na", SpinBox).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "extend-ok":
            self._submit()
        elif event.button.id == "extend-cancel":
            self.dismiss(())

    def _submit(self) -> None:
        na = self.query_one("#extend-na", SpinBox).value
        nb = self.query_one("#extend-nb", SpinBox).value
        nc = self.query_one("#extend-nc", SpinBox).value
        self.dismiss((na, nb, nc))

    def action_cancel(self) -> None:
        self.dismiss(())


class SorbateModal(ModalScreen[Optional[str]]):
    """Modal to select a sorbate molecule to insert into the MPMC PDB.

    ``standalone`` (Edit > Pack Sorbates) swaps Skip for "From .pqr file", which
    dismisses with ``FROM_FILE``.
    """

    FROM_FILE = "__file__"

    def __init__(self, standalone: bool = False) -> None:
        super().__init__()
        self._standalone = standalone

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("q", "cancel", "Close")]
    DEFAULT_CSS = """
    SorbateModal {
        align: center middle;
    }
    SorbateModal > Vertical {
        width: 70%;
        min-width: 45;
        max-width: 90;
        height: 70%;
        min-height: 16;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    SorbateModal Label {
        margin-bottom: 1;
    }
    SorbateModal .title {
        text-style: bold;
    }
    SorbateModal DataTable {
        height: 1fr;
        margin-bottom: 1;
    }
    SorbateModal Horizontal {
        height: auto;
        align-horizontal: center;
    }
    SorbateModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Insert Sorbate Molecule", classes="title")
            yield Label("Select a sorbate to insert (you choose how many next):")
            yield DataTable(id="sorbate-table", cursor_type="row")
            with Horizontal():
                yield Button("Insert", id="sorb-ok", variant="primary")
                if self._standalone:
                    yield Button("From .pqr file", id="sorb-file")
                else:
                    yield Button("Skip", id="sorb-skip")
                yield Button("Cancel", id="sorb-cancel")

    def on_mount(self) -> None:
        from pdb_wizard.sorbates import SORBATE_MODELS
        table = self.query_one("#sorbate-table", DataTable)
        table.add_columns("Model", "Mol", "Sites", "Description")
        groups = [
            ("lj", "--- LJ (nonpolar) ---"),
            ("lj_polar", "--- LJ (polarizable) ---"),
            ("phahst", "--- PHAHST (exp-6-8-10) ---"),
        ]
        for pot_type, header in groups:
            models = {k: v for k, v in SORBATE_MODELS.items() if v.pot_type == pot_type}
            if models:
                table.add_row("", "", "", header, key=f"_hdr_{pot_type}")
                for name, model in models.items():
                    table.add_row(name, model.label, str(len(model.sites)), model.description, key=name)
        table.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sorb-ok":
            table = self.query_one("#sorbate-table", DataTable)
            row_keys = list(table.rows.keys())
            if row_keys:
                row = max(0, min(table.cursor_row, len(row_keys) - 1))
                key = row_keys[row].value
                if key and not key.startswith("_hdr_"):
                    self.dismiss(key)
                    return
            self.dismiss("")
        elif event.button.id == "sorb-skip":
            self.dismiss("")  # Skip: no sorbate, but continue the flow
        elif event.button.id == "sorb-file":
            self.dismiss(self.FROM_FILE)
        else:
            self.dismiss(None)  # Cancel: abort the whole flow

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value and not event.row_key.value.startswith("_hdr_"):
            self.dismiss(event.row_key.value)

    def action_cancel(self) -> None:
        self.dismiss(None)  # Cancel: abort the whole flow


class ChargesFileModal(ModalScreen[Optional[dict]]):
    """File browser with preview and line-skip controls for loading charges."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("q", "cancel", "Close")]
    DEFAULT_CSS = """
    ChargesFileModal {
        align: center middle;
    }
    ChargesFileModal > Vertical {
        width: 90%;
        min-width: 50;
        max-width: 140;
        height: 85%;
        min-height: 16;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ChargesFileModal Label {
        margin-bottom: 1;
    }
    ChargesFileModal .title {
        text-style: bold;
    }
    ChargesFileModal .hint {
        color: $text-muted;
    }
    ChargesFileModal #browser-and-preview {
        height: 1fr;
    }
    ChargesFileModal DirectoryTree {
        width: 1fr;
        height: 100%;
        border-right: solid $accent;
    }
    ChargesFileModal #preview-panel {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }
    ChargesFileModal TextArea {
        height: 1fr;
    }
    ChargesFileModal #skip-controls {
        height: auto;
        margin-top: 1;
    }
    ChargesFileModal #skip-controls SpinBox {
        width: 16;
        margin: 0 1;
    }
    ChargesFileModal #skip-controls Label {
        width: auto;
        margin: 0;
    }
    ChargesFileModal #bottom-buttons {
        height: auto;
        align-horizontal: center;
        margin-top: 1;
    }
    ChargesFileModal Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._file_path = ""
        self._raw_lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Select Charges File", classes="title")
            yield Label(
                "Select a .resp file or a raw column of charges. "
                "Use skip controls to trim header/footer lines.",
                classes="hint",
            )
            with Horizontal(id="browser-and-preview"):
                yield DirectoryTree(".", id="charges-tree")
                with Vertical(id="preview-panel"):
                    yield Label("Preview (lines to be used):", classes="hint")
                    yield TextArea("", read_only=True, id="charges-preview")
            with Horizontal(id="skip-controls"):
                yield Label("Skip first:")
                yield SpinBox(value=0, min_val=0, max_val=9999, id="skip-first")
                yield Label("  Skip last:")
                yield SpinBox(value=0, min_val=0, max_val=9999, id="skip-last")
                yield Label("", id="line-count")
            with Horizontal(id="bottom-buttons"):
                yield Button("OK", id="charges-ok", variant="primary")
                yield Button("Cancel", id="charges-cancel")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._file_path = str(event.path)
        try:
            self._raw_lines = open(self._file_path).readlines()
        except (OSError, UnicodeDecodeError):
            self._raw_lines = []
        self._update_preview()

    def on_spin_box_changed(self, event: SpinBox.Changed) -> None:
        self._update_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "charges-ok":
            self._submit()
        elif event.button.id == "charges-cancel":
            self.dismiss({})

    def _update_preview(self) -> None:
        # CP2K .resp files are auto-parsed (headers/footer/Total stripped) — the
        # skip controls don't apply, so preview the extracted charges directly.
        if self._file_path.lower().endswith(".resp"):
            from pdb_wizard.io import parse_resp_charges
            charges = parse_resp_charges(self._raw_lines)
            preview = [".resp file — auto-parsed, skip controls ignored", ""]
            preview += [f"{i + 1:>4}: {c:+.6f}" for i, c in enumerate(charges)]
            self.query_one("#charges-preview", TextArea).text = "\n".join(preview)
            self.query_one("#line-count", Label).update(
                f"  {len(charges)} charges (.resp)"
            )
            return

        try:
            skip_first = self.query_one("#skip-first", SpinBox).value
        except Exception:
            skip_first = 0
        try:
            skip_last = self.query_one("#skip-last", SpinBox).value
        except Exception:
            skip_last = 0

        lines = self._raw_lines
        total = len(lines)
        end = max(0, total - skip_last)
        start = min(skip_first, end)
        used = lines[start:end]

        preview_lines = []
        for i, line in enumerate(self._raw_lines):
            prefix = "  " if start <= i < end else "X "
            preview_lines.append(f"{prefix}{i + 1:>4}: {line.rstrip()}")

        self.query_one("#charges-preview", TextArea).text = "\n".join(preview_lines)
        self.query_one("#line-count", Label).update(
            f"  {len(used)}/{total} lines"
        )

    def _submit(self) -> None:
        if not self._file_path:
            return
        try:
            skip_first = self.query_one("#skip-first", SpinBox).value
        except Exception:
            skip_first = 0
        try:
            skip_last = self.query_one("#skip-last", SpinBox).value
        except Exception:
            skip_last = 0
        self.dismiss({
            "path": self._file_path,
            "skip_first": skip_first,
            "skip_last": skip_last,
        })

    def action_cancel(self) -> None:
        self.dismiss({})


class PlotWidget(Widget):
    """Braille-based line plot widget."""

    DEFAULT_CSS = """
    PlotWidget {
        height: 1fr;
        border: solid $accent;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._x: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])
        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 5.0
        self._title = ""
        self._x_label = ""
        self._overlay_x: np.ndarray | None = None
        self._overlay_y: np.ndarray | None = None
        self._overlay_color = "rgb(255,100,80)"

    def set_data(self, x: np.ndarray, y: np.ndarray,
                 title: str = "", x_label: str = "",
                 y_min: float = 0.0, y_max: float = 5.0) -> None:
        self._x = x
        self._y = y
        self._title = title
        self._x_label = x_label
        self._y_min = y_min
        self._y_max = y_max
        if len(x) > 0:
            self._x_min = float(x.min())
            self._x_max = float(x.max())
        self.refresh()

    def set_overlay(self, x: np.ndarray | None, y: np.ndarray | None,
                    color: str = "rgb(255,100,80)") -> None:
        """Set a second data series to overlay (e.g. experimental PXRD)."""
        if x is not None and y is not None and len(x) == len(y) and len(x) > 1:
            x, y = np.asarray(x), np.asarray(y)
            if np.any(np.diff(x) < 0):
                # The plot looks samples up by x, so they must ascend (a pattern may be
                # stored from high to low angle).
                order = np.argsort(x, kind="stable")
                x, y = x[order], y[order]
        self._overlay_x = x
        self._overlay_y = y
        self._overlay_color = color
        self.refresh()

    def clear_overlay(self) -> None:
        self._overlay_x = None
        self._overlay_y = None
        self.refresh()

    def zoom_x(self, factor: float) -> None:
        cx = (self._x_min + self._x_max) / 2
        span = (self._x_max - self._x_min) * factor / 2
        self._x_min = max(0, cx - span)
        self._x_max = cx + span
        self.refresh()

    def render_line(self, y_row: int) -> Strip:
        w, h = self.size.width, self.size.height
        if len(self._x) == 0 or w < 10 or h < 4:
            return Strip.blank(w)

        left_margin = 8
        bottom_rows = 2
        plot_cols = w - left_margin
        plot_rows = h - bottom_rows - 1  # terminal rows for the plot area

        if plot_cols < 2 or plot_rows < 2:
            return Strip.blank(w)

        # Pixel dimensions (braille: 2 wide x 4 tall per cell)
        px_w = plot_cols * 2
        px_h = plot_rows * 4

        if y_row == 0:
            title = self._title[:w]
            pad = max(0, (w - len(title)) // 2)
            return Strip([Segment(" " * pad + title + " " * max(0, w - pad - len(title)))], w)

        if y_row > plot_rows:
            if y_row == plot_rows + 1:
                lo = f"{self._x_min:.1f}"
                hi = f"{self._x_max:.1f}"
                mid = f"{(self._x_min + self._x_max) / 2:.1f}"
                line = " " * left_margin + lo
                gap1 = max(1, plot_cols // 2 - len(lo) - len(mid) // 2)
                gap2 = max(1, plot_cols - len(lo) - gap1 - len(mid) - len(hi))
                line += " " * gap1 + mid + " " * gap2 + hi
                return Strip([Segment(line[:w])], w)
            elif y_row == plot_rows + 2:
                label = self._x_label
                pad = max(0, (w - len(label)) // 2)
                return Strip([Segment(" " * pad + label + " " * max(0, w - pad - len(label)))], w)
            return Strip.blank(w)

        # Plot area: render braille line graph for this row
        cell_row = y_row - 1

        x_range = self._x_max - self._x_min
        y_range = self._y_max - self._y_min
        if x_range <= 0:
            x_range = 1.0
        if y_range <= 0:
            y_range = 1.0

        py_top = cell_row * 4
        py_bot = py_top + 3

        def _rasterize(xdata, ydata, cells):
            prev_py = None
            for px in range(px_w):
                x_val = self._x_min + (px + 0.5) / px_w * x_range
                idx = np.searchsorted(xdata, x_val)
                if idx <= 0:
                    y_val = float(ydata[0])
                elif idx >= len(xdata):
                    y_val = float(ydata[-1])
                else:
                    x0, x1 = float(xdata[idx - 1]), float(xdata[idx])
                    y0, y1 = float(ydata[idx - 1]), float(ydata[idx])
                    t = (x_val - x0) / (x1 - x0) if x1 != x0 else 0
                    y_val = y0 + t * (y1 - y0)
                if not np.isfinite(y_val):
                    prev_py = None  # a gap: nothing drawn, and no line across it
                    continue
                py = int(px_h - 1 - (y_val - self._y_min) / y_range * (px_h - 1))
                py = max(0, min(px_h - 1, py))
                if prev_py is not None:
                    lo_py = min(prev_py, py)
                    hi_py = max(prev_py, py)
                else:
                    lo_py = hi_py = py
                for fill_py in range(lo_py, hi_py + 1):
                    if py_top <= fill_py <= py_bot:
                        col = px // 2
                        sub_x = px % 2
                        sub_y = fill_py - py_top
                        if sub_y < 3:
                            bit = sub_y + sub_x * 3
                        else:
                            bit = 6 + sub_x
                        cells[col] |= (1 << bit)
                prev_py = py

        braille_cells = np.zeros(plot_cols, dtype=np.uint8)
        _rasterize(self._x, self._y, braille_cells)

        overlay_cells = None
        if self._overlay_x is not None and self._overlay_y is not None and len(self._overlay_x) > 0:
            overlay_cells = np.zeros(plot_cols, dtype=np.uint8)
            _rasterize(self._overlay_x, self._overlay_y, overlay_cells)

        chars = [chr(0x2800 + int(b)) for b in braille_cells]

        # Y-axis label
        if cell_row == 0:
            y_lbl = f"{self._y_max:>7.2f} "
        elif cell_row == plot_rows - 1:
            y_lbl = f"{self._y_min:>7.2f} "
        elif cell_row == plot_rows // 2:
            mid_y = (self._y_max + self._y_min) / 2
            y_lbl = f"{mid_y:>7.2f} "
        else:
            y_lbl = " " * 8

        segments = [Segment(y_lbl, Style(color="rgb(150,150,150)"))]
        if overlay_cells is not None:
            # Interleave: where both have dots, show primary; overlay-only in overlay color
            overlay_chars = [chr(0x2800 + int(b)) for b in overlay_cells]
            for ci in range(plot_cols):
                if braille_cells[ci] and overlay_cells[ci]:
                    # Both: show primary over overlay
                    braille_cells[ci] & overlay_cells[ci]
                    braille_cells[ci] & ~overlay_cells[ci]
                    overlay_cells[ci] & ~braille_cells[ci]
                    # Combine: show primary color (overlay underneath is lost in braille)
                    segments.append(Segment(chars[ci], Style(color="rgb(80,200,255)")))
                elif braille_cells[ci]:
                    segments.append(Segment(chars[ci], Style(color="rgb(80,200,255)")))
                elif overlay_cells[ci]:
                    segments.append(Segment(overlay_chars[ci], Style(color=self._overlay_color)))
                else:
                    segments.append(Segment(chr(0x2800)))
        else:
            segments.append(Segment("".join(chars), Style(color="rgb(80,200,255)")))
        return Strip(segments, w)


# ---------------------------------------------------------------------------
# Base class for analysis screens — eliminates duplicate CSS, button handlers,
# CSV export, frame range slicing, and element/name filter updates.
# ---------------------------------------------------------------------------

_ANALYSIS_CSS = """
    {cls} > Vertical {{ width: 100%; height: 100%; background: $surface; padding: 0 1; }}
    {cls} Select {{ width: 16; margin: 0 1; }}
    {cls} SpinBox {{ width: 18; height: 3; margin: 0 1; }}
    {cls} Button {{ width: auto; min-width: 4; margin: 0 1; }}
    {cls} Label {{ width: auto; margin: 0; }}
    {cls} Checkbox {{ width: auto; margin: 0 1; }}
"""


class _AnalysisScreen(ModalScreen[None]):
    """Base class for analysis screens with common patterns."""

    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]

    # Subclasses set these
    _prefix = ""  # e.g. "msd", "rdf" — used for widget IDs
    _csv_header = ""
    _csv_columns: list[str] = []  # attribute names for CSV columns

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__()
        self._molecule = molecule
        self._frames = frames
        self._elements = sorted(set(a.element.symbol for a in molecule.atoms))
        self._compute_task: asyncio.Task | None = None

    def _spawn_task(self, coro) -> asyncio.Task:
        """Track the task so action_close can cancel it on dismiss.

        A failure inside the task is reported in the screen's status label and
        as a toast; without this an exception would only reach stderr at
        garbage-collection time and the status would stay at "Computing...".
        """
        if self._compute_task is not None and not self._compute_task.done():
            self._compute_task.cancel()
        label = _task_label(coro)
        self._compute_task = asyncio.create_task(coro)
        self._compute_task.add_done_callback(
            lambda t, label=label: self._on_compute_done(t, label)
        )
        return self._compute_task

    def _on_compute_done(self, task: asyncio.Task, label: str) -> None:
        exc = _task_failure(task)
        if exc is None:
            return
        msg = f"{label} failed: {exc}"
        try:
            self.query_one(f"#{self._prefix}-status", Label).update(f"Error: {exc}")
        except Exception:
            pass
        try:
            self.app.notify(msg, timeout=6, severity="error")
        except Exception:
            pass

    def _get_frame_slice(self) -> list:
        """Get frame range from Start/End SpinBoxes if present."""
        from pdb_wizard.analyses import frame_slice
        return frame_slice(self._frames, *self._frame_range())

    def _frame_range(self) -> tuple:
        """(start, end), 1-based inclusive, from the Start/End SpinBoxes; (1, None) without."""
        if self._frames and len(self._frames) > 1:
            try:
                return (self.query_one(f"#{self._prefix}-start", SpinBox).value,
                        self.query_one(f"#{self._prefix}-end", SpinBox).value)
            except Exception:
                pass
        return 1, None

    def _yield_frame_range(self, prefix: str = "") -> ComposeResult:
        """Yield Start/End SpinBox widgets for frame range selection."""
        p = prefix or self._prefix
        if self._frames and len(self._frames) > 1:
            n = len(self._frames)
            yield Label("Start:")
            yield SpinBox(value=1, min_val=1, max_val=n, id=f"{p}-start")
            yield Label("End:")
            yield SpinBox(value=n, min_val=1, max_val=n, id=f"{p}-end")

    def _update_name_select(self, element: str, name_select_id: str) -> None:
        """Update a name Select when element changes."""
        if element == "(all)":
            names = sorted(set(a.name.strip() for a in self._molecule.atoms))
        else:
            names = sorted(set(
                a.name.strip() for a in self._molecule.atoms
                if a.element.symbol == element
            ))
        sel = self.query_one(name_select_id, Select)
        sel.set_options([("(all)", "(all)")] + [(n, n) for n in names])
        sel.value = "(all)"

    def _do_export_csv(self, filepath: str, header: str, *arrays) -> None:
        """Write CSV with header and parallel arrays."""
        from pdb_wizard.analyses import write_csv
        if not filepath.strip():
            return
        path = filepath.strip()
        try:
            write_csv(path, header, *arrays)
        except OSError as e:
            # Raised inside a modal dismiss callback: an uncaught error here
            # would take down the whole app.
            self._safe_notify(f"Could not write {path}: {e}", severity="error")
            return
        self._safe_notify(f"Wrote {path}")

    def _safe_notify(self, message: str, severity: str = "information") -> None:
        """notify() via the app, tolerating a screen that is not mounted."""
        try:
            self.app.notify(message, timeout=6 if severity == "error" else 3,
                            severity=severity)
        except Exception:
            pass

    def action_close(self) -> None:
        if self._compute_task is not None and not self._compute_task.done():
            self._compute_task.cancel()
            self._compute_task = None
        self.dismiss(None)


class RdfScreen(_AnalysisScreen):
    """Full-screen RDF with element + molecule filtering, Y-axis control, zoom."""

    _prefix = "rdf"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="RdfScreen") + """
    RdfScreen #rdf-row1 { height: auto; margin-bottom: 1; }
    RdfScreen #rdf-row2 { height: auto; margin-bottom: 1; }
    RdfScreen PlotWidget { height: 1fr; }
    RdfScreen #rdf-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._names = sorted(set(a.name.strip() for a in molecule.atoms))
        self._last_r = np.array([])
        self._last_g = np.array([])
        self._last_title = ""

    def _export_csv(self) -> None:
        if len(self._last_r) == 0:
            self.query_one("#rdf-status", Label).update("No data to export.")
            return
        from pathlib import Path
        stem = Path(self._last_title.replace(" ", "_").replace("/", "-"))
        self.app.push_screen(
            FileSaveModal("Save RDF as CSV:", default=f"rdf_{stem}.csv"),
            callback=lambda fp: self._do_export_csv(fp, "r(A),g(r)", self._last_r, self._last_g),
        )

    def _names_for_element(self, el: str) -> list[str]:
        return sorted(set(a.name.strip() for a in self._molecule.atoms if a.element.symbol == el))

    def compose(self) -> ComposeResult:
        if not self._elements:
            self._elements = ["X"]
        el_opts = [(el, el) for el in self._elements]
        el1 = self._elements[0]
        el2 = self._elements[-1]
        names1 = self._names_for_element(el1)
        names2 = self._names_for_element(el2)
        name_opts1 = [("(all)", "(all)")] + [(n, n) for n in names1]
        name_opts2 = [("(all)", "(all)")] + [(n, n) for n in names2]
        with Vertical():
            with Horizontal(id="rdf-row1"):
                yield Label("Type 1:  El:")
                yield Select(el_opts, value=el1, id="rdf-el1", allow_blank=False)
                yield Label("  Name:")
                yield Select(name_opts1, value="(all)", id="rdf-name1", allow_blank=False)
                yield Label("    Type 2:  El:")
                yield Select(el_opts, value=el2, id="rdf-el2", allow_blank=False)
                yield Label("  Name:")
                yield Select(name_opts2, value="(all)", id="rdf-name2", allow_blank=False)
            with Horizontal(id="rdf-row2"):
                yield Label("Ymax:")
                yield SpinBox(value=5, min_val=1, max_val=100, id="rdf-ymax")
                yield Label("Xmax:")
                yield SpinBox(value=13, min_val=2, max_val=50, id="rdf-xmax")
                yield Label("Bins:")
                yield SpinBox(value=100, min_val=10, max_val=1000, step=10, id="rdf-bins")
                if self._frames and len(self._frames) > 1:
                    n = len(self._frames)
                    yield Checkbox("Traj", value=True, id="rdf-traj")
                    yield Label("Start:")
                    yield SpinBox(value=1, min_val=1, max_val=n, id="rdf-start")
                    yield Label("End:")
                    yield SpinBox(value=n, min_val=1, max_val=n, id="rdf-end")
                yield Button("Compute", id="rdf-compute", variant="primary")
                yield Button("CSV", id="rdf-csv")
                yield Button("Close", id="rdf-close")
            yield PlotWidget(id="rdf-plot")
            yield Label("Select atom types and click Compute  |  +/- zoom x  |  r reset", id="rdf-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "rdf-compute":
            self._spawn_task(self._compute_rdf_async())
        elif bid == "rdf-csv":
            self._export_csv()
        elif bid == "rdf-close":
            self.action_close()

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id
        if sid == "rdf-el1":
            self._update_name_select(str(event.value), "#rdf-name1")
        elif sid == "rdf-el2":
            self._update_name_select(str(event.value), "#rdf-name2")

    def on_spin_box_changed(self, event: SpinBox.Changed) -> None:
        sid = event.spin_box.id or ""
        plot = self.query_one("#rdf-plot", PlotWidget)
        if sid == "rdf-ymax":
            # Just update the plot Y range, no recompute
            if len(plot._x) > 0:
                plot._y_max = float(event.value)
                plot.refresh()
        elif sid in ("rdf-xmax", "rdf-bins"):
            # Recompute with new params if we have data
            if len(plot._x) > 0:
                self._spawn_task(self._compute_rdf_async())

    def _get_atom_filter(self, el_id: str, name_id: str):
        """Return a function that tests if an atom matches the filter."""
        el = self.query_one(f"#{el_id}", Select).value
        name = self.query_one(f"#{name_id}", Select).value
        if name == "(all)":
            return lambda a: a.element.symbol == el
        else:
            return lambda a: a.element.symbol == el and a.name.strip() == name

    async def _compute_rdf_async(self) -> None:
        from pdb_wizard.analyses import run_rdf

        el1 = str(self.query_one("#rdf-el1", Select).value)
        el2 = str(self.query_one("#rdf-el2", Select).value)
        name1 = str(self.query_one("#rdf-name1", Select).value)
        name2 = str(self.query_one("#rdf-name2", Select).value)
        label1 = el1 if name1 == "(all)" else name1
        label2 = el2 if name2 == "(all)" else name2

        y_max = float(self.query_one("#rdf-ymax", SpinBox).value)
        x_max = float(self.query_one("#rdf-xmax", SpinBox).value)
        n_bins = self.query_one("#rdf-bins", SpinBox).value

        use_traj = False
        try:
            use_traj = bool(self.query_one("#rdf-traj", Checkbox).value
                            and self._frames and len(self._frames) > 1)
        except Exception:
            pass
        start, end = self._frame_range() if use_traj else (1, None)

        status = self.query_one("#rdf-status", Label)
        n_total = (end - start + 1) if use_traj else 1

        # Shared progress state polled by a timer
        self._rdf_progress = [0.0]

        def _update_status() -> None:
            frac = self._rdf_progress[0]
            if use_traj:
                frame_num = int(frac * n_total)
                status.update(f"Computing g(r) {label1}-{label2}... frame {frame_num}/{n_total}")
            else:
                status.update(f"Computing g(r) {label1}-{label2}... {int(frac * 100)}%")

        def _prog(frac: float) -> None:
            self._rdf_progress[0] = frac

        poll_timer = self.set_interval(0.1, _update_status)
        status.update(f"Computing g(r) {label1}-{label2}... 0%")
        await asyncio.sleep(0)

        try:
            res = await asyncio.to_thread(
                run_rdf, self._molecule, el1, el2, name1, name2, x_max, n_bins,
                self._frames, start, end, use_traj, _prog,
            )
        finally:
            poll_timer.stop()

        if res.empty:
            status.update(res.summary)
            return

        # Store for CSV export
        self._last_r = res.x
        self._last_g = res.y
        self._last_title = res.title

        plot = self.query_one("#rdf-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0, y_max=y_max)
        status.update(res.summary)

class MsdScreen(_AnalysisScreen):
    """MSD analysis with element selector, frame range, and diffusion constant."""
    _prefix = "msd"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="MsdScreen") + """
    MsdScreen #msd-controls { height: auto; margin-bottom: 1; }
    MsdScreen #msd-row2 { height: auto; margin-bottom: 1; }
    MsdScreen #msd-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_lags = np.array([])
        self._last_msd = np.array([])

    def compose(self) -> ComposeResult:
        el_opts = [(el, el) for el in self._elements]
        names = sorted(set(a.name.strip() for a in self._molecule.atoms))
        name_opts = [("(all)", "(all)")] + [(n, n) for n in names]
        with Vertical():
            with Horizontal(id="msd-controls"):
                yield Label("Element:")
                yield Select(el_opts, value=self._elements[-1] if self._elements else "H",
                             id="msd-el", allow_blank=False)
                yield Label("Name:")
                yield Select(name_opts, value="(all)", id="msd-name", allow_blank=False)
                yield Button("Compute", id="msd-compute", variant="primary")
                yield Button("CSV", id="msd-csv")
                yield Button("Close", id="msd-close")
            with Horizontal(id="msd-row2"):
                yield Label("dt (ps):")
                yield SpinBox(value=1, min_val=1, max_val=10000, id="msd-dt")
                yield from self._yield_frame_range()
            yield PlotWidget(id="msd-plot")
            yield Label("Select atom type and click Compute", id="msd-status")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "msd-el":
            self._update_name_select(str(event.value), "#msd-name")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "msd-compute":
            self._spawn_task(self._run_msd())
        elif event.button.id == "msd-close":
            self.action_close()
        elif event.button.id == "msd-csv":
            if len(self._last_lags) == 0:
                return
            self.app.push_screen(FileSaveModal("Save MSD as CSV:", default="msd.csv"), callback=self._export_csv)

    async def _run_msd(self) -> None:
        from pdb_wizard.analyses import run_msd
        el = str(self.query_one("#msd-el", Select).value)
        name = str(self.query_one("#msd-name", Select).value)
        dt = float(self.query_one("#msd-dt", SpinBox).value)
        status = self.query_one("#msd-status", Label)
        status.update(f"Computing MSD for {el}...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(run_msd, self._frames, el, name, dt, *self._frame_range())
        if res.empty:
            status.update(res.summary)
            return
        self._last_lags, self._last_msd = res.csv_arrays()
        plot = self.query_one("#msd-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)
        if res.text:  # Monte Carlo trajectory: the curve is not diffusion
            self.app.notify(res.text, severity="warning", timeout=15)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "lag_frames,msd_A2", self._last_lags, self._last_msd)


class CoordinationScreen(_AnalysisScreen):
    """Coordination number analysis."""
    _prefix = "coord"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="CoordinationScreen") + """
    CoordinationScreen #coord-controls { height: auto; margin-bottom: 1; }
    CoordinationScreen #coord-row2 { height: auto; margin-bottom: 1; }
    CoordinationScreen #coord-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_r = np.array([])
        self._last_cn = np.array([])

    def compose(self) -> ComposeResult:
        el_opts = [(el, el) for el in self._elements]
        el1 = self._elements[0] if self._elements else "X"
        el2 = self._elements[-1] if self._elements else "X"
        with Vertical():
            with Horizontal(id="coord-controls"):
                yield Label("Center:")
                yield Select(el_opts, value=el1, id="coord-el1", allow_blank=False)
                yield Label("Neighbor:")
                yield Select(el_opts, value=el2, id="coord-el2", allow_blank=False)
                yield Label("Rmax:")
                yield SpinBox(value=5, min_val=2, max_val=20, id="coord-rmax")
                yield Button("Compute", id="coord-compute", variant="primary")
                yield Button("CSV", id="coord-csv")
                yield Button("Close", id="coord-close")
            with Horizontal(id="coord-row2"):
                if self._frames and len(self._frames) > 1:
                    yield Checkbox("Traj", value=False, id="coord-traj")
                    yield from self._yield_frame_range()
            yield PlotWidget(id="coord-plot")
            yield Label("Select center and neighbor elements, then Compute", id="coord-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "coord-compute":
            self._spawn_task(self._run_coord())
        elif event.button.id == "coord-close":
            self.action_close()
        elif event.button.id == "coord-csv":
            if len(self._last_r) == 0:
                return
            self.app.push_screen(FileSaveModal("Save coordination as CSV:", default="coordination.csv"), callback=self._export_csv)

    async def _run_coord(self) -> None:
        from pdb_wizard.analyses import run_coordination
        el1 = str(self.query_one("#coord-el1", Select).value)
        el2 = str(self.query_one("#coord-el2", Select).value)
        r_max = float(self.query_one("#coord-rmax", SpinBox).value)
        status = self.query_one("#coord-status", Label)
        use_traj = False
        try:
            use_traj = bool(self.query_one("#coord-traj", Checkbox).value
                            and self._frames and len(self._frames) > 1)
        except Exception:
            pass
        if use_traj:
            start, end = self._frame_range()
            n = len(self._get_frame_slice())
            status.update(f"Computing N({el2}) around {el1} over {n} frames...")
        else:
            start, end = 1, None
            status.update(f"Computing N({el2}) around {el1}...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(
            run_coordination, self._molecule, el1, el2, r_max, self._frames, start, end, use_traj,
        )
        if res.empty:
            status.update(res.summary)
            return
        self._last_r = res.x
        self._last_cn = res.y
        plot = self.query_one("#coord-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "r_A,coordination_number", self._last_r, self._last_cn)


class RmsdScreen(_AnalysisScreen):
    """RMSD vs time analysis."""
    _prefix = "rmsd"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="RmsdScreen") + """
    RmsdScreen #rmsd-controls { height: auto; margin-bottom: 1; }
    RmsdScreen #rmsd-row2 { height: auto; margin-bottom: 1; }
    RmsdScreen #rmsd-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_idx = np.array([])
        self._last_rmsd = np.array([])

    def compose(self) -> ComposeResult:
        el_opts = [("(all)", "(all)")] + [(el, el) for el in self._elements]
        names = sorted(set(a.name.strip() for a in self._molecule.atoms))
        name_opts = [("(all)", "(all)")] + [(n, n) for n in names]
        with Vertical():
            with Horizontal(id="rmsd-controls"):
                yield Label("Element:")
                yield Select(el_opts, value="(all)", id="rmsd-el", allow_blank=False)
                yield Label("Name:")
                yield Select(name_opts, value="(all)", id="rmsd-name", allow_blank=False)
                yield Button("Compute", id="rmsd-compute", variant="primary")
                yield Button("CSV", id="rmsd-csv")
                yield Button("Close", id="rmsd-close")
            with Horizontal(id="rmsd-row2"):
                yield Label("dt (ps):")
                yield SpinBox(value=1, min_val=1, max_val=10000, id="rmsd-dt")
                yield from self._yield_frame_range()
            yield PlotWidget(id="rmsd-plot")
            yield Label("Select atom filter and click Compute", id="rmsd-status")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "rmsd-el":
            self._update_name_select(str(event.value), "#rmsd-name")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rmsd-compute":
            self._spawn_task(self._run_rmsd())
        elif event.button.id == "rmsd-close":
            self.action_close()
        elif event.button.id == "rmsd-csv":
            if len(self._last_idx) == 0:
                return
            self.app.push_screen(FileSaveModal("Save RMSD as CSV:", default="rmsd.csv"), callback=self._export_csv)

    async def _run_rmsd(self) -> None:
        from pdb_wizard.analyses import run_rmsd
        el = str(self.query_one("#rmsd-el", Select).value)
        name = str(self.query_one("#rmsd-name", Select).value)
        dt = float(self.query_one("#rmsd-dt", SpinBox).value)
        status = self.query_one("#rmsd-status", Label)
        status.update(f"Computing RMSD for {el if el != '(all)' else 'all'}...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(run_rmsd, self._frames, el, name, dt, *self._frame_range())
        if res.empty:
            status.update(res.summary)
            return
        self._last_idx, self._last_rmsd = res.csv_arrays()
        plot = self.query_one("#rmsd-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)
        if res.text:  # Monte Carlo trajectory: the curve has no time axis
            self.app.notify(res.text, severity="warning", timeout=15)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "frame,rmsd_A", self._last_idx, self._last_rmsd)


class HbondScreen(_AnalysisScreen):
    """Hydrogen bond count over trajectory."""
    _prefix = "hb"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="HbondScreen") + """
    HbondScreen #hb-controls { height: auto; margin-bottom: 1; }
    HbondScreen #hb-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_idx = np.array([])
        self._last_counts = np.array([])

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="hb-controls"):
                yield Label("D-A cutoff:")
                yield SpinBox(value=35, min_val=20, max_val=50, id="hb-cutoff")
                yield Label("/10 A")
                yield Label("  Angle min:")
                yield SpinBox(value=120, min_val=90, max_val=170, id="hb-angle")
                yield Label("deg")
                yield from self._yield_frame_range()
                yield Button("Compute", id="hb-compute", variant="primary")
                yield Button("CSV", id="hb-csv")
                yield Button("Close", id="hb-close")
            yield PlotWidget(id="hb-plot")
            yield Label("Click Compute to count H-bonds per frame", id="hb-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hb-compute":
            self._spawn_task(self._run())
        elif event.button.id == "hb-close":
            self.action_close()
        elif event.button.id == "hb-csv":
            if len(self._last_idx) == 0:
                return
            self.app.push_screen(FileSaveModal("Save H-bonds as CSV:", default="hbonds.csv"), callback=self._export_csv)

    async def _run(self) -> None:
        from pdb_wizard.analyses import run_hbonds
        cutoff = self.query_one("#hb-cutoff", SpinBox).value / 10.0
        angle = float(self.query_one("#hb-angle", SpinBox).value)
        status = self.query_one("#hb-status", Label)
        if self._frames and len(self._frames) > 1:
            status.update(f"Counting H-bonds over {len(self._get_frame_slice())} frames...")
        else:
            status.update("Detecting H-bonds...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(run_hbonds, self._molecule, cutoff, angle, self._frames,
                                      *self._frame_range())
        if res.empty:
            status.update(res.summary)
            return
        self._last_idx = res.x
        self._last_counts = res.y
        plot = self.query_one("#hb-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "frame,hbond_count", self._last_idx, self._last_counts)


class GyrationScreen(_AnalysisScreen):
    """Radius of gyration vs time."""
    _prefix = "rg"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="GyrationScreen") + """
    GyrationScreen #rg-controls { height: auto; margin-bottom: 1; }
    GyrationScreen #rg-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_idx = np.array([])
        self._last_rg = np.array([])

    def compose(self) -> ComposeResult:
        el_opts = [("(all)", "(all)")] + [(el, el) for el in self._elements]
        with Vertical():
            with Horizontal(id="rg-controls"):
                yield Label("Element:")
                yield Select(el_opts, value="(all)", id="rg-el", allow_blank=False)
                yield Label("dt (ps):")
                yield SpinBox(value=1, min_val=1, max_val=10000, id="rg-dt")
                yield from self._yield_frame_range()
                yield Button("Compute", id="rg-compute", variant="primary")
                yield Button("CSV", id="rg-csv")
                yield Button("Close", id="rg-close")
            yield PlotWidget(id="rg-plot")
            yield Label("Select filter and click Compute", id="rg-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rg-compute":
            self._spawn_task(self._run())
        elif event.button.id == "rg-close":
            self.action_close()
        elif event.button.id == "rg-csv":
            if len(self._last_idx) == 0:
                return
            self.app.push_screen(FileSaveModal("Save Rg as CSV:", default="rg.csv"), callback=self._export_csv)

    async def _run(self) -> None:
        from pdb_wizard.analyses import run_gyration
        el = str(self.query_one("#rg-el", Select).value)
        dt = float(self.query_one("#rg-dt", SpinBox).value)
        status = self.query_one("#rg-status", Label)
        status.update(f"Computing Rg for {el if el != '(all)' else 'all'}...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(run_gyration, self._frames, el, dt, *self._frame_range())
        if res.empty:
            status.update(res.summary)
            return
        self._last_idx, self._last_rg = res.csv_arrays()
        plot = self.query_one("#rg-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=res.y_min,
                      y_max=res.y_max)
        status.update(res.summary)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "frame,rg_A", self._last_idx, self._last_rg)


class ReduceCellScreen(ModalScreen[Optional[Molecule]]):
    """Reduce a supercell to a primitive cell, selecting which sorbates to keep."""
    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]
    DEFAULT_CSS = """
    ReduceCellScreen > Vertical { width: 100%; height: 100%; background: $surface; padding: 1 2; }
    ReduceCellScreen #rc-controls { height: auto; margin-bottom: 1; }
    ReduceCellScreen SpinBox { width: 10; height: 3; margin: 0 1; }
    ReduceCellScreen Button { width: auto; min-width: 6; margin: 0 1; }
    ReduceCellScreen Label { width: auto; margin: 0; }
    ReduceCellScreen DataTable { height: 1fr; }
    ReduceCellScreen #rc-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__()
        self._molecule = molecule
        self._framework_idx: list[int] = []
        self._sorbate_mols: list[list[int]] = []
        self._keep: list[bool] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="rc-controls"):
                yield Label("Supercell:")
                yield SpinBox(value=2, min_val=1, max_val=8, id="rc-na")
                yield Label("x")
                yield SpinBox(value=2, min_val=1, max_val=8, id="rc-nb")
                yield Label("x")
                yield SpinBox(value=2, min_val=1, max_val=8, id="rc-nc")
                yield Button("Analyze", id="rc-analyze", variant="primary")
                yield Button("Reduce", id="rc-reduce")
                yield Button("Close", id="rc-close")
            yield DataTable(id="rc-table")
            yield Label("Set supercell dimensions and click Analyze", id="rc-status")

    def on_mount(self) -> None:
        table = self.query_one("#rc-table", DataTable)
        table.cursor_type = "row"  # Enter / click toggles a whole row
        table.add_columns("Keep", "#", "Atoms", "Elements", "Center (A)")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rc-analyze":
            self._do_analyze()
        elif event.button.id == "rc-reduce":
            self._do_reduce()
        elif event.button.id == "rc-close":
            self.dismiss(None)

    def _do_analyze(self) -> None:
        from pdb_wizard.reduce_cell import identify_framework_and_sorbates
        na = self.query_one("#rc-na", SpinBox).value
        nb = self.query_one("#rc-nb", SpinBox).value
        nc = self.query_one("#rc-nc", SpinBox).value
        status = self.query_one("#rc-status", Label)

        try:
            fw, sorbs = identify_framework_and_sorbates(self._molecule, na, nb, nc)
        except ValueError as e:  # not an na x nb x nc supercell
            self._framework_idx, self._sorbate_mols, self._keep = [], [], []
            self.query_one("#rc-table", DataTable).clear()
            status.update(str(e))
            return
        self._framework_idx = fw
        self._sorbate_mols = sorbs
        self._keep = [True] * len(sorbs)

        table = self.query_one("#rc-table", DataTable)
        table.clear()

        n_mult = na * nb * nc
        status_msg = f"Framework: {len(fw)} unique sites ({len(fw) * n_mult} total atoms)"
        status_msg += f"  |  Sorbate molecules: {len(sorbs)}"

        for i, mol_indices in enumerate(sorbs):
            elements = {}
            positions = []
            for idx in mol_indices:
                a = self._molecule.atoms[idx]
                elements[a.element.symbol] = elements.get(a.element.symbol, 0) + 1
                positions.append(a.x)
            elem_str = " ".join(f"{el}{n}" for el, n in sorted(elements.items()))
            center = np.mean(positions, axis=0) if positions else np.zeros(3)
            center_str = f"{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}"
            table.add_row("[X]", str(i + 1), str(len(mol_indices)), elem_str, center_str,
                          key=str(i))

        status.update(status_msg + "  |  Click rows to toggle keep/discard, then Reduce")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Toggle keep/discard on row click."""
        table = self.query_one("#rc-table", DataTable)
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._keep):
            self._keep[row_idx] = not self._keep[row_idx]
            mark = "[X]" if self._keep[row_idx] else "[ ]"
            keep_column = next(iter(table.columns))
            table.update_cell(event.row_key, keep_column, mark)

    def _do_reduce(self) -> None:
        if not self._framework_idx and not self._sorbate_mols:
            self.app.notify("Click Analyze first", timeout=2)
            return
        from pdb_wizard.reduce_cell import reduce_to_primitive
        na = self.query_one("#rc-na", SpinBox).value
        nb = self.query_one("#rc-nb", SpinBox).value
        nc = self.query_one("#rc-nc", SpinBox).value
        keep_indices = [i for i, k in enumerate(self._keep) if k]
        try:
            result = reduce_to_primitive(self._molecule, na, nb, nc, keep_indices)
        except ValueError as e:  # the sizes were changed after Analyze
            self.query_one("#rc-status", Label).update(str(e))
            return
        n_kept = len(keep_indices)
        n_total = len(self._sorbate_mols)
        self.app.notify(
            f"Reduced to primitive cell: {len(result.atoms)} atoms, "
            f"kept {n_kept}/{n_total} sorbate molecules",
            timeout=3,
        )
        self.dismiss(result)

    def action_close(self) -> None:
        self.dismiss(None)


class InputGeneratorScreen(ModalScreen[None]):
    """Unified template-driven input generator.

    Renders an input file (and any companion files) from a template under
    `pdb_wizard.templates`. Engines:
      - mpmc_isotherm  : multi-pressure μVT MC tree
      - cp2k_cell_opt  : CP2K cell optimization
      - cp2k_dft_md    : CP2K Born-Oppenheimer NVT MD
      - openmm_npt     : OpenMM NPT MD driver (Python)
      - openmm_nvt     : OpenMM NVT MD driver (Python)

    The screen reads its form definition from `templates.ENGINES`. Switching
    the engine rebuilds the form. Edit the shipped template files in place
    to tweak the rendered output beyond what the form exposes.
    """

    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]
    DEFAULT_CSS = """
    InputGeneratorScreen > Vertical { width: 100%; height: 100%; background: $surface; padding: 1 2; }
    InputGeneratorScreen #gen-row-engine { height: auto; margin-bottom: 1; }
    InputGeneratorScreen #gen-row-outdir { height: auto; margin-bottom: 1; }
    InputGeneratorScreen #gen-row-buttons { height: auto; margin-top: 1; margin-bottom: 1; }
    InputGeneratorScreen #gen-form { height: auto; max-height: 16; margin-bottom: 1; }
    InputGeneratorScreen .gen-form-row { height: auto; margin-bottom: 1; }
    InputGeneratorScreen Select { width: 24; margin: 0 1; }
    InputGeneratorScreen #gen-engine { width: 24; }
    InputGeneratorScreen SpinBox { width: 18; height: 3; margin: 0 1; }
    InputGeneratorScreen Input { width: 1fr; margin: 0 1; }
    InputGeneratorScreen Button { width: auto; min-width: 6; margin: 0 1; }
    InputGeneratorScreen Label { width: auto; margin: 0; }
    InputGeneratorScreen .gen-var-label { width: 28; }
    InputGeneratorScreen TextArea { height: 1fr; }
    InputGeneratorScreen #gen-status { height: 1; color: $text-muted; }
    InputGeneratorScreen #gen-description { height: 1; color: $text-muted; }
    """

    def __init__(
        self,
        molecule: Molecule,
        filepath: str = "",
        engine: str = "mpmc_isotherm",
    ) -> None:
        super().__init__()
        from pdb_wizard.templates import ENGINES
        if engine not in ENGINES:
            raise ValueError(f"unknown engine key: {engine!r}")
        self._molecule = molecule
        self._filepath = filepath
        self._engine_key = engine
        # widget refs for value extraction, populated by _build_form
        self._form_widgets: dict[str, Widget] = {}
        # cached current values per engine so switching back restores them
        self._values_by_engine: dict[str, dict] = {}

    # ---------------------------------------------------------------
    # Compose
    # ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        from pdb_wizard.templates import ENGINES
        engine_opts = [(spec.label, key) for key, spec in ENGINES.items()]
        with Vertical():
            with Horizontal(id="gen-row-engine"):
                yield Label("Engine:")
                yield Select(engine_opts, value=self._engine_key, id="gen-engine",
                             allow_blank=False)
                yield Label("", id="gen-description")
                yield Button("Close", id="gen-close")
            with Horizontal(id="gen-row-outdir"):
                yield Label("Output dir:")
                yield Input(value=self._default_outdir(), id="gen-outdir")
            yield Vertical(id="gen-form")
            with Horizontal(id="gen-row-buttons"):
                yield Button("Validate", id="gen-validate")
                yield Button("Show template", id="gen-show-template")
                yield Button("Generate", id="gen-generate", variant="primary")
            yield TextArea(id="gen-preview", read_only=True)
            yield Label("Pick an engine, set values, then Generate.", id="gen-status")

    def _default_outdir(self) -> str:
        """Folder name next to the loaded file (or cwd) for the current engine."""
        from pdb_wizard.inputgen import default_engine_outdir
        return default_engine_outdir(self._filepath, self._engine_key)

    # ---------------------------------------------------------------
    # Mount + form building
    # ---------------------------------------------------------------
    def on_mount(self) -> None:
        self._build_form()
        self._update_description()
        # If MPMC, run validation immediately so the user sees errors
        if self._engine_key == "mpmc_isotherm":
            self._show_validation()
        else:
            self._set_status("Edit values, then Generate.")
            self._set_preview(
                f"Engine: {self._engine_key}\n"
                f"Output file: {self._engine_spec().output_filename}\n\n"
                f"Click Show template to inspect the source template, "
                f"or Generate to write the rendered files."
            )

    def _engine_spec(self):
        from pdb_wizard.templates import ENGINES
        return ENGINES[self._engine_key]

    def _build_form(self) -> None:
        """(Re)build the dynamic form area for the current engine."""
        form = self.query_one("#gen-form", Vertical)
        # Tear down existing children
        for child in list(form.children):
            child.remove()
        self._form_widgets.clear()
        spec = self._engine_spec()
        # Persisted values (if user switched away and back)
        persisted = self._values_by_engine.get(self._engine_key, {})

        for var in spec.user_vars:
            widget_id = f"gen-{var.name}"
            value = persisted.get(var.name, var.default)
            row = Horizontal(classes="gen-form-row")
            form.mount(row)
            row.mount(Label(var.label + ":", classes="gen-var-label"))
            widget: Widget
            if var.kind == "select":
                choices = var.choices or ()
                widget = Select(
                    [(c, c) for c in choices],
                    value=value if value in choices else (choices[0] if choices else ""),
                    id=widget_id,
                    allow_blank=not choices,
                )
            elif var.kind == "int":
                widget = SpinBox(
                    value=int(value),
                    min_val=int(var.min_val) if var.min_val is not None else 0,
                    max_val=int(var.max_val) if var.max_val is not None else 99,
                    step=int(var.step) if var.step is not None else 1,
                    id=widget_id,
                )
            else:
                # 'float', 'str', 'csv_floats' all map to Input
                widget = Input(value=str(value), id=widget_id)
            row.mount(widget)
            self._form_widgets[var.name] = widget

    def _update_description(self) -> None:
        try:
            self.query_one("#gen-description", Label).update(
                self._engine_spec().description
            )
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Event handling
    # ---------------------------------------------------------------
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "gen-engine":
            return
        # Snapshot the outgoing engine's values before swapping
        self._snapshot_values()
        new_key = str(event.value)
        if new_key == self._engine_key:
            return
        self._engine_key = new_key
        # Reset outdir to engine default
        self.query_one("#gen-outdir", Input).value = self._default_outdir()
        self._build_form()
        self._update_description()
        if new_key == "mpmc_isotherm":
            self._show_validation()
        else:
            self._set_status("Edit values, then Generate.")
            self._set_preview(
                f"Engine: {new_key}\nOutput: {self._engine_spec().output_filename}"
            )

    def _snapshot_values(self) -> None:
        try:
            self._values_by_engine[self._engine_key] = self._collect_values(strict=False)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "gen-close":
            self.dismiss(None)
        elif bid == "gen-validate":
            self._show_validation()
        elif bid == "gen-show-template":
            self._show_template()
        elif bid == "gen-generate":
            self._do_generate()

    def action_close(self) -> None:
        self.dismiss(None)

    # ---------------------------------------------------------------
    # Value collection
    # ---------------------------------------------------------------
    def _collect_values(self, *, strict: bool = True) -> dict:
        """Read each form widget back to its declared type."""
        from pdb_wizard.inputgen import parse_form_value
        out: dict = {}
        for var in self._engine_spec().user_vars:
            w = self._form_widgets.get(var.name)
            if w is None:
                if strict:
                    raise KeyError(f"missing form widget for {var.name!r}")
                continue
            try:
                out[var.name] = parse_form_value(var, w.value)
            except ValueError:
                if strict:
                    raise
        return out

    # ---------------------------------------------------------------
    # Status / preview helpers
    # ---------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#gen-status", Label).update(text)
        except Exception:
            pass

    def _set_preview(self, text: str) -> None:
        try:
            self.query_one("#gen-preview", TextArea).load_text(text)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # MPMC validation
    # ---------------------------------------------------------------
    def _show_validation(self) -> None:
        from pdb_wizard.inputgen import check_engine
        self._show_outcome(check_engine(self._molecule, self._engine_key,
                                        self._collect_values(strict=False)))

    def _show_outcome(self, outcome) -> None:
        if outcome.preview is not None:
            self._set_preview(outcome.preview)
        self._set_status(outcome.status)

    # ---------------------------------------------------------------
    # Show template
    # ---------------------------------------------------------------
    def _show_template(self) -> None:
        from pdb_wizard.inputgen import engine_template_text
        self._show_outcome(engine_template_text(self._engine_key))

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------
    def _do_generate(self) -> None:
        from pdb_wizard.inputgen import generate_engine_inputs
        try:
            values = self._collect_values(strict=True)
        except (ValueError, KeyError) as e:
            self._set_status(f"Invalid input: {e}")
            return
        outdir = self.query_one("#gen-outdir", Input).value.strip()
        outcome = generate_engine_inputs(self._molecule, self._engine_key, values, outdir)
        self._show_outcome(outcome)
        if outcome.status.startswith("Refusing"):
            self.app.notify(
                "Cannot generate — fix errors shown in the preview",
                severity="error", timeout=5,
            )

# Backward-compat: existing imports of IsothermPlannerScreen keep working.
# Subclass so the engine arg defaults to MPMC isotherm without callers needing
# to pass it.
class IsothermPlannerScreen(InputGeneratorScreen):
    def __init__(self, molecule: Molecule, filepath: str = "") -> None:
        super().__init__(molecule, filepath, engine="mpmc_isotherm")


class DatabaseSearchScreen(ModalScreen[None]):
    """Unified database search.

    One screen for COD, RCSB PDB, PubChem, and Materials Project.
    Per-source search modes (text / formula / direct ID), source-aware
    placeholder hints, per-source result counts, and a preview line for
    the currently-selected row before fetching."""
    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]
    DEFAULT_CSS = """
    DatabaseSearchScreen > Vertical { width: 100%; height: 100%; background: $surface; padding: 1 2; }
    DatabaseSearchScreen #db-row1 { height: auto; margin-bottom: 1; }
    DatabaseSearchScreen #db-row2 { height: auto; margin-bottom: 1; }
    DatabaseSearchScreen Input { width: 1fr; margin: 0 1; }
    DatabaseSearchScreen #db-source { width: 22; margin: 0 1; }
    DatabaseSearchScreen #db-mode { width: 14; margin: 0 1; }
    DatabaseSearchScreen Button { width: auto; min-width: 6; margin: 0 1; }
    DatabaseSearchScreen Label { width: auto; margin: 0; }
    DatabaseSearchScreen DataTable { height: 1fr; }
    DatabaseSearchScreen #db-preview { height: 1; color: $text-muted; margin-top: 1; }
    DatabaseSearchScreen #db-status { height: 1; color: $text-muted; }
    """

    # Modes the "All"-source aggregate accepts. We dispatch the user's query
    # to every backend that supports the chosen mode.
    _ALL_MODES = ("text", "formula")

    def __init__(self) -> None:
        super().__init__()
        self._results: list[dict] = []
        self._compute_task: asyncio.Task | None = None

    def _spawn_task(self, coro) -> asyncio.Task:
        if self._compute_task is not None and not self._compute_task.done():
            self._compute_task.cancel()
        label = _task_label(coro)
        self._compute_task = asyncio.create_task(coro)
        self._compute_task.add_done_callback(
            lambda t, label=label: self._on_compute_done(t, label)
        )
        return self._compute_task

    def _on_compute_done(self, task: asyncio.Task, label: str) -> None:
        exc = _task_failure(task)
        if exc is None:
            return
        try:
            self.query_one("#db-status", Label).update(f"Error: {exc}")
        except Exception:
            pass
        try:
            self.app.notify(f"{label} failed: {exc}", timeout=6, severity="error")
        except Exception:
            pass

    def _modes_for(self, source: str) -> tuple[str, ...]:
        from pdb_wizard.database import modes_for
        return modes_for(source)

    def _placeholder_for(self, source: str, mode: str) -> str:
        from pdb_wizard.database import placeholder_for
        return placeholder_for(source, mode)

    def compose(self) -> ComposeResult:
        from pdb_wizard.database import BACKENDS
        db_opts = [("All databases", "all")] + [
            (spec["label"], code) for code, spec in BACKENDS.items()
        ]
        # Default to "all" + "text" — the broadest, most useful start.
        default_source = "all"
        default_mode = "text"
        mode_opts = [(m.capitalize(), m) for m in self._modes_for(default_source)]
        with Vertical():
            with Horizontal(id="db-row1"):
                yield Label("Database:")
                yield Select(db_opts, value=default_source, id="db-source", allow_blank=False)
                yield Label("Search by:")
                yield Select(mode_opts, value=default_mode, id="db-mode", allow_blank=False)
                yield Button("Close", id="db-close")
            with Horizontal(id="db-row2"):
                yield Input(
                    placeholder=self._placeholder_for(default_source, default_mode),
                    id="db-query",
                )
                yield Button("Search", id="db-search", variant="primary")
                yield Button("Fetch by ID", id="db-fetch-id")
                yield Button("Fetch selected", id="db-fetch")
            yield DataTable(id="db-results")
            yield Label("", id="db-preview")
            yield Label(
                "Pick a source + mode → type a query → Search. "
                "Or 'Fetch by ID' to skip search.",
                id="db-status",
            )

    def on_mount(self) -> None:
        table = self.query_one("#db-results", DataTable)
        table.add_columns("Source", "ID", "Formula", "Title")
        table.cursor_type = "row"
        # Auto-update the preview when the user moves the cursor
        table.show_cursor = True

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id
        if sid == "db-source":
            # Refresh the mode dropdown to match the newly-picked source
            new_source = str(event.value)
            new_modes = self._modes_for(new_source)
            mode_select = self.query_one("#db-mode", Select)
            current_mode = str(mode_select.value)
            mode_select.set_options([(m.capitalize(), m) for m in new_modes])
            # Preserve mode if still valid; otherwise pick first
            mode_select.value = current_mode if current_mode in new_modes else new_modes[0]
            self._update_placeholder()
        elif sid == "db-mode":
            self._update_placeholder()

    def _update_placeholder(self) -> None:
        try:
            source = str(self.query_one("#db-source", Select).value)
            mode = str(self.query_one("#db-mode", Select).value)
            self.query_one("#db-query", Input).placeholder = self._placeholder_for(source, mode)
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event) -> None:
        # Update preview pane
        idx = event.cursor_row
        if idx is None or idx < 0 or idx >= len(self._results):
            return
        r = self._results[idx]
        from pdb_wizard.database import BACKENDS
        src = BACKENDS.get(r.get("source", ""), {}).get("label", r.get("source", ""))
        formula = r.get("formula", "")
        title = r.get("title", "")
        bits = [f"[bold]{src}[/bold] {r.get('id', '')}"]
        if formula:
            bits.append(f"formula: {formula}")
        if title:
            bits.append(f"title: {title}")
        try:
            self.query_one("#db-preview", Label).update("  •  ".join(bits))
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "db-query":
            self._spawn_task(self._do_search())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "db-search":
            self._spawn_task(self._do_search())
        elif bid == "db-close":
            self.action_close()
        elif bid == "db-fetch":
            self._fetch_selected()
        elif bid == "db-fetch-id":
            self._fetch_by_id()

    async def _do_search(self) -> None:
        from pdb_wizard.database import BACKENDS

        query = self.query_one("#db-query", Input).value.strip()
        source = str(self.query_one("#db-source", Select).value)
        mode = str(self.query_one("#db-mode", Select).value)

        # Direct-ID mode skips search → pop the user straight into a fetch
        if mode == "id":
            if not query:
                self.query_one("#db-status", Label).update(
                    "Enter an ID then click 'Fetch by ID'"
                )
                return
            self._spawn_task(self._do_fetch(source, query))
            return

        if not query:
            return

        from pdb_wizard.database import backends_for, search_backends
        status = self.query_one("#db-status", Label)
        codes = backends_for(source, mode)
        status.update(
            f"Searching {', '.join(BACKENDS[c]['label'] for c in codes)} "
            f"({mode})..."
        )
        await asyncio.sleep(0)
        results, msg = await asyncio.to_thread(search_backends, source, mode, query)
        self._results = results
        table = self.query_one("#db-results", DataTable)
        table.clear()
        for r in results:
            label = BACKENDS.get(r.get("source", ""), {}).get("label", r.get("source", ""))
            table.add_row(
                label,
                str(r.get("id") or ""),
                str(r.get("formula") or "")[:25],
                str(r.get("title") or "")[:60],
            )
        status.update(msg)
        self.query_one("#db-preview", Label).update("")

    def _fetch_selected(self) -> None:
        table = self.query_one("#db-results", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self._results):
            self.app.notify("Select a result first", timeout=2)
            return
        result = self._results[table.cursor_row]
        source = result.get("source", "")
        rid = result.get("id", "")
        self._spawn_task(self._do_fetch(source, rid))

    def _fetch_by_id(self) -> None:
        """Direct-ID fetch path (skip search)."""
        source = str(self.query_one("#db-source", Select).value)
        if source == "all":
            self.app.notify("Pick a specific database first (not 'All')", timeout=3)
            return
        rid = self.query_one("#db-query", Input).value.strip()
        if not rid:
            self.app.notify("Enter an ID in the query field", timeout=2)
            return
        self._spawn_task(self._do_fetch(source, rid))

    async def _do_fetch(self, source: str, rid: str) -> None:
        from pdb_wizard.database import BACKENDS

        spec = BACKENDS.get(source)
        if spec is None:
            self.query_one("#db-status", Label).update(f"Unknown source: {source!r}")
            return

        label = spec["label"]
        status = self.query_one("#db-status", Label)
        status.update(f"Fetching {label} {rid}...")
        await asyncio.sleep(0)

        try:
            cache_dir = self.app._db_cache_dir()
            path = await asyncio.to_thread(spec["fetch"], rid, cache_dir)
            # Parsing a large CIF is the synchronous slow part — offload it.
            from pdb_wizard.io import read_file
            load_msgs: list[str] = []
            mol = await asyncio.to_thread(read_file, path, True, load_msgs.append)
            # The download sits in a per-session cache that is deleted on exit. The tab
            # keeps only the file name, so Save and Export default to the working directory.
            self.app.open_in_new_tab(mol, Path(path).name)
            self.app.notify(
                f"Opened {label} {rid} ({len(mol.atoms)} atoms)", timeout=3,
            )
            self.app.notify_overlaps_removed(load_msgs)
            self.app.notify_load_notes(mol)
            self.dismiss(None)  # auto-close on success
        except Exception as e:
            status.update(f"Error: {e}")
            self.app.notify(f"Fetch failed: {e}", timeout=4, severity="error")

    def action_close(self) -> None:
        if self._compute_task is not None and not self._compute_task.done():
            self._compute_task.cancel()
            self._compute_task = None
        self.dismiss(None)


class EnergyPlotScreen(_AnalysisScreen):
    """Energy plot screen for MPMC energy.dat files."""
    _prefix = "energy"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="EnergyPlotScreen") + """
    EnergyPlotScreen Select { width: 20; }
    EnergyPlotScreen #energy-controls { height: auto; margin-bottom: 1; }
    EnergyPlotScreen #energy-status { height: 1; color: $text-muted; }
    """

    def __init__(self, data: dict, filepath: str) -> None:
        # Skip _AnalysisScreen.__init__ — no molecule for energy plots
        ModalScreen.__init__(self)
        self._data = data
        self._filepath = filepath
        self._molecule = None
        self._frames = None
        self._elements = []
        self._compute_task = None

    def _n_steps(self) -> int:
        """Number of rows in the energy data."""
        from pdb_wizard.analyses import energy_n_steps
        return energy_n_steps(self._data)

    def compose(self) -> ComposeResult:
        cols = [k for k in self._data if k != "step"]
        options = [(c, c) for c in cols]
        # Pick a sensible default: N (molecule count) or energy
        default = cols[0] if cols else "N"
        for prefer in ("N", "energy"):
            if prefer in cols:
                default = prefer
                break
        n = self._n_steps()
        with Vertical():
            with Horizontal(id="energy-controls"):
                yield Label("Column:")
                yield Select(options, value=default, id="energy-col",
                             allow_blank=not options)
                if n > 1:
                    yield Label("Start:")
                    yield SpinBox(value=1, min_val=1, max_val=n, id="energy-start")
                    yield Label("End:")
                    yield SpinBox(value=n, min_val=1, max_val=n, id="energy-end")
                yield Button("CSV", id="energy-csv")
                yield Button("Close", id="energy-close")
            yield PlotWidget(id="energy-plot")
            yield Label(f"Loaded {self._filepath}", id="energy-status")

    def on_mount(self) -> None:
        col = str(self.query_one("#energy-col", Select).value)
        self._plot_column(col)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "energy-col" and event.value is not None:
            self._plot_column(str(event.value))

    def on_spin_box_changed(self, event) -> None:
        # Re-plot whenever Start or End changes.
        if event.spin_box.id in ("energy-start", "energy-end"):
            try:
                col = str(self.query_one("#energy-col", Select).value)
            except Exception:
                return
            self._plot_column(col)

    def _step_slice(self) -> slice:
        """Resolve the user's Start/End SpinBoxes into a Python slice.
        Returns the full range if the SpinBoxes don't exist (no data)."""
        from pdb_wizard.analyses import energy_slice
        try:
            start = self.query_one("#energy-start", SpinBox).value
            end = self.query_one("#energy-end", SpinBox).value
        except Exception:
            return energy_slice(self._data)
        return energy_slice(self._data, start, end)

    def _plot_column(self, col: str) -> None:
        from pdb_wizard.analyses import run_energy
        if col not in self._data:
            return
        sl = self._step_slice()
        res = run_energy(self._data, col, sl.start + 1, sl.stop)
        if res.empty:
            self.query_one("#energy-status", Label).update(res.summary)
            return
        plot = self.query_one("#energy-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label,
                      y_min=res.y_min, y_max=res.y_max)
        self.query_one("#energy-status", Label).update(res.summary)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "energy-close":
            self.dismiss(None)
        elif event.button.id == "energy-csv":
            self.app.push_screen(
                FileSaveModal("Save energy as CSV:", default="energy.csv"),
                callback=self._export_csv,
            )

    def _export_csv(self, filepath: str) -> None:
        from pdb_wizard.analyses import write_energy_csv
        if not filepath.strip():
            return
        sl = self._step_slice()
        path = filepath.strip()
        try:
            write_energy_csv(self._data, path, sl)
        except OSError as e:
            self.query_one("#energy-status", Label).update(f"Error: {e}")
            self.app.notify(f"Could not write {path}: {e}", timeout=6, severity="error")
            return
        self.query_one("#energy-status", Label).update(
            f"Exported steps {sl.start + 1}-{sl.stop} to {path}"
        )


class DensityScreen(_AnalysisScreen):
    """Density profile with element/axis selectors."""
    _prefix = "dens"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="DensityScreen") + """
    DensityScreen #dens-controls { height: auto; margin-bottom: 1; }
    DensityScreen #dens-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_pos = np.array([])
        self._last_dens = np.array([])

    def compose(self) -> ComposeResult:
        el_opts = [(el, el) for el in self._elements]
        names = sorted(set(a.name.strip() for a in self._molecule.atoms))
        name_opts = [("(all)", "(all)")] + [(n, n) for n in names]
        axis_opts = [("a (x)", "0"), ("b (y)", "1"), ("c (z)", "2")]
        with Vertical():
            with Horizontal(id="dens-controls"):
                yield Label("Element:")
                yield Select(el_opts, value=self._elements[0] if self._elements else "H",
                             id="dens-el", allow_blank=False)
                yield Label("Name:")
                yield Select(name_opts, value="(all)", id="dens-name", allow_blank=False)
                yield Label("Axis:")
                yield Select(axis_opts, value="2", id="dens-axis", allow_blank=False)
                if self._frames and len(self._frames) > 1:
                    yield Checkbox("Traj", value=True, id="dens-traj")
                yield from self._yield_frame_range()
                yield Button("Compute", id="dens-compute", variant="primary")
                yield Button("CSV", id="dens-csv")
                yield Button("Close", id="dens-close")
            yield PlotWidget(id="dens-plot")
            yield Label("Select atom type, axis, and click Compute", id="dens-status")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "dens-el":
            self._update_name_select(str(event.value), "#dens-name")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dens-compute":
            self._spawn_task(self._run_density())
        elif event.button.id == "dens-close":
            self.action_close()
        elif event.button.id == "dens-csv":
            if len(self._last_pos) == 0:
                return
            self.app.push_screen(FileSaveModal("Save density as CSV:", default="density.csv"), callback=self._export_csv)

    async def _run_density(self) -> None:
        from pdb_wizard.analyses import run_density_profile
        el = str(self.query_one("#dens-el", Select).value)
        name = str(self.query_one("#dens-name", Select).value)
        axis = int(self.query_one("#dens-axis", Select).value)
        use_traj = False
        try:
            use_traj = bool(self.query_one("#dens-traj", Checkbox).value
                            and self._frames and len(self._frames) > 1)
        except Exception:
            pass
        start, end = self._frame_range() if use_traj else (1, None)
        status = self.query_one("#dens-status", Label)
        status.update(f"Computing density profile {el} along {'abc'[axis]}...")
        await asyncio.sleep(0)
        res = await asyncio.to_thread(run_density_profile, self._molecule, el, name, axis,
                                      self._frames, start, end, use_traj)
        if res.empty:
            status.update(res.summary)
            return
        self._last_pos = res.x
        self._last_dens = res.y
        plot = self.query_one("#dens-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "position_A,density", self._last_pos, self._last_dens)


class PxrdScreen(_AnalysisScreen):
    """Simulated Powder XRD pattern."""
    _prefix = "pxrd"
    DEFAULT_CSS = _ANALYSIS_CSS.format(cls="PxrdScreen") + """
    PxrdScreen Select { width: 14; }
    PxrdScreen #pxrd-controls { height: auto; margin-bottom: 1; }
    PxrdScreen #pxrd-row2 { height: auto; margin-bottom: 1; }
    PxrdScreen #pxrd-status { height: 1; color: $text-muted; }
    """

    def __init__(self, molecule: Molecule, frames: list | None = None) -> None:
        super().__init__(molecule, frames)
        self._last_tt = np.array([])
        self._last_intensity = np.array([])

    def compose(self) -> ComposeResult:
        wl_opts = [("Cu Ka 1.5406", "1.5406"), ("Mo Ka 0.7107", "0.7107"), ("Co Ka 1.7903", "1.7903")]
        with Vertical():
            with Horizontal(id="pxrd-controls"):
                yield Label("Source:")
                yield Select(wl_opts, value="1.5406", id="pxrd-wl", allow_blank=False)
                yield Label("2T max:")
                yield SpinBox(value=50, min_val=10, max_val=90, id="pxrd-ttmax")
                yield Label("Width:")
                yield SpinBox(value=10, min_val=1, max_val=50, id="pxrd-width")
                yield Button("Compute", id="pxrd-compute", variant="primary")
                yield Button("Load Exp.", id="pxrd-load-exp")
                yield Button("CSV", id="pxrd-csv")
                yield Button("Close", id="pxrd-close")
            if self._frames and len(self._frames) > 1:
                with Horizontal(id="pxrd-row2"):  # own row: row 1 still fits 120 columns
                    yield Checkbox("Traj", value=True, id="pxrd-traj")
                    yield from self._yield_frame_range()
            yield PlotWidget(id="pxrd-plot")
            yield Label("Select parameters and click Compute  |  Load Exp. to overlay experimental data", id="pxrd-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pxrd-compute":
            self._spawn_task(self._run_pxrd())
        elif event.button.id == "pxrd-close":
            self.action_close()
        elif event.button.id == "pxrd-csv":
            if len(self._last_tt) == 0:
                return
            self.app.push_screen(FileSaveModal("Save PXRD as CSV:", default="pxrd.csv"), callback=self._export_csv)
        elif event.button.id == "pxrd-load-exp":
            self.app.push_screen(
                FileSaveModal("Load experimental PXRD (CSV/XY):", default="", button_label="Open"),
                callback=self._load_experimental,
            )

    async def _run_pxrd(self) -> None:
        from pdb_wizard.analyses import run_pxrd
        wl = float(self.query_one("#pxrd-wl", Select).value)
        tt_max = float(self.query_one("#pxrd-ttmax", SpinBox).value)
        width = self.query_one("#pxrd-width", SpinBox).value / 100.0  # hundredths to degrees
        use_traj = False
        try:
            use_traj = bool(self.query_one("#pxrd-traj", Checkbox).value
                            and self._frames and len(self._frames) > 1)
        except Exception:
            pass
        start, end = self._frame_range() if use_traj else (1, None)
        n_total = len(self._get_frame_slice() or []) if use_traj else 1
        status = self.query_one("#pxrd-status", Label)
        status.update("Computing PXRD pattern...")
        self._rdf_progress = [0.0]

        def _prog(frac):
            self._rdf_progress[0] = frac

        def _update_status() -> None:
            frac = self._rdf_progress[0]
            if use_traj:
                status.update(f"Computing PXRD... frame {int(frac * n_total)}/{n_total}")
            else:
                status.update(f"Computing PXRD... {int(frac * 100)}%")

        poll = self.set_interval(0.1, _update_status)
        await asyncio.sleep(0)
        try:
            res = await asyncio.to_thread(run_pxrd, self._molecule, wl, tt_max, width, _prog,
                                          self._frames, start, end, use_traj)
        finally:
            poll.stop()
        if res.empty:
            status.update(res.summary)
            return
        self._last_tt = res.x
        self._last_intensity = res.y
        plot = self.query_one("#pxrd-plot", PlotWidget)
        plot.set_data(res.x, res.y, title=res.title, x_label=res.x_label, y_min=0.0,
                      y_max=res.y_max)
        status.update(res.summary)

    def _load_experimental(self, filepath: str) -> None:
        from pdb_wizard.analyses import read_experimental_pxrd
        if not filepath.strip():
            return
        try:
            exp_tt, exp_int = read_experimental_pxrd(filepath.strip())
        except Exception as e:
            self.query_one("#pxrd-status", Label).update(f"Error loading: {e}")
            return
        plot = self.query_one("#pxrd-plot", PlotWidget)
        plot.set_overlay(exp_tt, exp_int, color="rgb(255,130,60)")
        self.query_one("#pxrd-status", Label).update(
            f"Experimental data loaded ({len(exp_tt)} points) — "
            "cyan=simulated, orange=experimental"
        )

    def _export_csv(self, filepath: str) -> None:
        self._do_export_csv(filepath, "two_theta_deg,intensity", self._last_tt, self._last_intensity)


class MoleculeView(Widget):
    """Braille-based 3D molecule renderer."""

    can_focus = True

    def __init__(self) -> None:
        super().__init__()
        self.molecule: Molecule | None = None
        self.pbc: PBC | None = None
        self.rot_matrix = rotation_matrix(-0.2, -0.5, 0.0)
        self.camera_distance = 4.0
        self.show_bonds = True
        self.dark_bg = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False
        self.highlighted_atoms: set[int] = set()
        self.show_atom_numbers = False
        self.licorice = False
        self.vdw = False
        self.color_by_charge = False
        self.hide_water = False
        self.ribbon = False
        self.color_mode = "element"
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.ambient = 0.50
        self.diffuse = 0.60
        self.specular = 0.40
        self.shininess = 32.0
        self._cached_strips: list[Strip] = []
        self._cached_size: tuple[int, int] = (0, 0)
        self._dragging = False
        self._drag_moved = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._locked_centroid = None
        self._blink_on = True
        self._blink_timer = None
        self._picked_atoms: list[int] = []  # for measurement mode
        self.measure_mode = False
        self._measurements: list[tuple[list[int], str]] = []  # [(atom_indices, label_text)]
        self.show_polyhedra = False
        self.orthographic = False
        self._density_positions: np.ndarray | None = None
        self._density_values: np.ndarray | None = None
        self._density_element: str = ""
        self._atom_count = 0  # of the structure last given to set_molecule

    class AtomPicked(Message):
        def __init__(self, atom_index: int, shift: bool) -> None:
            super().__init__()
            self.atom_index = atom_index
            self.shift = shift

    def _pick_atom_at(self, term_x: int, term_y: int) -> int | None:
        """Find the atom index closest to terminal cell (term_x, term_y)."""
        if self.molecule is None or len(self.molecule.atoms) == 0:
            return None
        w, h = self.size.width, self.size.height
        if w == 0 or h == 0:
            return None
        px_w = w * 2
        px_h = h * 4
        # Convert terminal cell to pixel center
        click_px = term_x * 2 + 1
        click_py = term_y * 4 + 2
        fov = 1.5
        scale = min(px_w, px_h) / 2
        centroid = self._locked_centroid if self._locked_centroid is not None else self.molecule.center()
        rot = self.rot_matrix
        best_idx = None
        best_dist_sq = float("inf")
        for i, atom in enumerate(self.molecule.atoms):
            pos = rot @ (atom.position - centroid)
            pos[0] += self.pan_x
            pos[1] += self.pan_y
            pos[2] += self.camera_distance
            if pos[2] <= 0.1:
                continue
            sx = px_w / 2 + pos[0] * fov / pos[2] * scale
            sy = px_h / 2 - pos[1] * fov / pos[2] * scale
            d2 = (sx - click_px) ** 2 + (sy - click_py) ** 2
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_idx = i
        # Only pick if within reasonable pixel distance
        if best_dist_sq > (30 * 30):
            return None
        return best_idx

    def on_mouse_down(self, event) -> None:
        self._dragging = True
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_moved = False
        self.capture_mouse()

    def on_mouse_up(self, event) -> None:
        # A release can arrive without a press on this widget: the button went down on
        # another widget or before the app was ready, or the terminal was left reporting
        # the mouse by an earlier program. That is not a click here, so it is ignored.
        if not self._dragging:
            return
        was_drag = self._drag_moved
        self._dragging = False
        self._drag_moved = False
        self.release_mouse()
        if not was_drag and self.measure_mode:
            # Only pick atoms in measure mode
            idx = self._pick_atom_at(event.x, event.y)
            if idx is not None:
                self.post_message(self.AtomPicked(idx, event.shift))

    def on_mouse_move(self, event) -> None:
        if not self._dragging:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        if abs(dx) > 1 or abs(dy) > 1:
            self._drag_moved = True
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        if self.pan_mode:
            self.pan_x += dx * self.camera_distance * 0.01
            self.pan_y -= dy * self.camera_distance * 0.01
            self._clamp_pan()
        else:
            self.rot_matrix = rotation_matrix(-dy * 0.02, -dx * 0.02, 0) @ self.rot_matrix
        self._invalidate_cache()

    def on_mouse_scroll_up(self, event) -> None:
        self.camera_distance = max(1.0, self.camera_distance - 0.5)
        self._invalidate_cache()

    def on_mouse_scroll_down(self, event) -> None:
        self.camera_distance += 0.5
        self._invalidate_cache()

    def set_molecule(self, molecule: Molecule, keep_camera: bool = False) -> None:
        n = len(molecule.atoms)
        if n != self._atom_count:
            # Another frame or an edit changed the atom count: indices picked before no
            # longer name the same atoms (or any atom).
            self._picked_atoms = []
            self.highlighted_atoms = {i for i in self.highlighted_atoms if i < n}
            self._atom_count = n
        self.molecule = molecule
        self.pbc = molecule.pbc
        if not keep_camera:
            mol_radius = molecule.radius()
            self.camera_distance = max(4.0, mol_radius * 3.0)
            self._locked_centroid = None
        self._invalidate_cache()

    def _clamp_pan(self) -> None:
        if self.molecule is None:
            return
        max_pan = self.molecule.radius() * 0.5
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y))

    def _invalidate_cache(self) -> None:
        self._cached_size = (0, 0)
        self.refresh()

    def _start_blink(self) -> None:
        if self._blink_timer is None:
            self._blink_on = True
            self._blink_timer = self.set_interval(0.5, self._toggle_blink)

    def _stop_blink(self) -> None:
        if self._blink_timer is not None:
            self._blink_timer.stop()
            self._blink_timer = None
            self._blink_on = True
            self._invalidate_cache()

    def _toggle_blink(self) -> None:
        if not self.highlighted_atoms:
            self._stop_blink()
            return
        self._blink_on = not self._blink_on
        self._invalidate_cache()

    def render_line(self, y: int) -> Strip:
        w, h = self.size.width, self.size.height
        if (w, h) != self._cached_size:
            self._rebuild(w, h)
        if 0 <= y < len(self._cached_strips):
            return self._cached_strips[y]
        return Strip.blank(w)

    def _rebuild(self, cols: int, rows: int) -> None:
        self._cached_size = (cols, rows)

        if self.molecule is None or cols == 0 or rows == 0:
            self._cached_strips = [Strip.blank(cols) for _ in range(rows)]
            return

        px_w = cols * 2
        px_h = rows * 4

        bg = (0, 0, 0) if self.dark_bg else (255, 255, 255)
        rot = self.rot_matrix

        mol = self.molecule
        if self.hide_water:
            # Detect actual water molecules: O bonded to exactly 2 H (and nothing else)
            from pdb_wizard.geometry import water_atom_indices
            water_atoms = water_atom_indices(mol)
            if water_atoms:
                keep = [i for i in range(len(mol.atoms)) if i not in water_atoms]
                keep_set = set(keep)
                old_to_new = {old: new for new, old in enumerate(keep)}
                new_atoms = [mol.atoms[i] for i in keep]
                new_bonds = [
                    (old_to_new[a], old_to_new[b])
                    for a, b in mol.bonds if a in keep_set and b in keep_set
                ]
                mol = Molecule(atoms=new_atoms, bonds=new_bonds, pbc=mol.pbc)
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[], pbc=mol.pbc)

        hl = self.highlighted_atoms if (self.highlighted_atoms and self._blink_on) else None
        pixels, hit = render_scene(
            px_w, px_h, mol, rot, self.camera_distance,
            bg_color=bg, pbc=self.pbc, ssaa=1,
            pan=(self.pan_x, self.pan_y), highlighted_atoms=hl,
            licorice=self.licorice, vdw=self.vdw,
            ambient=self.ambient, diffuse=self.diffuse,
            specular=self.specular, shininess=self.shininess,
            atom_scale=self.atom_scale, bond_radius=self.bond_radius,
            color_by_charge=self.color_by_charge,
            color_mode=self.color_mode,
            centroid_override=self._locked_centroid,
            ribbon=self.ribbon,
            show_polyhedra=self.show_polyhedra,
            density_positions=self._density_positions,
            density_values=self._density_values,
            orthographic=self.orthographic,
        )

        blocks = pixels.reshape(rows, 4, cols, 2, 3)
        is_on = hit.reshape(rows, 4, cols, 2)

        on_count = is_on.sum(axis=(1, 3))
        on_mask = is_on[:, :, :, :, None]
        color_sum = (blocks * on_mask).sum(axis=(1, 3))
        safe_count = np.maximum(on_count, 1)[:, :, None]
        avg_fg = (color_sum / safe_count).astype(np.uint8)

        braille_bits = np.where(is_on, _BRAILLE_MAP[None, :, None, :], 0)
        codepoints = 0x2800 + braille_bits.sum(axis=(1, 3)).astype(np.uint32)

        any_hit = on_count > 0
        bg_style = Style(bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})")

        label_cells: dict[tuple[int, int], tuple[str, Style]] = {}
        if self.show_atom_numbers and self.molecule is not None:
            fov = 1.5
            scale = min(px_w, px_h) / 2
            centroid = (self._locked_centroid if self._locked_centroid is not None
                        else self.molecule.center())
            label_style = Style(
                color="rgb(255,255,0)" if self.dark_bg else "rgb(0,0,180)",
                bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                bold=True,
            )
            for idx, atom in enumerate(self.molecule.atoms):
                pos = rot @ (atom.position - centroid)
                pos[0] += self.pan_x
                pos[1] += self.pan_y
                pos[2] += self.camera_distance
                if pos[2] <= 0.1:
                    continue
                sx = px_w / 2 + pos[0] * fov / pos[2] * scale
                sy = px_h / 2 - pos[1] * fov / pos[2] * scale
                cell_col = int(sx / 2)
                cell_row = int(sy / 4) - 1
                label = str(idx + 1)
                for ci, ch in enumerate(label):
                    c = cell_col + ci
                    if 0 <= cell_row < rows and 0 <= c < cols:
                        label_cells[(cell_row, c)] = (ch, label_style)

        # Axis labels (a, b, c) in bottom-left corner
        if self.pbc is not None:
            fov = 1.5
            scale = min(px_w, px_h) / 2
            basis = self.pbc.basis_matrix.copy()
            for i in range(3):
                basis[i] = basis[i] / (np.linalg.norm(basis[i]) + 1e-10)
            cam = self.camera_distance
            arrow_len = cam * 0.08
            ox = -cam * 0.85
            oy = -cam * 0.45
            origin = np.array([ox, oy, cam])
            axis_labels = ["a", "b", "c"]
            axis_colors = ["rgb(255,80,80)", "rgb(80,255,80)", "rgb(80,80,255)"]
            for i in range(3):
                direction = rot @ basis[i]
                tip = origin + direction * arrow_len * 1.4
                if tip[2] > 0.1:
                    sx = px_w / 2 + tip[0] * fov / tip[2] * scale
                    sy = px_h / 2 - tip[1] * fov / tip[2] * scale
                    cell_col = int(sx / 2)
                    cell_row = int(sy / 4)
                    style = Style(
                        color=axis_colors[i],
                        bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                        bold=True,
                    )
                    if 0 <= cell_row < rows and 0 <= cell_col < cols:
                        label_cells[(cell_row, cell_col)] = (axis_labels[i], style)

        # Charge colorbar on right side
        if self.color_by_charge and self.molecule is not None:
            charges = [a.charge for a in self.molecule.atoms]
            if charges:
                max_q = max(abs(min(charges)), abs(max(charges)), 0.01)
                bar_height = min(rows - 4, 16)
                bar_col = cols - 8
                bar_start = (rows - bar_height) // 2
                from pdb_wizard.renderer import ImageRenderer
                for bi in range(bar_height):
                    frac = 1.0 - bi / max(bar_height - 1, 1)  # top=+max, bottom=-max
                    charge_val = max_q * (2 * frac - 1)
                    cr, cg, cb = ImageRenderer._charge_color(charge_val)
                    r_row = bar_start + bi
                    if 0 <= r_row < rows:
                        # Color block
                        block_style = Style(bgcolor=f"rgb({cr},{cg},{cb})")
                        label_cells[(r_row, bar_col)] = (" ", block_style)
                        label_cells[(r_row, bar_col + 1)] = (" ", block_style)
                # Labels
                top_label = f"+{max_q:.2f}"
                mid_label = " 0.00"
                bot_label = f"-{max_q:.2f}"
                lbl_style = Style(
                    color="rgb(255,255,255)" if self.dark_bg else "rgb(0,0,0)",
                    bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                )
                for ci, ch in enumerate(top_label):
                    c = bar_col + 2 + ci
                    if 0 <= c < cols:
                        label_cells[(bar_start, c)] = (ch, lbl_style)
                mid_row = bar_start + bar_height // 2
                for ci, ch in enumerate(mid_label):
                    c = bar_col + 2 + ci
                    if 0 <= c < cols:
                        label_cells[(mid_row, c)] = (ch, lbl_style)
                for ci, ch in enumerate(bot_label):
                    c = bar_col + 2 + ci
                    if 0 <= c < cols:
                        label_cells[(bar_start + bar_height - 1, c)] = (ch, lbl_style)

        # Pinned measurement labels
        if self._measurements and self.molecule is not None:
            fov = 1.5
            scale = min(px_w, px_h) / 2
            centroid = self._locked_centroid if self._locked_centroid is not None else self.molecule.center()
            meas_style = Style(
                color="rgb(0,255,0)" if self.dark_bg else "rgb(0,120,0)",
                bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                bold=True,
            )
            for atom_indices, label_text in self._measurements:
                # Compute midpoint of all measured atoms
                valid = [i for i in atom_indices if i < len(self.molecule.atoms)]
                if not valid:
                    continue
                mid = np.mean([self.molecule.atoms[i].position for i in valid], axis=0)
                pos = rot @ (mid - centroid)
                pos[0] += self.pan_x
                pos[1] += self.pan_y
                pos[2] += self.camera_distance
                if pos[2] <= 0.1:
                    continue
                sx = px_w / 2 + pos[0] * fov / pos[2] * scale
                sy = px_h / 2 - pos[1] * fov / pos[2] * scale
                cell_col = int(sx / 2) + 1
                cell_row = int(sy / 4) - 1
                for ci, ch in enumerate(label_text):
                    c = cell_col + ci
                    if 0 <= cell_row < rows and 0 <= c < cols:
                        label_cells[(cell_row, c)] = (ch, meas_style)

        strips = []
        for row in range(rows):
            segments = []
            prev_style = None
            run_chars: list[str] = []
            for x in range(cols):
                if (row, x) in label_cells:
                    ch, style = label_cells[(row, x)]
                elif self.dark_bg:
                    cp = int(codepoints[row, x])
                    if cp == 0x2800:
                        style = bg_style
                        ch = " "
                    else:
                        fg = avg_fg[row, x]
                        style = Style(
                            color=f"rgb({int(fg[0])},{int(fg[1])},{int(fg[2])})",
                            bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                        )
                        ch = chr(cp)
                else:
                    cp = int(codepoints[row, x])
                    if not any_hit[row, x]:
                        style = bg_style
                        ch = " "
                    else:
                        fc = avg_fg[row, x]
                        n_on = int(on_count[row, x])
                        if n_on == 0:
                            style = bg_style
                            ch = " "
                        else:
                            # Blend color with white based on coverage (feathered edges)
                            frac = n_on / 8.0
                            r = int(fc[0] * frac + bg[0] * (1 - frac))
                            g = int(fc[1] * frac + bg[1] * (1 - frac))
                            b = int(fc[2] * frac + bg[2] * (1 - frac))
                            style = Style(bgcolor=f"rgb({r},{g},{b})")
                            ch = " "

                if style == prev_style:
                    run_chars.append(ch)
                else:
                    if run_chars and prev_style is not None:
                        segments.append(Segment("".join(run_chars), prev_style))
                    run_chars = [ch]
                    prev_style = style
            if run_chars and prev_style is not None:
                segments.append(Segment("".join(run_chars), prev_style))
            strips.append(Strip(segments, cols))

        self._cached_strips = strips


def _mpmc_wizard_molecule(app):
    """The molecule the MPMC export wizard was started for. The wizard runs across several
    dialogs and a background step, and the active tab can change in between. Steps called
    without a wizard (or before this was recorded) use the active molecule."""
    mol = getattr(app, "_mpmc_state", {}).get("mol")
    return app.molecule if mol is None else mol


def _task_label(coro) -> str:
    """Human-readable name for a coroutine, e.g. '_do_wrap_async' -> 'wrap'."""
    name = getattr(coro, "__name__", None) or "task"
    for prefix in ("_do_", "_calc_", "_compute_", "_run_", "_load_", "_export_", "_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("_async",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ") or "task"


def _task_failure(task: asyncio.Task) -> BaseException | None:
    """Return the exception a finished task raised, or None (also for cancels)."""
    if task.cancelled():
        return None
    try:
        return task.exception()
    except asyncio.CancelledError:
        return None


def _find_nonoverlap_position(mol: Molecule) -> np.ndarray:
    """Find a position inside the unit cell that maximizes minimum distance to MOF atoms.

    Samples a grid of candidate positions in fractional coordinates, computes
    the minimum distance to any MOF atom for each candidate (using PBC min image),
    and returns the position with the largest minimum distance.
    """
    if mol.pbc is None or len(mol.atoms) == 0:
        return mol.center()

    pbc = mol.pbc
    coords = np.array([a.x for a in mol.atoms])

    # Sample a grid of candidate positions in fractional coordinates
    n = 8  # 8x8x8 = 512 candidates
    fracs = np.linspace(0.1, 0.9, n)
    best_pos = np.array([0.5, 0.5, 0.5]) @ pbc.basis_matrix
    best_min_dist = 0.0

    for fi in fracs:
        for fj in fracs:
            for fk in fracs:
                candidate = np.array([fi, fj, fk]) @ pbc.basis_matrix
                # Compute min-image distances to all atoms
                dx = coords - candidate
                frac_dx = dx @ pbc.reciprocal_basis_matrix
                wrapped = dx - np.round(frac_dx) @ pbc.basis_matrix
                dists = np.linalg.norm(wrapped, axis=1)
                min_dist = dists.min()
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_pos = candidate

    return best_pos


class MenuBar(Widget):
    """Custom menu bar with File/Edit/View dropdowns and integrated tab bar."""

    DEFAULT_CSS = """
    MenuBar {
        dock: top;
        height: 1;
        background: $panel;
        color: $foreground;
        layout: horizontal;
    }
    MenuBar .menu-btn {
        width: auto;
        height: 1;
        padding: 0 2;
        background: $panel;
        text-style: bold;
    }
    MenuBar .menu-btn:hover {
        background: $accent;
    }
    MenuBar .menu-sep {
        width: 1;
        height: 1;
        background: $accent-darken-2;
    }
    MenuBar .tab-label {
        width: auto;
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }
    MenuBar .tab-label.active {
        background: $accent;
        text-style: bold;
    }
    MenuBar .tab-label:hover {
        background: $accent 50%;
    }
    MenuBar .tab-add {
        width: auto;
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }
    MenuBar .tab-add:hover {
        background: $accent;
    }
    MenuBar .menu-spacer {
        width: 1fr;
        height: 1;
    }
    """

    class MenuAction(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    class TabSelected(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class TabClose(Message):
        def __init__(self, index: int, screen_x: int = 0) -> None:
            super().__init__()
            self.index = index
            self.screen_x = screen_x

    class TabAdd(Message):
        pass

    def __init__(self, title: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._tab_names: list[str] = []
        self._active_tab = 0

    def compose(self) -> ComposeResult:
        yield Label(" File ", id="menu-file", classes="menu-btn")
        yield Label(" Edit ", id="menu-edit", classes="menu-btn")
        yield Label(" View ", id="menu-view", classes="menu-btn")
        yield Label(" Analysis ", id="menu-analysis", classes="menu-btn")
        yield Label(" <Undo ", id="menu-undo", classes="menu-btn")
        yield Label(" Redo> ", id="menu-redo", classes="menu-btn")
        yield Label("", classes="menu-sep")
        # Tab labels will be added dynamically via set_tabs()

    def set_title(self, title: str) -> None:
        self._title = title

    def set_tabs(self, names: list[str], active: int) -> None:
        self._tab_names = names
        self._active_tab = active
        self.call_after_refresh(self._rebuild_tabs)

    async def _rebuild_tabs(self) -> None:
        # Remove old tab labels and add button (must await removal)
        to_remove = [
            child for child in self.children
            if child.has_class("tab-label") or child.has_class("tab-add")
        ]
        for child in to_remove:
            await child.remove()
        for i, name in enumerate(self._tab_names):
            cls = "tab-label active" if i == self._active_tab else "tab-label"
            truncated = (name[:19] + "…") if len(name) > 20 else name
            await self.mount(Label(f" {truncated} ", id=f"tab-{i}", classes=cls, markup=False))
        await self.mount(Label(" + ", id="tab-add", classes="tab-add"))

    def on_click(self, event) -> None:
        try:
            widget = self.screen.get_widget_at(event.screen_x, event.screen_y)[0]
            if not hasattr(widget, 'id') or not widget.id:
                return
            # Tab clicks
            if widget.id == "tab-add":
                self.post_message(self.TabAdd())
                return
            if widget.id.startswith("tab-") and widget.has_class("tab-label"):
                idx = int(widget.id.replace("tab-", ""))
                if idx == self._active_tab:
                    # Clicking the active tab opens a tiny context menu
                    self.post_message(self.TabClose(idx, screen_x=widget.region.x))
                else:
                    self.post_message(self.TabSelected(idx))
                return
            # Menu clicks
            if widget.id.startswith("menu-") and widget.id != "menu-title-label":
                menu_name = widget.id.replace("menu-", "")
                self.post_message(self.MenuAction(menu_name))
        except Exception:
            pass


class MenuDropdown(ModalScreen[Optional[str]]):
    """Dropdown menu that appears under a menu button."""

    BINDINGS = [
        Binding("escape", "close", ""),
        Binding("q", "close", ""),
    ]
    DEFAULT_CSS = """
    MenuDropdown {
        align: left top;
    }
    MenuDropdown > Vertical {
        width: 35;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $accent;
        margin: 1 0 0 0;
    }
    MenuDropdown .menu-item {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    MenuDropdown .menu-item:hover {
        background: $accent;
    }
    MenuDropdown .menu-sep {
        width: 100%;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, items: list[tuple[str, str]], offset: int = 0) -> None:
        super().__init__()
        self._items = items
        self._offset = offset
        self._already_dismissed = False

    def compose(self) -> ComposeResult:
        with Vertical():
            for action, label in self._items:
                if action == "---":
                    yield Label(chr(0x2500) * 33, classes="menu-sep")
                else:
                    yield Label(label, id=f"mdrop-{action}", classes="menu-item")

    def on_mount(self) -> None:
        v = self.query_one(Vertical)
        v.styles.margin = (1, 0, 0, self._offset)

    def on_click(self, event) -> None:
        # Guard against double-dismiss when click events bubble after screen close
        if not self.is_attached or self._already_dismissed:
            return
        try:
            widget = self.screen.get_widget_at(event.screen_x, event.screen_y)[0]
            if hasattr(widget, 'id') and widget.id and widget.id.startswith("mdrop-"):
                self._already_dismissed = True
                self.dismiss(widget.id.replace("mdrop-", ""))
                return
        except Exception:
            pass
        self._already_dismissed = True
        self.dismiss("")

    def action_close(self) -> None:
        if self._already_dismissed:
            return
        self._already_dismissed = True
        self.dismiss("")


MENU_FILE = [
    ("open_file", "Open File..."),
    ("---", ""),
    ("write_xyz", "Save as XYZ"),
    ("write_pdb", "Save as PDB"),
    ("write_cif", "Save as CIF"),
    ("write_poscar", "Save as POSCAR"),
    ("write_lammps", "Save as LAMMPS Data"),
    ("write_crystal", "Save as CRYSTAL23 .d12"),
    ("write_mpmc", "Save as MPMC PDB"),
    ("---", ""),
    ("frame_to_tab", "Open Frame in New Tab"),
    ("export_png", "Export PNG"),
    ("export_gif", "Export Rotation GIF"),
    ("---", ""),
    ("quit", "Quit"),
]

MENU_EDIT = [
    ("undo", "Undo"),
    ("---", ""),
    ("wrap_center", "Wrap (centered)"),
    ("wrap_forward", "Wrap (forward)"),
    ("sort", "Sort Atoms"),
    ("extend", "Extend Axis"),
    ("del_lone", "Delete Lone Atoms"),
    ("edit_h", "Edit H Distances"),
    ("update_cell", "Update Unit Cell"),
    ("substitute", "Substitute Element"),
    ("reduce_cell", "Reduce Supercell"),
    ("pack_sorbates", "Pack Sorbates"),
    ("---", ""),
    ("qeq_charges", "Generate QEq Charges"),
]

MENU_VIEW = [
    ("toggle_bonds", "Toggle Bonds (b)"),
    ("toggle_style", "Cycle Style (v)"),
    ("toggle_bg", "Toggle Dark/Light (i)"),
    ("toggle_atom_numbers", "Atom Numbers (#)"),
    ("cycle_color_mode", "Cycle Color Mode (C)"),
    ("toggle_hide_water", "Hide Water (w)"),
    ("toggle_measure", "Measure Mode (m)"),
    ("clear_measurements", "Clear Measurements"),
    ("toggle_polyhedra", "Polyhedra (P)"),
    ("toggle_orthographic", "Orthographic Projection (O)"),
    ("---", ""),
    ("toggle_operations", "Operations Panel (o)"),
    ("toggle_geometry", "Geometry Panel (g)"),
    ("toggle_visual", "Visual Settings (V)"),
    ("---", ""),
    ("reset_view", "Reset View (r)"),
]

MENU_ANALYSIS = [
    ("info", "System Info"),
    ("check", "Check Structure"),
    ("symmetry", "Find Symmetry"),
    ("void_vol", "Void Volume"),
    ("surface_area", "Surface Area"),
    ("pore_size", "Pore Size Distribution"),
    ("rdf", "Radial Distribution"),
    ("coordination", "Coordination Number"),
    ("hbonds", "Hydrogen Bonds"),
    ("msd", "Mean Sq. Displacement"),
    ("rmsd", "RMSD vs Time"),
    ("gyration", "Radius of Gyration"),
    ("density", "Density Profile"),
    ("density3d", "3D Density Map"),
    ("pxrd", "Powder XRD"),
    ("energy_plot", "Energy Plot"),
    ("---", ""),
    ("input_generator", "Input Generator"),
    ("isotherm_plot", "Plot Isotherm Results"),
    ("db_search", "Database Search"),
]


class TabState:
    """Per-tab state — isolates each open structure completely."""
    def __init__(self, molecule: Molecule, filepath: str = "", frames: list | None = None):
        self.molecule = molecule
        self.filepath = filepath
        self.frames = frames
        self.frame_idx = 0
        self.undo_stack: list = []
        self.redo_stack: list = []
        # View state
        self.rot_matrix = rotation_matrix(-0.2, -0.5, 0.0)
        self.camera_distance = 4.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_mode = False
        self.dark_bg = True
        self.show_bonds = True
        self.show_atom_numbers = False
        self.color_by_charge = False
        self.hide_water = False
        self.ribbon = False
        self.show_polyhedra = False
        self.orthographic = False
        self.color_mode = "element"
        self.licorice = False
        self.vdw = False
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.locked_centroid = None
        self.camera_initialized = False

    @property
    def name(self) -> str:
        from pathlib import Path
        return Path(self.filepath).name if self.filepath else "untitled"



class PdbWizardApp(App):
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: vertical; }
    #main-content { height: 1fr; }
    MoleculeView { width: 1fr; }
    #status-bar {
        height: 1;
        max-height: 1;
        overflow: hidden;
    }
    #status-bar Label {
        width: auto;
        max-width: 60;
        margin-right: 1;
        color: $text-muted;
    }
    #status-bar ProgressBar {
        width: 1fr;
        display: none;
    }
    #status-bar.busy ProgressBar {
        display: block;
    }
    #status-bar.busy Label {
        color: $text;
    }
    #traj-bar {
        height: 3;
        display: none;
        background: $accent-darken-3;
        dock: bottom;
    }
    #traj-bar.visible {
        display: block;
    }
    #traj-bar Button {
        width: auto;
        min-width: 3;
        height: 3;
        margin: 0;
    }
    #traj-bar TrackSlider {
        width: 1fr;
        height: 3;
        margin: 0 1;
    }
    #traj-bar Label {
        width: auto;
        height: 3;
        content-align: center middle;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("k,up", "rotate_up", "Rot up", show=False),
        Binding("j,down", "rotate_down", "Rot down", show=False),
        Binding("h,left", "rotate_left", "Rot left", show=False),
        Binding("l,right", "rotate_right", "Rot right", show=False),
        Binding("comma", "rotate_cw", ",/. roll", show=False),
        Binding("full_stop", "rotate_ccw", show=False),
        Binding("K", "zoom_in", show=False),
        Binding("J", "zoom_out", show=False),
        Binding("plus,equals_sign", "zoom_in", "Zoom", key_display="+/-"),
        Binding("minus,underscore", "zoom_out", show=False),
        Binding("t", "toggle_mode", "Pan/Rot"),
        Binding("c", "center", show=False),
        Binding("r", "reset_view", show=False),
        Binding("b", "toggle_bonds", show=False),
        Binding("v", "toggle_style", show=False),
        Binding("i", "toggle_bg", show=False),
        Binding("escape", "close_panel", show=False),
        Binding("o", "toggle_operations", show=False),
        Binding("g", "toggle_geometry", show=False),
        Binding("e", "export_png", show=False),
        Binding("number_sign", "toggle_atom_numbers", show=False),
        Binding("C", "toggle_charge_colors", show=False),
        Binding("w", "toggle_hide_water", show=False),
        Binding("m", "toggle_measure", show=False),
        Binding("P", "toggle_polyhedra", show=False),
        Binding("O", "toggle_orthographic", show=False),
        Binding("right_square_bracket", "next_frame", "Frame", key_display="[/]"),
        Binding("left_square_bracket", "prev_frame", show=False),
        Binding("home", "first_frame", show=False),
        Binding("end", "last_frame", show=False),
        Binding("space", "toggle_play", show=False),
        Binding("n", "panel_next", show=False),
        Binding("p", "panel_prev", show=False),
        Binding("V", "toggle_visual", show=False),
        Binding("q", "request_quit", "Quit"),
    ]

    def __init__(self, molecule: Molecule, filepath: str = "", frames: list | None = None,
                 load_trajectory: bool = False, load_messages: list[str] | None = None):
        super().__init__()
        # Messages produced while reading the startup file (e.g. overlapping
        # atoms deleted by read_file). Shown as a toast once the UI is mounted.
        self._startup_load_messages: list[str] = list(load_messages or [])
        # Tab system
        self._tabs: list[TabState] = [TabState(molecule, filepath, frames)]
        self._active_tab = 0
        # Convenience accessors (always point to active tab)
        self.molecule = molecule
        self.filepath = filepath
        self._frames = frames
        self._frame_idx = 0
        self._load_trajectory = load_trajectory
        self._update_title()
        self._quit_pending = False
        self._quit_timer = None
        self._play_timer = None
        self._play_speed = 1.0
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._play_interval = 0.15
        # Background tasks tracked so we can cancel pending ones on app exit.
        # Without this, "Task was destroyed but it is pending" warnings flood
        # stderr when the user quits while async work is in flight.
        self._bg_tasks: set[asyncio.Task] = set()
        # Per-session cache for downloaded structures. Created lazily on first
        # fetch; removed on app teardown.
        self._db_cache_dir_path: str | None = None
        # The edit being applied in the background (worker thread plus refresh), if any.
        self._edit_task: asyncio.Task | None = None

    def _edit_running(self) -> bool:
        """True, with a toast, while an earlier edit is still being applied. A second edit
        or an undo started now would race it on the same molecule."""
        task = self._edit_task
        if task is None or task.done():
            return False
        self.notify("The previous edit is still running. Try again when it has finished.",
                    timeout=4, severity="warning")
        return True

    def _start_edit(self, coro) -> asyncio.Task:
        self._edit_task = self._track_task(coro)
        return self._edit_task

    @staticmethod
    def _has_real_cell(mol) -> bool:
        """A unit cell from the file or the user, not the huge stand-in box given to
        trajectories that carry no cell."""
        from pdb_wizard.io import is_fallback_cell
        return mol.pbc is not None and not is_fallback_cell(mol.pbc)

    def notify_load_notes(self, mol) -> None:
        """Warn about anything the file reader could not honour (e.g. a CIF read as its
        asymmetric unit); Check Structure repeats it."""
        from pdb_wizard.checks import load_notes
        for note in load_notes(mol):
            self.notify(note, timeout=20, severity="warning")

    def notify_overlaps_removed(self, messages: list[str]) -> None:
        """Toast a summary of atoms deleted by read_file's overlap check.

        The classic menu prints each deletion; in the TUI stdout is not
        visible, so the user must be told here or the atoms vanish silently.
        """
        n = len(messages)
        if n == 0:
            return
        from pdb_wizard.geometry import OVERLAP_CUTOFF
        plural = "s" if n != 1 else ""
        self.notify(
            f"Deleted {n} overlapping atom{plural} (closer than {OVERLAP_CUTOFF} Å)",
            timeout=6, severity="warning",
        )

    def _track_task(self, coro) -> asyncio.Task:
        """Create an asyncio task and track it for cleanup.

        A failure inside the task is surfaced as an error toast and the busy
        progress bar is cleared. Otherwise the exception would only be printed
        to stderr at garbage-collection time ("Task exception was never
        retrieved") and the status bar would stay stuck on "...".
        """
        label = _task_label(coro)
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(lambda t, label=label: self._on_tracked_task_done(t, label))
        return task

    def _on_tracked_task_done(self, task: asyncio.Task, label: str) -> None:
        exc = _task_failure(task)
        if exc is None:
            return
        try:
            self._hide_progress()
        except Exception:
            pass
        try:
            self.notify(f"{label} failed: {exc}", timeout=6, severity="error")
        except Exception:
            pass

    def _db_cache_dir(self) -> str:
        """Return a single per-session directory for downloaded structures.
        Created on first call, cleaned up in on_unmount."""
        if self._db_cache_dir_path is None:
            import tempfile
            self._db_cache_dir_path = tempfile.mkdtemp(prefix="pdb_wizard_db_")
        return self._db_cache_dir_path

    async def on_unmount(self) -> None:
        """Cancel any in-flight background tasks and clean caches."""
        for task in list(self._bg_tasks):
            if not task.done():
                task.cancel()
        self._bg_tasks.clear()
        if self._db_cache_dir_path is not None:
            import shutil
            shutil.rmtree(self._db_cache_dir_path, ignore_errors=True)
            self._db_cache_dir_path = None

    def _refresh_tab_bar(self) -> None:
        names = self._disambiguated_tab_names()
        try:
            self.query_one("#app-menu", MenuBar).set_tabs(names, self._active_tab)
        except Exception:
            pass

    def _disambiguated_tab_names(self) -> list[str]:
        """Return tab names with parent directory prefixed for duplicates."""
        from collections import Counter
        base_names = [t.name for t in self._tabs]
        counts = Counter(base_names)
        result = []
        for tab in self._tabs:
            base = tab.name
            if counts[base] > 1 and tab.filepath:
                parent = Path(tab.filepath).parent.name
                if parent:
                    result.append(f"{parent}/{base}")
                else:
                    result.append(base)
            else:
                result.append(base)
        return result

    def _save_current_tab_state(self) -> None:
        """Save current view state back to the active tab."""
        tab = self._tabs[self._active_tab]
        view = self.query_one(MoleculeView)
        tab.molecule = self.molecule
        tab.frames = self._frames
        tab.frame_idx = self._frame_idx
        tab.undo_stack = self._undo_stack
        tab.redo_stack = self._redo_stack
        tab.rot_matrix = view.rot_matrix
        tab.camera_distance = view.camera_distance
        tab.pan_x = view.pan_x
        tab.pan_y = view.pan_y
        tab.pan_mode = view.pan_mode
        tab.dark_bg = view.dark_bg
        tab.show_bonds = view.show_bonds
        tab.show_atom_numbers = view.show_atom_numbers
        tab.color_by_charge = view.color_by_charge
        tab.hide_water = view.hide_water
        tab.ribbon = view.ribbon
        tab.show_polyhedra = view.show_polyhedra
        tab.orthographic = view.orthographic
        tab.color_mode = view.color_mode
        tab.licorice = view.licorice
        tab.vdw = view.vdw
        tab.atom_scale = view.atom_scale
        tab.bond_radius = view.bond_radius
        tab.locked_centroid = view._locked_centroid

    def _restore_tab_state(self, idx: int) -> None:
        """Restore state from tab idx to the view."""
        tab = self._tabs[idx]
        self._active_tab = idx
        self.molecule = tab.molecule
        self.filepath = tab.filepath
        self._frames = tab.frames
        self._frame_idx = tab.frame_idx
        self._undo_stack = tab.undo_stack
        self._redo_stack = tab.redo_stack

        view = self.query_one(MoleculeView)
        view.rot_matrix = tab.rot_matrix
        view.camera_distance = tab.camera_distance
        view.pan_x = tab.pan_x
        view.pan_y = tab.pan_y
        view.pan_mode = tab.pan_mode
        view.dark_bg = tab.dark_bg
        view.show_bonds = tab.show_bonds
        view.show_atom_numbers = tab.show_atom_numbers
        view.color_by_charge = tab.color_by_charge
        view.hide_water = tab.hide_water
        view.ribbon = tab.ribbon
        view.show_polyhedra = tab.show_polyhedra
        view.orthographic = getattr(tab, "orthographic", False)
        view.color_mode = tab.color_mode
        view.licorice = tab.licorice
        view.vdw = tab.vdw
        view.atom_scale = tab.atom_scale
        view.bond_radius = tab.bond_radius
        view._locked_centroid = tab.locked_centroid

        if len(self.molecule.atoms) > 0:
            if tab.camera_initialized:
                view.set_molecule(self.molecule, keep_camera=True)
            else:
                view.set_molecule(self.molecule, keep_camera=False)
                tab.camera_initialized = True
            self.query_one(GeometryPanel).set_molecule(self.molecule)
        else:
            # An empty tab (a trajectory still loading): do not leave the previous tab's
            # structure on screen.
            view.molecule = None
            view.pbc = None
            view.highlighted_atoms = set()
            view._picked_atoms = []
            view._atom_count = 0
            panel = self.query_one(GeometryPanel)
            panel._molecule = None
            panel._populated_tabs = set()
            for dt in panel.query(DataTable):
                dt.clear()
        self._update_title()
        self._refresh_tab_bar()
        self._setup_traj_bar()
        view._invalidate_cache()

    def open_in_new_tab(self, mol: Molecule, filepath: str = "",
                        frames: list | None = None) -> int:
        """Append a new tab and switch to it. Returns the new tab's index.

        Centralizes the save → append → restore sequence so callers (modal
        screens, file dialogs, DB fetch) don't have to reach into the
        internals of self._tabs and self._active_tab. Stops playback first
        so a running animation on the old tab doesn't bleed into the new one.
        """
        if self._play_timer:
            self._play_timer.stop()
            self._play_timer = None
        self._save_current_tab_state()
        self._tabs.append(TabState(mol, filepath, frames))
        new_idx = len(self._tabs) - 1
        self._restore_tab_state(new_idx)
        return new_idx

    def on_menu_bar_tab_selected(self, event: MenuBar.TabSelected) -> None:
        if event.index == self._active_tab:
            return
        if event.index < 0 or event.index >= len(self._tabs):
            return
        # Stop playback
        if self._play_timer:
            self._play_timer.stop()
            self._play_timer = None
        self._save_current_tab_state()
        self._restore_tab_state(event.index)

    def on_menu_bar_tab_close(self, event: MenuBar.TabClose) -> None:
        if len(self._tabs) <= 1:
            self.notify("Can't close the last tab", timeout=2)
            return
        idx = event.index
        disambig = self._disambiguated_tab_names()
        name = disambig[idx] if idx < len(disambig) else self._tabs[idx].name
        items = [("close_tab", f"Close {name}")]
        self.push_screen(
            MenuDropdown(items, offset=event.screen_x),
            callback=lambda action: self._confirm_close_tab(action, idx),
        )

    def _confirm_close_tab(self, action: str, idx: int) -> None:
        if action != "close_tab":
            return
        if idx < 0 or idx >= len(self._tabs) or len(self._tabs) <= 1:
            return
        # Stop playback
        if self._play_timer:
            self._play_timer.stop()
            self._play_timer = None
        closed_name = self._tabs[idx].name
        self._tabs.pop(idx)
        if self._active_tab >= len(self._tabs):
            self._active_tab = len(self._tabs) - 1
        elif self._active_tab > idx:
            self._active_tab -= 1
        self._restore_tab_state(self._active_tab)
        self.notify(f"Closed {closed_name}", timeout=2)

    def on_menu_bar_tab_add(self, event: MenuBar.TabAdd) -> None:
        self.push_screen(
            FileSaveModal("Open file:", default="", button_label="Open"),
            callback=self._open_new_tab,
        )

    @staticmethod
    def _is_trajectory_file(filepath: str) -> bool:
        """True for multi-frame PDB/XYZ files and readable DCD files."""
        from pdb_wizard.trajectory import is_trajectory_file
        return is_trajectory_file(filepath)

    def _open_new_tab(self, filepath: str) -> None:
        if not filepath.strip():
            return
        filepath = filepath.strip()
        try:
            from pdb_wizard.io import read_file
            is_traj = self._is_trajectory_file(filepath)

            if is_traj:
                # Load trajectory in background into a new tab
                # Don't switch yet — stay on current tab until loading finishes
                self._save_current_tab_state()
                empty = Molecule(atoms=[])
                new_tab = TabState(empty, filepath)
                self._tabs.append(new_tab)
                self._refresh_tab_bar()
                self.notify(f"Loading {new_tab.name}...", timeout=2)
                self._track_task(self._load_traj_into_tab(filepath, new_tab))
            else:
                load_msgs: list[str] = []
                mol = read_file(filepath, on_message=load_msgs.append)
                self.open_in_new_tab(mol, filepath)
                self.notify(f"Opened {self._tabs[-1].name}", timeout=2)
                self.notify_overlaps_removed(load_msgs)
                self.notify_load_notes(mol)
        except Exception as e:
            self.notify(f"Error: {e}", timeout=3)

    async def _load_traj_into_tab(self, filepath: str, tab) -> None:
        """Load a trajectory file into a specific tab in the background.

        ``tab`` is the TabState itself: tabs may be closed or opened while the file loads,
        so a position in the tab bar would go stale. (An index is still accepted.)
        """
        if isinstance(tab, int):
            tab = self._tabs[tab] if 0 <= tab < len(self._tabs) else None
        self._show_progress("Loading trajectory...", total=100)
        await asyncio.sleep(0)

        _progress = [0.0]

        def _prog(frac: float) -> None:
            _progress[0] = frac

        def _poll_progress() -> None:
            pct = int(_progress[0] * 80)
            self._update_progress(pct)
            self.query_one("#status-label", Label).update(
                f"Loading trajectory... {int(_progress[0] * 100)}%"
            )

        timer = self.set_interval(0.1, _poll_progress)

        def _read():
            from pdb_wizard.trajectory import read_trajectory
            return read_trajectory(filepath, _prog)

        try:
            frames = await asyncio.to_thread(_read)
        finally:
            timer.stop()

        self._warn_trajectory_caveats(frames)

        self.query_one("#status-label", Label).update("Detecting bonds...")
        self._update_progress(85)
        await asyncio.sleep(0)

        if frames:
            await asyncio.to_thread(frames[0].detect_bonds)

        self._hide_progress()

        tab_idx = next((i for i, t in enumerate(self._tabs) if t is tab), None)
        if tab_idx is None:
            return  # the tab was closed while its file was loading

        # Update the tab state
        tab.frames = frames
        tab.molecule = frames[0] if frames else Molecule(atoms=[])

        # Switch to the newly loaded tab. When loading into the already-active
        # (empty) tab, saving the current state first would clobber the frames
        # we just stored with the view's stale None.
        if tab_idx == self._active_tab:
            self._restore_tab_state(tab_idx)
        elif len(self.screen_stack) > 1:
            # A dialog is open for the structure on screen: its answer must not land on
            # another tab, so the new one waits for a click.
            self.notify(f"Loaded {len(frames)} frames into the {tab.name} tab", timeout=4)
            return
        else:
            self._save_current_tab_state()
            self._restore_tab_state(tab_idx)
        self.notify(f"Loaded {len(frames)} frames", timeout=2)

    def _update_title(self) -> None:
        # Use the disambiguated tab name (includes parent dir if name duplicated)
        try:
            disambig = self._disambiguated_tab_names()
            name = disambig[self._active_tab] if 0 <= self._active_tab < len(disambig) else "PDB Wizard"
        except Exception:
            name = Path(self.filepath).name if self.filepath else "PDB Wizard"
        if self._frames and len(self._frames) > 1:
            title = f"{name}  frame {self._frame_idx + 1}/{len(self._frames)}"
        else:
            title = name
        self.title = title
        try:
            self.query_one("#app-menu", MenuBar).set_title(title)
        except Exception:
            pass

    def action_request_quit(self) -> None:
        if self._quit_pending:
            self.exit()
        else:
            self._quit_pending = True
            self.notify("Press q again to quit", timeout=3)
            self._quit_timer = self.set_timer(3.0, self._cancel_quit)

    def _cancel_quit(self) -> None:
        self._quit_pending = False

    def compose(self) -> ComposeResult:
        yield MenuBar(title=self.title, id="app-menu")
        with Horizontal(id="main-content"):
            yield OperationsPanel()
            yield MoleculeView()
            yield GeometryPanel()
            yield VisualPanel()
        with Horizontal(id="status-bar"):
            yield Label("", id="status-label")
            yield ProgressBar(id="status-progress", total=100, show_eta=False)
        with Horizontal(id="traj-bar"):
            yield Button("|<", id="traj-first")
            yield Button("<", id="traj-prev")
            yield Button("Play", id="traj-play")
            yield Button(">", id="traj-next")
            yield Button(">|", id="traj-last")
            yield Button("Slow", id="traj-slower")
            yield Label("1x", id="traj-speed")
            yield Button("Fast", id="traj-faster")
            yield TrackSlider(id="traj-slider")
            yield Label("1/1", id="traj-label")
        yield Footer()

    def on_mount(self) -> None:
        # Open operations panel by default
        ops = self.query_one(OperationsPanel)
        ops.add_class("visible")
        for dt in ops.query(DataTable):
            dt.focus()
            break

        # Initialize tab bar
        self._refresh_tab_bar()

        if self._load_trajectory:
            # Load trajectory in background
            self._track_task(self._load_trajectory_async())
        elif len(self.molecule.atoms) > 0:
            self._show_progress("Loading molecule...")
            view = self.query_one(MoleculeView)
            view.set_molecule(self.molecule)
            self._tabs[self._active_tab].camera_initialized = True
            self._show_progress("Building geometry tables...")
            panel = self.query_one(GeometryPanel)
            panel.set_molecule(self.molecule)
            self._hide_progress()
            self._setup_traj_bar()

        if self._startup_load_messages:
            self.notify_overlaps_removed(self._startup_load_messages)
            self._startup_load_messages = []
        self.notify_load_notes(self.molecule)

    async def _load_trajectory_async(self) -> None:
        tab = self._tabs[self._active_tab]  # the tab this file belongs to
        self._show_progress("Loading trajectory...", total=100)
        await asyncio.sleep(0)


        def _prog(frac: float) -> None:
            self.call_from_thread(self._update_progress, int(frac * 80))
            self.call_from_thread(
                self.query_one("#status-label", Label).update,
                f"Loading trajectory... {int(frac * 100)}%"
            )

        def _read():
            from pdb_wizard.trajectory import read_trajectory
            return read_trajectory(self.filepath, _prog)

        frames = await asyncio.to_thread(_read)
        self._warn_trajectory_caveats(frames)

        self.query_one("#status-label", Label).update("Detecting bonds...")
        self._update_progress(85)
        await asyncio.sleep(0)

        # Detect bonds on first frame only (for display), rest on demand
        if frames:
            await asyncio.to_thread(frames[0].detect_bonds)

        self._update_progress(80)

        if tab is not self._tabs[self._active_tab]:
            # Another file was opened in a new tab while this one loaded: the frames belong
            # to the startup tab, not to whatever is on screen now.
            self._hide_progress()
            if any(t is tab for t in self._tabs):
                tab.frames = frames
                tab.frame_idx = 0
                if frames:
                    tab.molecule = frames[0]
                self.notify(f"Loaded {len(frames)} frames into the {tab.name} tab", timeout=4)
            return

        self._frames = frames
        self._frame_idx = 0
        if frames:
            self.molecule = frames[0]

        view = self.query_one(MoleculeView)
        if self.molecule and len(self.molecule.atoms) > 0:
            view.set_molecule(self.molecule)
            self.query_one(GeometryPanel).set_molecule(self.molecule)

        self._update_progress(90)
        self.query_one("#status-label", Label).update("Ready")
        await asyncio.sleep(0)
        self._hide_progress()
        self._update_title()
        self._setup_traj_bar()

    def _warn_trajectory_caveats(self, frames: list | None) -> None:
        """Toast anything the trajectory reader had to assume or fake."""
        from pdb_wizard.trajectory import trajectory_caveats
        for note in trajectory_caveats(frames):
            self.notify(note, severity="warning", timeout=8)

    def _setup_traj_bar(self) -> None:
        traj_bar = self.query_one("#traj-bar")
        if self._frames and len(self._frames) > 1:
            traj_bar.add_class("visible")
            n = len(self._frames)
            idx = self._frame_idx + 1
            self.query_one("#traj-slider", TrackSlider).set_position(idx, n)
            self.query_one("#traj-label", Label).update(f"{idx}/{n}")
        else:
            traj_bar.remove_class("visible")

    def _show_progress(self, label: str, total: int = 100) -> None:
        bar = self.query_one("#status-bar")
        bar.add_class("busy")
        self.query_one("#status-label", Label).update(label)
        pb = self.query_one("#status-progress", ProgressBar)
        pb.update(total=total, progress=0)

    def _update_progress(self, progress: int) -> None:
        self.query_one("#status-progress", ProgressBar).update(progress=progress)

    def _hide_progress(self) -> None:
        bar = self.query_one("#status-bar")
        bar.remove_class("busy")
        # Clear the label so the always-visible bar looks empty when idle.
        self.query_one("#status-label", Label).update("")

    def _run_with_progress(self, label: str, func, *args, **kwargs):
        """Run a function async with a progress bar shown."""
        self._show_progress(label)
        self._track_task(self._run_bg_task(func, *args, **kwargs))

    async def _run_bg_task(self, func, *args, **kwargs):
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
        finally:
            self._hide_progress()
        return result

    async def _populate_tables_async(self, panel: GeometryPanel, mol: Molecule) -> None:
        """Populate geometry tables one at a time, yielding between each for UI updates."""
        panel._populating = True

        # Use cached data
        bonds = getattr(mol, '_cached_bond_lengths', None) or mol.get_bond_lengths()
        self.query_one("#status-label", Label).update("Populating bonds table...")
        self._update_progress(86)
        await asyncio.sleep(0)
        table = panel.query_one("#bonds-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Length (\u00c5)")
        for i, j, dist in bonds:
            table.add_row(panel._atom_label(i), panel._atom_label(j), f"{dist:.4f}", key=f"{i}-{j}")

        angles = getattr(mol, '_cached_angles', None) or mol.get_angles()
        self.query_one("#status-label", Label).update("Populating angles table...")
        self._update_progress(88)
        await asyncio.sleep(0)
        table = panel.query_one("#angles-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Vertex", "Atom 3", "Angle (\u00b0)")
        for i, j, k, angle in angles:
            table.add_row(
                panel._atom_label(i), panel._atom_label(j), panel._atom_label(k),
                f"{angle:.3f}", key=f"{i}-{j}-{k}",
            )

        dihedrals = getattr(mol, '_cached_dihedrals', None) or mol.get_dihedrals()
        self.query_one("#status-label", Label).update("Populating dihedrals table...")
        self._update_progress(90)
        await asyncio.sleep(0)
        table = panel.query_one("#dihedrals-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle (\u00b0)")
        for i, j, k, l_idx, angle in dihedrals:
            table.add_row(
                panel._atom_label(i), panel._atom_label(j),
                panel._atom_label(k), panel._atom_label(l_idx),
                f"{angle:.3f}", key=f"{i}-{j}-{k}-{l_idx}",
            )

        contacts = getattr(mol, '_cached_contacts', None) or []
        self.query_one("#status-label", Label).update("Populating contacts table...")
        self._update_progress(92)
        await asyncio.sleep(0)
        table = panel.query_one("#contacts-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Distance (\u00c5)")
        for msg in contacts:
            parts = msg.split()
            el_pair = parts[0]
            id1, id2 = int(parts[1]), int(parts[2])
            r_val = parts[-1]
            table.add_row(
                f"{id1}:{el_pair.split('-')[0]}", f"{id2}:{el_pair.split('-')[1]}",
                r_val, key=f"{id1 - 1}-{id2 - 1}",
            )

        submols = getattr(mol, '_cached_molecules', None) or mol.find_molecules()
        self.query_one("#status-label", Label).update("Populating molecules table...")
        self._update_progress(94)
        await asyncio.sleep(0)
        table = panel.query_one("#molecules-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Atoms", "Formula")
        for mi, sm in enumerate(submols):
            sm.atoms.sort(key=lambda a: a.atomic_number, reverse=True)
            from collections import Counter
            counts = Counter(a.element.symbol for a in sm.atoms)
            formula = "".join(f"{el}{n}" if n > 1 else el for el, n in counts.items())
            atom_indices = "-".join(str(mol.atoms.index(a)) for a in sm.atoms)
            table.add_row(str(mi + 1), str(len(sm.atoms)), formula, key=atom_indices)

        lone = getattr(mol, '_cached_lone_atoms', None) or []
        self.query_one("#status-label", Label).update("Populating coords table...")
        self._update_progress(96)
        await asyncio.sleep(0)
        table = panel.query_one("#lone-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Element", "x", "y", "z")
        for a in lone:
            idx = mol.atoms.index(a)
            table.add_row(
                str(a.id), a.element.symbol,
                f"{a.x[0]:.4f}", f"{a.x[1]:.4f}", f"{a.x[2]:.4f}",
                key=str(idx),
            )

        self._update_progress(98)
        await asyncio.sleep(0)
        table = panel.query_one("#coords-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "El", "x", "y", "z")
        for idx, a in enumerate(mol.atoms):
            table.add_row(
                str(a.id), a.element.symbol,
                f"{a.x[0]:.4f}", f"{a.x[1]:.4f}", f"{a.x[2]:.4f}",
                key=str(idx),
            )

        # Clear caches
        for attr in ('_cached_bond_lengths', '_cached_angles', '_cached_dihedrals',
                     '_cached_contacts', '_cached_molecules', '_cached_lone_atoms'):
            if hasattr(mol, attr):
                delattr(mol, attr)

        panel._populating = False

    @staticmethod
    def _precompute_geometry(mol: Molecule, progress=None) -> None:
        """Pre-compute geometry data in a thread so the panel populate is fast."""
        from pdb_wizard.geometry import get_close_contacts, get_lone_atoms
        from pdb_wizard.molecule import set_atom_ids
        set_atom_ids(mol.atoms)

        if progress:
            progress(0, "Computing bond lengths...")
        mol._cached_bond_lengths = mol.get_bond_lengths()

        if progress:
            progress(5, "Computing angles...")
        mol._cached_angles = mol.get_angles()

        if progress:
            progress(15, "Computing dihedrals...")
        mol._cached_dihedrals = mol.get_dihedrals()

        if progress:
            progress(30, "Finding close contacts...")
        mol._cached_contacts = get_close_contacts(mol)

        if progress:
            progress(55, "Finding molecules...")
        mol._cached_molecules = mol.find_molecules()

        if progress:
            progress(75, "Finding lone atoms...")
        mol._cached_lone_atoms = get_lone_atoms(mol)

        if progress:
            progress(100, "Done computing geometry")

    def _panel_is_open(self) -> bool:
        return (
            self.query_one(OperationsPanel).has_class("visible")
            or self.query_one(GeometryPanel).has_class("visible")
            or self.query_one(VisualPanel).has_class("visible")
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("panel_next", "panel_prev"):
            if not self._panel_is_open():
                return False
        if action in ("next_frame", "prev_frame", "first_frame", "last_frame"):
            if not self._frames or len(self._frames) <= 1:
                return False
        return True

    # --- Rotation / Zoom / Pan ---

    def action_rotate_up(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(-0.1, 0, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_down(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_y += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0.1, 0, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_left(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x += view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0, 0.1, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_right(self) -> None:
        view = self.query_one(MoleculeView)
        if view.pan_mode:
            view.pan_x -= view.camera_distance * 0.05
            view._clamp_pan()
        else:
            view.rot_matrix = rotation_matrix(0, -0.1, 0) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_cw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(0, 0, 0.1) @ view.rot_matrix
        view._invalidate_cache()

    def action_rotate_ccw(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(0, 0, -0.1) @ view.rot_matrix
        view._invalidate_cache()

    def action_zoom_in(self) -> None:
        view = self.query_one(MoleculeView)
        view.camera_distance = max(1.0, view.camera_distance - 0.5)
        view._invalidate_cache()

    def action_zoom_out(self) -> None:
        view = self.query_one(MoleculeView)
        view.camera_distance += 0.5
        view._invalidate_cache()

    def action_toggle_mode(self) -> None:
        view = self.query_one(MoleculeView)
        view.pan_mode = not view.pan_mode
        mode = "PAN" if view.pan_mode else "ROT"
        self.notify(f"Mode: {mode}", timeout=1)

    def action_center(self) -> None:
        view = self.query_one(MoleculeView)
        view.pan_x = 0.0
        view.pan_y = 0.0
        view._invalidate_cache()

    # --- Display toggles ---

    def action_toggle_style(self) -> None:
        view = self.query_one(MoleculeView)
        # Cycle: CPK -> Licorice -> VDW -> Ribbon -> CPK
        if not view.licorice and not view.vdw and not view.ribbon:
            view.licorice = True
            view.vdw = False
            view.ribbon = False
            view.bond_radius = 0.15
        elif view.licorice:
            view.licorice = False
            view.vdw = True
            view.ribbon = False
            view.bond_radius = 0.08
        elif view.vdw:
            view.licorice = False
            view.vdw = False
            view.ribbon = True
            view.bond_radius = 0.08
        else:
            view.licorice = False
            view.vdw = False
            view.ribbon = False
            view.bond_radius = 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice, vdw=view.vdw, ribbon=view.ribbon,
                ambient=view.ambient, diffuse=view.diffuse,
                specular=view.specular, shininess=view.shininess,
                atom_scale=view.atom_scale, bond_radius=view.bond_radius,
            )
        view._invalidate_cache()

    def action_toggle_atom_numbers(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_atom_numbers = not view.show_atom_numbers
        view._invalidate_cache()

    def action_toggle_charge_colors(self) -> None:
        self._cycle_color_mode()

    def _cycle_color_mode(self) -> None:
        view = self.query_one(MoleculeView)
        modes = ["element", "charge", "residue", "chain", "index"]
        try:
            idx = modes.index(view.color_mode)
        except ValueError:
            idx = 0
        view.color_mode = modes[(idx + 1) % len(modes)]
        view.color_by_charge = (view.color_mode == "charge")
        self.notify(f"Color: {view.color_mode}", timeout=1)
        view._invalidate_cache()

    def action_toggle_hide_water(self) -> None:
        view = self.query_one(MoleculeView)
        view.hide_water = not view.hide_water
        label = "hidden" if view.hide_water else "visible"
        self.notify(f"Water: {label}", timeout=1)
        view._invalidate_cache()

    def action_toggle_measure(self) -> None:
        view = self.query_one(MoleculeView)
        view.measure_mode = not view.measure_mode
        if view.measure_mode:
            view._picked_atoms = []
            self.notify("Measure mode ON — click atoms to measure distances/angles", timeout=3)
        else:
            view._picked_atoms = []
            view.highlighted_atoms = set()
            view._stop_blink()
            self.notify("Measure mode OFF", timeout=1)
        view._invalidate_cache()

    def action_toggle_polyhedra(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_polyhedra = not view.show_polyhedra
        label = "ON" if view.show_polyhedra else "OFF"
        self.notify(f"Polyhedra: {label}", timeout=1)
        view._invalidate_cache()

    def action_toggle_orthographic(self) -> None:
        view = self.query_one(MoleculeView)
        view.orthographic = not view.orthographic
        label = "Orthographic" if view.orthographic else "Perspective"
        self.notify(f"Projection: {label}", timeout=1)
        view._invalidate_cache()

    def _clear_measurements(self) -> None:
        view = self.query_one(MoleculeView)
        view._measurements.clear()
        self.notify("Measurements cleared", timeout=1)
        view._invalidate_cache()

    def _switch_frame(self, idx: int) -> None:
        if not self._frames or len(self._frames) <= 1:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        if idx == self._frame_idx:
            return
        self._frame_idx = idx
        self.molecule = self._frames[idx]
        # Reuse bonds from frame 0 if this frame has none (same topology)
        if not self.molecule.bonds and len(self.molecule.atoms) > 0:
            if self._frames[0].bonds and len(self._frames[0].atoms) == len(self.molecule.atoms):
                self.molecule.bonds = self._frames[0].bonds
            else:
                self.molecule.detect_bonds()
        view = self.query_one(MoleculeView)
        # Lock centroid to first frame so MOF doesn't jitter
        if view._locked_centroid is None:
            view._locked_centroid = self._frames[0].center()
        view.set_molecule(self.molecule, keep_camera=True)
        self._update_title()
        n = len(self._frames)
        self.query_one("#traj-slider", TrackSlider).set_position(idx + 1, n)
        self.query_one("#traj-label", Label).update(f"{idx + 1}/{n}")
        # Update geometry panel only when not auto-playing (expensive)
        if self._play_timer is None:
            self.query_one(GeometryPanel).set_molecule(self.molecule)
        view._invalidate_cache()

    def on_track_slider_seeked(self, event: TrackSlider.Seeked) -> None:
        if self._frames:
            idx = int(event.fraction * (len(self._frames) - 1))
            self._switch_frame(idx)

    def _on_traj_button(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "traj-first":
            self._switch_frame(0)
        elif bid == "traj-prev":
            self._switch_frame(self._frame_idx - 1)
        elif bid == "traj-next":
            self._switch_frame(self._frame_idx + 1)
        elif bid == "traj-last":
            if self._frames:
                self._switch_frame(len(self._frames) - 1)
        elif bid == "traj-play":
            self._toggle_play()
        elif bid == "traj-slower":
            self._play_speed = max(0.125, self._play_speed / 2)
            self._update_play_speed()
        elif bid == "traj-faster":
            self._play_speed = min(16.0, self._play_speed * 2)
            self._update_play_speed()

    def _update_play_speed(self) -> None:
        self.query_one("#traj-speed", Label).update(f"{self._play_speed:g}x")
        # Restart timer at new speed if playing
        if self._play_timer is not None:
            self._play_timer.stop()
            interval = self._play_interval / self._play_speed
            self._play_timer = self.set_interval(interval, self._play_step)

    def _toggle_play(self) -> None:
        if not self._frames:
            return
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
            self.query_one("#traj-play", Button).label = "Play"
            # Update geometry panel now that playback stopped
            self.query_one(GeometryPanel).set_molecule(self.molecule)
        else:
            self.query_one("#traj-play", Button).label = "Pause"
            interval = self._play_interval / self._play_speed
            self._play_timer = self.set_interval(interval, self._play_step)

    def _play_step(self) -> None:
        if not self._frames:
            return
        # Skip frames if rendering is slow — advance by play_speed to keep up
        step = max(1, int(self._play_speed))
        next_idx = self._frame_idx + step
        if next_idx >= len(self._frames):
            next_idx = 0  # loop
        self._switch_frame(next_idx)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("traj-"):
            self._on_traj_button(event)

    def action_next_frame(self) -> None:
        self._switch_frame(self._frame_idx + 1)

    def action_prev_frame(self) -> None:
        self._switch_frame(self._frame_idx - 1)

    def action_first_frame(self) -> None:
        self._switch_frame(0)

    def action_last_frame(self) -> None:
        if self._frames:
            self._switch_frame(len(self._frames) - 1)

    def action_toggle_play(self) -> None:
        if self._frames and len(self._frames) > 1:
            self._toggle_play()

    def action_toggle_bonds(self) -> None:
        view = self.query_one(MoleculeView)
        view.show_bonds = not view.show_bonds
        view._invalidate_cache()

    def action_toggle_bg(self) -> None:
        view = self.query_one(MoleculeView)
        view.dark_bg = not view.dark_bg
        self.theme = "textual-dark" if view.dark_bg else "textual-light"
        view._invalidate_cache()

    def action_reset_view(self) -> None:
        view = self.query_one(MoleculeView)
        view.rot_matrix = rotation_matrix(-0.2, -0.5, 0.0)
        view.pan_x = 0.0
        view.pan_y = 0.0
        view.pan_mode = False
        view.highlighted_atoms = set()
        if view.molecule:
            view.camera_distance = max(4.0, view.molecule.radius() * 3.0)
        view._invalidate_cache()

    def _frame_to_tab(self) -> None:
        """Open the current trajectory frame as a standalone structure in a new tab."""
        if not self._frames or not self.molecule or len(self.molecule.atoms) == 0:
            self.notify("No trajectory frame to open", timeout=2)
            return
        import copy
        mol = copy.deepcopy(self.molecule)
        stem = Path(self.filepath).stem if self.filepath else "frame"
        name = f"{stem}_frame{self._frame_idx + 1}"
        self.open_in_new_tab(mol, name)
        self.notify(f"Opened frame {self._frame_idx + 1} as {name}", timeout=2)

    def action_export_png(self) -> None:
        view = self.query_one(MoleculeView)
        if view.molecule is None:
            return
        self.notify("Exporting PNG...", timeout=2)
        self._track_task(self._export_png_async())

    async def _calc_void_volume(self) -> None:
        from pdb_wizard.analyses import void_volume_report
        from pdb_wizard.operations import void_volume
        mol = self.molecule
        self._show_progress("Calculating void volume (geometric)...")
        result_geo = await asyncio.to_thread(void_volume, mol, 100000, 0.0)
        self.query_one("#status-label", Label).update("Calculating void volume (He probe)...")
        self._update_progress(50)
        result_he = await asyncio.to_thread(void_volume, mol, 100000, 1.14)
        self._hide_progress()
        self.push_screen(InfoModal("Void Volume", void_volume_report(result_geo, result_he)))

    async def _calc_surface_area(self) -> None:
        from pdb_wizard.analyses import surface_area_text
        mol = self.molecule
        self._show_progress("Calculating surface area (N2 probe)...")
        text = await asyncio.to_thread(surface_area_text, mol, 100000, 1.4)
        self._hide_progress()
        self.push_screen(InfoModal("Surface Area", text))

    async def _calc_pore_size(self) -> None:
        from pdb_wizard.analyses import run_pore_size
        mol = self.molecule
        self._show_progress("Computing pore size distribution...")
        res = await asyncio.to_thread(run_pore_size, mol, 50000, 50, 20.0)
        self._hide_progress()
        if res.empty:
            self.notify(res.summary, timeout=3)
            return

        class PsdScreen(ModalScreen[None]):
            BINDINGS = [Binding("escape", "close", ""), Binding("q", "close", "")]
            DEFAULT_CSS = ("PsdScreen > Vertical { width: 100%; height: 100%; "
                           "background: $surface; padding: 0 1; }")

            def __init__(self, result):
                super().__init__()
                self._res = result

            def compose(self_inner) -> ComposeResult:
                with Vertical():
                    yield PlotWidget(id="psd-plot")
                    yield Label(f"{self_inner._res.summary}  |  press q to close", id="psd-status")

            def on_mount(self_inner) -> None:
                r = self_inner._res
                self_inner.query_one("#psd-plot", PlotWidget).set_data(
                    r.x, r.y, title=r.title, x_label=r.x_label, y_min=0.0, y_max=r.y_max)

            def action_close(self_inner) -> None:
                self_inner.dismiss(None)

        self.push_screen(PsdScreen(res))

    async def _export_png_async(self) -> None:
        view = self.query_one(MoleculeView)
        if view.molecule is None:
            return

        mol = view.molecule
        if not view.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[], pbc=mol.pbc)

        export_w, export_h = 1600, 1200
        bg = (0, 0, 0) if view.dark_bg else (255, 255, 255)

        pixels, _ = await asyncio.to_thread(
            render_scene, export_w, export_h, mol, view.rot_matrix,
            view.camera_distance, bg_color=bg, pbc=view.pbc, ssaa=2,
            pan=(view.pan_x, view.pan_y),
            licorice=view.licorice, vdw=view.vdw, ribbon=view.ribbon,
            color_mode=view.color_mode, color_by_charge=view.color_by_charge,
            ambient=0.31, diffuse=0.72, specular=0.42, shininess=96.0,
            atom_scale=view.atom_scale, bond_radius=view.bond_radius,
            show_polyhedra=view.show_polyhedra,
            centroid_override=view._locked_centroid,
            density_positions=view._density_positions,
            density_values=view._density_values,
        )

        try:
            from PIL import Image
        except ImportError:
            self.notify("Pillow required: pip install Pillow", timeout=3)
            return

        stem = Path(self.filepath).stem or "untitled"
        out_path = Path(self.filepath).parent / f"{stem}.png"
        img = Image.fromarray(pixels)
        await asyncio.to_thread(img.save, str(out_path))
        self.notify(f"Saved {out_path}", timeout=3)

    async def _export_gif_async(self) -> None:
        view = self.query_one(MoleculeView)
        if view.molecule is None:
            return
        try:
            from PIL import Image
        except ImportError:
            self.notify("Pillow required: pip install Pillow", timeout=3)
            return

        mol = view.molecule
        if not view.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[], pbc=mol.pbc)

        export_w, export_h = 800, 600
        bg = (0, 0, 0) if view.dark_bg else (255, 255, 255)
        n_frames_gif = 36
        self._show_progress("Rendering GIF...", total=n_frames_gif)

        frames_img = []
        base_rot = view.rot_matrix.copy()
        for i in range(n_frames_gif):
            angle = 2 * np.pi * i / n_frames_gif
            rot = rotation_matrix(0, angle, 0) @ base_rot
            pixels, _ = await asyncio.to_thread(
                render_scene, export_w, export_h, mol, rot,
                view.camera_distance, bg_color=bg, pbc=view.pbc, ssaa=2,
                pan=(view.pan_x, view.pan_y),
                licorice=view.licorice, vdw=view.vdw, ribbon=view.ribbon,
                color_mode=view.color_mode, color_by_charge=view.color_by_charge,
                ambient=0.31, diffuse=0.72, specular=0.42, shininess=96.0,
                atom_scale=view.atom_scale, bond_radius=view.bond_radius,
                show_polyhedra=view.show_polyhedra,
                centroid_override=view._locked_centroid,
                density_positions=view._density_positions,
                density_values=view._density_values,
            )
            frames_img.append(Image.fromarray(pixels))
            self._update_progress(i + 1)

        self._hide_progress()
        stem = Path(self.filepath).stem or "untitled"
        out_path = Path(self.filepath).parent / (stem + ".gif")
        await asyncio.to_thread(
            frames_img[0].save,
            str(out_path),
            save_all=True,
            append_images=frames_img[1:],
            duration=80,
            loop=0,
        )
        self.notify(f"Saved {out_path} ({n_frames_gif} frames)", timeout=3)

    # --- Panel management ---

    def _active_panel_table(self) -> DataTable | None:
        ops = self.query_one(OperationsPanel)
        if ops.has_class("visible"):
            for dt in ops.query(DataTable):
                return dt
        geom = self.query_one(GeometryPanel)
        if geom.has_class("visible"):
            tabs = geom.query_one(TabbedContent)
            pane = tabs.get_pane(tabs.active)
            for dt in pane.query(DataTable):
                return dt
        return None

    def action_panel_next(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_next()
            return
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_down()

    def action_panel_prev(self) -> None:
        if self.query_one(VisualPanel).has_class("visible"):
            self.screen.focus_previous()
            return
        dt = self._active_panel_table()
        if dt is not None:
            dt.action_cursor_up()

    def action_close_panel(self) -> None:
        geom = self.query_one(GeometryPanel)
        vis = self.query_one(VisualPanel)
        if geom.has_class("visible") or vis.has_class("visible"):
            self._close_panels()
            view = self.query_one(MoleculeView)
            view._invalidate_cache()
            view.focus()

    def _close_panels(self) -> None:
        """Close right-side panels. Operations panel is pinned and stays open."""
        view = self.query_one(MoleculeView)
        geom = self.query_one(GeometryPanel)
        vis = self.query_one(VisualPanel)
        if geom.has_class("visible"):
            geom.remove_class("visible")
            view.highlighted_atoms = set()
            view._stop_blink()
        if vis.has_class("visible"):
            vis.remove_class("visible")

    def action_toggle_operations(self) -> None:
        ops = self.query_one(OperationsPanel)
        view = self.query_one(MoleculeView)
        if ops.has_class("visible"):
            ops.remove_class("visible")
            view.focus()
        else:
            ops.add_class("visible")
            for dt in ops.query(DataTable):
                dt.focus()
                break
        view._invalidate_cache()

    def action_toggle_geometry(self) -> None:
        panel = self.query_one(GeometryPanel)
        was_visible = panel.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            panel.add_class("visible")
            # Populate active tab now that the panel is being shown
            panel._populate_active_tab()
            for dt in panel.query(DataTable):
                dt.focus()
                panel._emit_current_highlight(dt)
                break
        else:
            view.focus()
        view._invalidate_cache()

    def action_toggle_visual(self) -> None:
        vis = self.query_one(VisualPanel)
        was_visible = vis.has_class("visible")
        self._close_panels()
        view = self.query_one(MoleculeView)
        if not was_visible:
            vis.set_state(
                licorice=view.licorice, vdw=view.vdw, ribbon=view.ribbon,
                ambient=view.ambient, diffuse=view.diffuse,
                specular=view.specular, shininess=view.shininess,
                atom_scale=view.atom_scale, bond_radius=view.bond_radius,
            )
            vis.add_class("visible")
            for child in vis.query("*"):
                if child.can_focus:
                    child.focus()
                    break
        else:
            view.focus()
        view._invalidate_cache()

    # --- Event handlers ---

    def on_visual_panel_color_mode_changed(self, event: VisualPanel.ColorModeChanged) -> None:
        view = self.query_one(MoleculeView)
        view.color_mode = event.mode
        view.color_by_charge = (event.mode == "charge")
        view._invalidate_cache()

    def on_visual_panel_toggle_changed(self, event: VisualPanel.ToggleChanged) -> None:
        view = self.query_one(MoleculeView)
        if event.toggle == "bonds":
            view.show_bonds = event.value
        elif event.toggle == "atom_numbers":
            view.show_atom_numbers = event.value
        elif event.toggle == "charge_colors":
            view.color_by_charge = event.value
        elif event.toggle == "hide_water":
            view.hide_water = event.value
        elif event.toggle == "polyhedra":
            view.show_polyhedra = event.value
        view._invalidate_cache()

    def on_visual_panel_theme_changed(self, event: VisualPanel.ThemeChanged) -> None:
        view = self.query_one(MoleculeView)
        view.dark_bg = event.dark
        self.theme = "textual-dark" if event.dark else "textual-light"
        view._invalidate_cache()

    def on_visual_panel_style_changed(self, event: VisualPanel.StyleChanged) -> None:
        view = self.query_one(MoleculeView)
        view.licorice = event.licorice
        view.vdw = event.vdw
        view.ribbon = event.ribbon
        view.bond_radius = 0.15 if event.licorice else 0.08
        vis = self.query_one(VisualPanel)
        if vis.has_class("visible"):
            vis.set_state(
                licorice=view.licorice, vdw=view.vdw, ribbon=view.ribbon,
                ambient=view.ambient, diffuse=view.diffuse,
                specular=view.specular, shininess=view.shininess,
                atom_scale=view.atom_scale, bond_radius=view.bond_radius,
            )
        view._invalidate_cache()

    def on_visual_panel_lighting_changed(self, event: VisualPanel.LightingChanged) -> None:
        view = self.query_one(MoleculeView)
        view.ambient = event.ambient
        view.diffuse = event.diffuse
        view.specular = event.specular
        view.shininess = event.shininess
        view._invalidate_cache()

    def on_visual_panel_size_changed(self, event: VisualPanel.SizeChanged) -> None:
        view = self.query_one(MoleculeView)
        view.atom_scale = event.atom_scale
        view.bond_radius = event.bond_radius
        view._invalidate_cache()

    def on_menu_bar_menu_action(self, event: MenuBar.MenuAction) -> None:
        menu_map = {
            "file": (MENU_FILE, 0),
            "edit": (MENU_EDIT, 6),
            "view": (MENU_VIEW, 12),
            "analysis": (MENU_ANALYSIS, 20),
        }
        if event.action == "undo":
            self._do_undo()
        elif event.action == "redo":
            self._do_redo()
        elif event.action in menu_map:
            items, offset = menu_map[event.action]
            self.push_screen(
                MenuDropdown(items, offset=offset),
                callback=self._handle_menu_action,
            )

    def _handle_menu_action(self, action: str) -> None:
        if not action:
            return
        # Map menu actions to existing methods
        action_map = {
            "quit": lambda: self.exit(),
            "frame_to_tab": lambda: self._frame_to_tab(),
            "export_png": lambda: self.action_export_png(),
            "export_gif": lambda: self._track_task(self._export_gif_async()),
            "toggle_bonds": lambda: self.action_toggle_bonds(),
            "toggle_style": lambda: self.action_toggle_style(),
            "toggle_bg": lambda: self.action_toggle_bg(),
            "toggle_atom_numbers": lambda: self.action_toggle_atom_numbers(),
            "toggle_charge_colors": lambda: self.action_toggle_charge_colors(),
            "cycle_color_mode": lambda: self._cycle_color_mode(),
            "toggle_hide_water": lambda: self.action_toggle_hide_water(),
            "toggle_measure": lambda: self.action_toggle_measure(),
            "clear_measurements": lambda: self._clear_measurements(),
            "toggle_polyhedra": lambda: self.action_toggle_polyhedra(),
            "toggle_orthographic": lambda: self.action_toggle_orthographic(),
            "toggle_operations": lambda: self.action_toggle_operations(),
            "toggle_geometry": lambda: self.action_toggle_geometry(),
            "toggle_visual": lambda: self.action_toggle_visual(),
            "reset_view": lambda: self.action_reset_view(),
        }
        if action in action_map:
            action_map[action]()
        else:
            # Dispatch to operations panel handler
            from pdb_wizard.panels import OperationsPanel
            self.on_operations_panel_run_command(OperationsPanel.RunCommand(action))

    def on_molecule_view_atom_picked(self, event: MoleculeView.AtomPicked) -> None:
        view = self.query_one(MoleculeView)
        mol = self.molecule
        idx = event.atom_index
        if not mol or not mol.atoms or idx >= len(mol.atoms):
            return
        atom = mol.atoms[idx]

        if view.measure_mode:
            # Measure mode: clicks accumulate atoms and pin labels
            picked = view._picked_atoms
            if idx not in picked:
                picked.append(idx)
            if len(picked) > 4:
                picked.pop(0)
            view.highlighted_atoms = set(picked)
            view._start_blink()

            if len(picked) == 1:
                a = mol.atoms[picked[0]]
                self.notify(f"Atom {picked[0]+1}: {a.element.symbol} — click more to measure", timeout=3)
            elif len(picked) == 2:
                i, j = picked
                if mol.pbc:
                    d = mol.pbc.min_image(mol.atoms[i].x - mol.atoms[j].x)
                else:
                    d = float(np.linalg.norm(mol.atoms[i].x - mol.atoms[j].x))
                label = f"{d:.3f}A"
                view._measurements.append(([i, j], label))
                self.notify(f"Distance {i+1}-{j+1} = {d:.4f} A (pinned)", timeout=3)
                view._picked_atoms = []  # reset for next measurement
                view.highlighted_atoms = set()
                view._stop_blink()
            elif len(picked) == 3:
                i, j, k = picked
                if mol.pbc:
                    v1 = mol.pbc.wrap(mol.atoms[i].x - mol.atoms[j].x)
                    v2 = mol.pbc.wrap(mol.atoms[k].x - mol.atoms[j].x)
                else:
                    v1 = mol.atoms[i].x - mol.atoms[j].x
                    v2 = mol.atoms[k].x - mol.atoms[j].x
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                label = f"{angle:.1f}deg"
                view._measurements.append(([i, j, k], label))
                self.notify(f"Angle {i+1}-{j+1}-{k+1} = {angle:.2f} deg (pinned)", timeout=3)
                view._picked_atoms = []
                view.highlighted_atoms = set()
                view._stop_blink()
        elif event.shift:
            # Legacy shift+click measurement (toast only)
            picked = view._picked_atoms
            if idx not in picked:
                picked.append(idx)
            if len(picked) > 3:
                picked.pop(0)
            view.highlighted_atoms = set(picked)
            view._start_blink()

            if len(picked) == 1:
                a = mol.atoms[picked[0]]
                self.notify(f"Atom {picked[0]+1}: {a.element.symbol} ({a.name})", timeout=3)
            elif len(picked) == 2:
                i, j = picked
                if mol.pbc:
                    d = mol.pbc.min_image(mol.atoms[i].x - mol.atoms[j].x)
                else:
                    d = float(np.linalg.norm(mol.atoms[i].x - mol.atoms[j].x))
                self.notify(
                    f"Distance {i+1}:{mol.atoms[i].element.symbol} - "
                    f"{j+1}:{mol.atoms[j].element.symbol} = {d:.4f} A", timeout=5,
                )
            elif len(picked) == 3:
                i, j, k = picked
                if mol.pbc:
                    v1 = mol.pbc.wrap(mol.atoms[i].x - mol.atoms[j].x)
                    v2 = mol.pbc.wrap(mol.atoms[k].x - mol.atoms[j].x)
                else:
                    v1 = mol.atoms[i].x - mol.atoms[j].x
                    v2 = mol.atoms[k].x - mol.atoms[j].x
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                self.notify(
                    f"Angle {i+1}-{j+1}-{k+1} = {angle:.2f} deg", timeout=5,
                )
        else:
            # Single click: show atom info, highlight it
            view._picked_atoms = [idx]
            view.highlighted_atoms = {idx}
            view._start_blink()
            charge_str = f"  q={atom.charge:.4f}" if abs(atom.charge) > 1e-10 else ""
            self.notify(
                f"Atom {idx+1}: {atom.element.symbol} ({atom.name})"
                f"  [{atom.x[0]:.3f}, {atom.x[1]:.3f}, {atom.x[2]:.3f}]{charge_str}",
                timeout=5,
            )
        view._invalidate_cache()

    def on_geometry_panel_highlight_atoms(self, event: GeometryPanel.HighlightAtoms) -> None:
        view = self.query_one(MoleculeView)
        view.highlighted_atoms = set(event.atom_indices)
        if view.highlighted_atoms:
            view._start_blink()
        else:
            view._stop_blink()
        view._invalidate_cache()

    def on_geometry_panel_edit_coord(self, event: GeometryPanel.EditCoord) -> None:
        idx = event.atom_index
        if not self.molecule or idx >= len(self.molecule.atoms):
            return
        a = self.molecule.atoms[idx]
        self.push_screen(
            InputModal(
                f"Edit coordinates for atom {idx + 1} ({a.element.symbol}):",
                default=f"{event.x:.6f} {event.y:.6f} {event.z:.6f}",
                hint="x y z",
            ),
            callback=lambda val: self._do_edit_coord(idx, val),
        )

    def _do_edit_coord(self, idx: int, value: str) -> None:
        if not value.strip():
            return
        if self._edit_running():
            return
        try:
            import numpy as np
            parts = value.strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            self._save_undo()
            self.molecule.atoms[idx].x = np.array([x, y, z])
            self._start_edit(self._refresh_molecule(f"Updated atom {idx + 1} position"))
        except (ValueError, IndexError):
            self.notify("Invalid input. Need: x y z", timeout=3)

    def on_operations_panel_run_command(self, event: OperationsPanel.RunCommand) -> None:
        mol = self.molecule
        cmd = event.command
        view = self.query_one(MoleculeView)

        if cmd == "info":
            from pdb_wizard.analyses import system_info_text
            self.push_screen(InfoModal("System Info", system_info_text(mol, self.filepath)))
            return

        elif cmd == "check":
            from pdb_wizard.analyses import check_text
            self.push_screen(InfoModal("Check Structure", check_text(mol, self._frames)))
            return

        elif cmd == "symmetry":
            from pdb_wizard.analyses import symmetry_text
            self.push_screen(InfoModal("Find Symmetry", symmetry_text(mol)))
            return

        elif cmd == "write_crystal":
            if not self._require_cell("Writing a CRYSTAL23 input"):
                return
            self.push_screen(
                InputModal("Space group for the CRYSTAL23 input:", default="1",
                           hint="1 writes every atom in P1. A number from Find Symmetry "
                                "writes only that group's asymmetric unit."),
                callback=lambda v: self._crystal_step_group(v, mol),
            )
            return

        elif cmd == "rdf":
            if len(mol.atoms) == 0:
                self.notify("No atoms loaded yet", timeout=2)
                return
            self.push_screen(RdfScreen(mol, frames=self._frames))
            return

        elif cmd == "coordination":
            if len(mol.atoms) == 0:
                self.notify("No atoms loaded", timeout=2)
                return
            self.push_screen(CoordinationScreen(mol, frames=self._frames))
            return

        elif cmd == "hbonds":
            if len(mol.atoms) == 0:
                self.notify("No atoms loaded", timeout=2)
                return
            self.push_screen(HbondScreen(mol, self._frames))
            return

        elif cmd == "msd":
            if not self._frames or len(self._frames) < 3:
                self.notify("Need a trajectory with 3+ frames for MSD", timeout=3)
                return
            self.push_screen(MsdScreen(mol, self._frames))
            return

        elif cmd == "rmsd":
            if not self._frames or len(self._frames) < 3:
                self.notify("Need a trajectory with 3+ frames for RMSD", timeout=3)
                return
            self.push_screen(RmsdScreen(mol, self._frames))
            return

        elif cmd == "gyration":
            if not self._frames or len(self._frames) < 2:
                self.notify("Need a trajectory for Rg", timeout=3)
                return
            self.push_screen(GyrationScreen(mol, self._frames))
            return

        elif cmd == "density":
            if len(mol.atoms) == 0:
                self.notify("No atoms loaded", timeout=2)
                return
            self.push_screen(DensityScreen(mol, self._frames))
            return

        elif cmd == "pxrd":
            if len(mol.atoms) == 0 or not self._has_real_cell(mol):
                self.notify("Need a crystal structure with PBC for PXRD", timeout=3)
                return
            self.push_screen(PxrdScreen(mol, self._frames))
            return

        elif cmd == "density3d":
            if not self._frames or len(self._frames) < 2:
                self.notify("Need a trajectory with PBC for 3D density", timeout=3)
                return
            if not self._has_real_cell(mol):
                self.notify("Need PBC for 3D density map", timeout=3)
                return
            self._show_density3d()
            return

        elif cmd in ("input_generator", "isotherm"):
            # "isotherm" kept as legacy command code for any saved keymaps;
            # both open the unified Input Generator (defaults to MPMC isotherm).
            self.push_screen(InputGeneratorScreen(mol, self.filepath))
            return

        elif cmd == "isotherm_plot":
            self.push_screen(
                FileSaveModal("Isotherm results directory:", default="isotherm", button_label="Open"),
                callback=self._plot_isotherm,
            )
            return

        elif cmd == "db_search":
            self.push_screen(DatabaseSearchScreen())
            return

        elif cmd == "energy_plot":
            self._show_energy_plot()
            return

        elif cmd == "undo":
            self._do_undo()
            return

        elif cmd == "void_vol":
            if len(mol.atoms) == 0 or not self._has_real_cell(mol):
                self.notify("Need PBC for void volume", timeout=3)
                return
            self.notify("Calculating void volume (100k samples)...", timeout=2)
            self._track_task(self._calc_void_volume())
            return

        elif cmd == "surface_area":
            if not self._has_real_cell(mol):
                self.notify("Need PBC for surface area", timeout=3)
                return
            self._track_task(self._calc_surface_area())
            return

        elif cmd == "pore_size":
            if not self._has_real_cell(mol):
                self.notify("Need PBC for pore size", timeout=3)
                return
            self._track_task(self._calc_pore_size())
            return

        elif cmd == "substitute":
            self.push_screen(
                InputModal("Substitute element:", default="Cu Zn", hint="old new (e.g. Cu Zn)"),
                callback=self._do_substitute,
            )
            return

        elif cmd == "pack_sorbates":
            if not self._require_cell("Packing sorbates"):
                return
            self.push_screen(SorbateModal(standalone=True), callback=self._pack_step_model)
            return

        elif cmd == "reduce_cell":
            if not self._has_real_cell(mol):
                self.notify("Need PBC for supercell reduction", timeout=3)
                return
            self.push_screen(ReduceCellScreen(mol), callback=self._apply_reduced_cell)
            return

        elif cmd == "open_geom":
            self.action_toggle_geometry()
            return

        elif cmd == "wrap_center":
            if not self._require_cell("Wrapping atoms"):
                return
            self.push_screen(
                ConfirmModal(
                    "Wrap Atoms (centered)",
                    "Wrap all atoms into the unit cell centered at the origin.\n"
                    "Fractional coords mapped to (-0.5, 0.5).",
                ),
                callback=lambda ok: self._do_wrap(ok, forward=False),
            )
            return

        elif cmd == "wrap_forward":
            if not self._require_cell("Wrapping atoms"):
                return
            self.push_screen(
                ConfirmModal(
                    "Wrap Atoms (forward)",
                    "Wrap all atoms into the unit cell forward of the origin.\n"
                    "Fractional coords mapped to (0, 1).",
                ),
                callback=lambda ok: self._do_wrap(ok, forward=True),
            )
            return

        elif cmd == "sort":
            if not self._require_cell("Sorting atoms into molecules"):
                return
            self.push_screen(
                ConfirmModal(
                    "Sort Atoms",
                    "Sort atoms by element within each molecule, then sort\n"
                    "molecules by size and composition. Bonds will be\n"
                    "recalculated.",
                ),
                callback=self._do_sort,
            )
            return

        elif cmd == "del_lone":
            if not self._require_cell("Finding lone atoms"):
                return
            from pdb_wizard.checks import lone_atom_summary, lone_atom_warning
            found = lone_atom_summary(mol)
            caution = lone_atom_warning(mol)
            self.push_screen(
                ConfirmModal(
                    "Delete Lone Atoms",
                    "Remove all atoms that have no neighbors within\n"
                    "VDW contact distance. Undo restores them.\n\n"
                    + (f"Would delete: {found}" if found else "No lone atoms found.")
                    + (f"\n\n⚠ {caution}" if caution else ""),
                ),
                callback=self._do_del_lone,
            )
            return

        elif cmd == "extend":
            if not self._require_cell("Extending the cell"):
                return
            self.push_screen(
                ExtendAxisModal(mol.pbc),
                callback=self._do_extend,
            )
            return

        elif cmd == "edit_h":
            if not self._require_cell("Editing H distances"):
                return
            self.push_screen(
                InputModal(
                    "Edit H bond distances:",
                    default="O 1.0",
                    hint="element and distance, e.g. 'O 1.0' or 'C 1.09'",
                ),
                callback=self._do_edit_h,
            )
            return

        elif cmd == "update_cell":
            if mol.pbc is not None:
                pbc = mol.pbc
                default = f"{pbc.a:.4f} {pbc.b:.4f} {pbc.c:.4f} {pbc.alpha:.2f} {pbc.beta:.2f} {pbc.gamma:.2f}"
            else:
                default = "10.0 10.0 10.0 90.0 90.0 90.0"
            self.push_screen(
                InputModal(
                    "Unit cell parameters:",
                    default=default,
                    hint="a b c alpha beta gamma  •  type 'none' to remove the box",
                ),
                callback=self._do_update_cell,
            )
            return

        elif cmd == "qeq_charges":
            self.push_screen(
                ConfirmModal(
                    "Generate QEq Charges",
                    "Compute Rappe-Goddard charge equilibration\n"
                    "charges for all atoms? This will overwrite\n"
                    "any existing charges.\n\n"
                    f"{len(mol.atoms)} atoms",
                ),
                callback=self._do_qeq_charges,
            )
            return

        elif cmd in ("write_pdb", "write_lammps") and len(mol.atoms) == 0:
            self.notify("Nothing to save: no atoms loaded", timeout=3, severity="warning")
            return

        elif cmd == "write_xyz":
            stem = Path(self.filepath).stem
            default = str(Path(self.filepath).parent / f"{stem}_out.xyz")
            self.push_screen(
                FileSaveModal("Save XYZ as:", default=default),
                callback=lambda v: self._do_write_xyz(v, mol),
            )
            return

        elif cmd == "write_pdb":
            stem = Path(self.filepath).stem
            default = str(Path(self.filepath).parent / f"{stem}_out.pdb")
            self.push_screen(
                FileSaveModal("Save standard PDB as:", default=default),
                callback=lambda v: self._do_write_pdb(v, mol),
            )
            return

        elif cmd == "write_cif":
            if not self._require_cell("Writing a CIF"):
                return
            stem = Path(self.filepath).stem
            default = str(Path(self.filepath).parent / f"{stem}_out.cif")
            self.push_screen(FileSaveModal("Save CIF as:", default=default),
                             callback=lambda v: self._do_write_cif(v, mol))
            return

        elif cmd == "write_poscar":
            if not self._require_cell("Writing a POSCAR"):
                return
            stem = Path(self.filepath).stem
            default = str(Path(self.filepath).parent / "POSCAR")
            self.push_screen(FileSaveModal("Save POSCAR as:", default=default),
                             callback=lambda v: self._do_write_poscar(v, mol))
            return

        elif cmd == "write_lammps":
            stem = Path(self.filepath).stem
            default = str(Path(self.filepath).parent / (stem + ".data"))
            self.push_screen(FileSaveModal("Save LAMMPS data as:", default=default),
                             callback=lambda v: self._do_write_lammps(v, mol))
            return

        elif cmd == "open_file":
            self.push_screen(
                FileSaveModal(
                    "Open structure file:",
                    default=self.filepath or str(Path.cwd()) + "/",
                    button_label="Open",
                ),
                callback=self._do_open_file,
            )
            return

        elif cmd == "frame_to_tab":
            self._frame_to_tab()
            return

        elif cmd == "export_png":
            self.action_export_png()
            return

        elif cmd == "export_gif":
            self._track_task(self._export_gif_async())
            return

        elif cmd == "write_mpmc":
            if not self._require_cell("Writing an MPMC PDB"):
                return
            # The wizard mutates charges and force-field parameters on the
            # live molecule, so make the whole export undoable in one step.
            self._save_undo()
            self._mpmc_state = {"write_charges": False, "write_ff": False,
                                "mol": mol, "filepath": self.filepath}
            # Check if atoms already have charges
            has_charges = any(abs(a.charge) > 1e-10 for a in mol.atoms)
            if has_charges:
                total_q = sum(a.charge for a in mol.atoms)
                self.push_screen(
                    ConfirmModal(
                        "Charges Detected",
                        f"Atoms already have charges (total: {total_q:.4f} e).\n\n"
                        "Keep existing charges?\n"
                        "Yes = keep, No = choose new charges",
                        yes_no=True,
                    ),
                    callback=self._mpmc_step_keep_charges,
                )
            else:
                self._mpmc_step_charge_source()
            return

        elif cmd == "quit":
            # Direct exit: selecting Quit from the menu is already deliberate,
            # so it doesn't use the bare-'q' "press again" confirmation.
            self.exit()
            return

        view._invalidate_cache()

    def _require_cell(self, what: str) -> bool:
        """Toast and return False when the active molecule has no unit cell.

        Core operations such as wrap/extend/sort and the PDB/CIF/POSCAR/MPMC
        writers are no-ops without a cell; guarding here keeps the TUI from
        reporting success for something that did nothing.
        """
        if self._has_real_cell(self.molecule):
            return True
        self.notify(
            f"{what} needs a unit cell (PBC). This structure has none — "
            "set one with Edit > Update Cell first.",
            timeout=5, severity="warning",
        )
        return False

    def _discard_last_undo(self, stack=None) -> None:
        """Drop the most recent undo entry (an operation turned out to be a no-op).
        ``stack`` is the undo stack the entry was pushed on (default: the active tab's)."""
        stack = self._undo_stack if stack is None else stack
        if stack:
            stack.pop()

    def _write_text_file(self, filepath: str, writer, what: str) -> bool:
        """Render with ``writer(handle)`` into memory, then write the file.

        Buffering means a writer that raises (e.g. CellRequiredError) never
        leaves a truncated or empty file behind. Returns True on success.
        """
        import io as _io_mod
        buf = _io_mod.StringIO()
        try:
            writer(buf)
        except ValueError as e:
            self.notify(f"{what} failed: {e}", timeout=6, severity="error")
            return False
        try:
            with open(filepath, "w") as f:
                f.write(buf.getvalue())
        except OSError as e:
            self.notify(f"Error: {e}", timeout=4, severity="error")
            return False
        self.notify(f"Wrote {filepath}", timeout=3)
        return True

    async def _refresh_molecule(self, msg: str = "", mol=None) -> None:
        """Redetect bonds and refresh the view and geometry panel.

        ``mol`` is the molecule that was edited (default: the active one). If another tab
        is on screen by the time a step finishes, the view and panel are left alone.
        """
        mol = self.molecule if mol is None else mol
        self._show_progress("Detecting bonds...", total=100)
        await asyncio.sleep(0)

        def _bond_progress(frac: float) -> None:
            self.call_from_thread(self._update_progress, int(frac * 90))

        await asyncio.to_thread(mol.detect_bonds, progress_callback=_bond_progress)
        self._update_progress(80)
        view = self.query_one(MoleculeView)
        if mol is self.molecule:
            view.set_molecule(mol)
        self.query_one("#status-label", Label).update("Computing geometry tables...")
        await asyncio.sleep(0)
        # Pre-compute the heavy geometry data in a thread
        def _geo_progress(pct: int, label: str) -> None:
            self.call_from_thread(self._update_progress, 80 + pct // 7)
            self.call_from_thread(
                self.query_one("#status-label", Label).update, label
            )

        await asyncio.to_thread(self._precompute_geometry, mol, _geo_progress)
        self._update_progress(95)
        self.query_one("#status-label", Label).update("Populating tables...")
        await asyncio.sleep(0)
        panel = self.query_one(GeometryPanel)
        if mol is self.molecule:
            panel._molecule = mol
            if panel.is_mounted:
                await self._populate_tables_async(panel, mol)
        else:
            for attr in ('_cached_bond_lengths', '_cached_angles', '_cached_dihedrals',
                         '_cached_contacts', '_cached_molecules', '_cached_lone_atoms'):
                if hasattr(mol, attr):
                    delattr(mol, attr)
        self._update_progress(100)
        self._hide_progress()
        if msg:
            self.notify(msg, timeout=2)
        view._invalidate_cache()

    def _do_wrap(self, confirmed: bool, forward: bool) -> None:
        if not confirmed or self._edit_running():
            return
        self._start_edit(self._do_wrap_async(forward))

    async def _do_wrap_async(self, forward: bool) -> None:
        from pdb_wizard.operations import wrap_atoms
        mol = self.molecule
        self._save_undo()
        self._show_progress("Wrapping atoms...")
        await asyncio.sleep(0)
        await asyncio.to_thread(wrap_atoms, mol, forward)
        label = "forward" if forward else "centered"
        await self._refresh_molecule(f"Wrapped atoms ({label})", mol)

    def _do_sort(self, confirmed: bool) -> None:
        if not confirmed or self._edit_running():
            return
        self._start_edit(self._do_sort_async())

    async def _do_sort_async(self) -> None:
        mol = self.molecule
        self._save_undo()
        from pdb_wizard.operations import sort_system
        self._show_progress("Sorting atoms...")
        await asyncio.sleep(0)
        await asyncio.to_thread(sort_system, mol, False)
        await self._refresh_molecule("Sorted atoms", mol)

    def _do_del_lone(self, confirmed: bool) -> None:
        if not confirmed or self._edit_running():
            return
        self._start_edit(self._do_del_lone_async())

    async def _do_del_lone_async(self) -> None:
        mol = self.molecule
        self._save_undo()
        undo = self._undo_stack
        from pdb_wizard.geometry import remove_lone_atoms
        self._show_progress("Finding lone atoms...")
        await asyncio.sleep(0)
        _, deleted = await asyncio.to_thread(remove_lone_atoms, mol)
        if not deleted:
            self._discard_last_undo(undo)
            self._hide_progress()
            self.notify("No lone atoms found", timeout=3)
            return
        n = len(deleted)
        await self._refresh_molecule(f"Deleted {n} lone atom{'s' if n != 1 else ''}", mol)

    def _do_open_file(self, value: str | None) -> None:
        """Callback for the File > Open... picker."""
        if not value or not value.strip():
            return
        filepath = value.strip()
        # If the buffer is empty (no atoms, fresh launch), reuse the active
        # tab instead of opening a new one — avoids a leftover blank tab.
        if not self.molecule.atoms and not self.filepath:
            try:
                from pdb_wizard.io import read_file
                if self._is_trajectory_file(filepath):
                    # Same background path as "open in new tab", but into the
                    # empty active tab so no blank tab is left behind.
                    tab = self._tabs[self._active_tab]
                    tab.filepath = filepath
                    self.filepath = filepath
                    self._refresh_tab_bar()
                    self.notify(f"Loading {Path(filepath).name}...", timeout=2)
                    self._track_task(self._load_traj_into_tab(filepath, tab))
                    return
                load_msgs: list[str] = []
                mol = read_file(filepath, on_message=load_msgs.append)
                self.molecule = mol
                self.filepath = filepath
                self._tabs[self._active_tab].molecule = mol
                self._tabs[self._active_tab].filepath = filepath
                view = self.query_one(MoleculeView)
                view.set_molecule(mol, keep_camera=False)
                view._invalidate_cache()
                panel = self.query_one(GeometryPanel)
                panel.set_molecule(mol)
                self._update_title()
                self._refresh_tab_bar()
                self.notify(f"Opened {Path(filepath).name}", timeout=2)
                self.notify_overlaps_removed(load_msgs)
                self.notify_load_notes(mol)
            except Exception as e:
                self.notify(f"Error opening file: {e}", timeout=4, severity="error")
            return
        # Otherwise open in a new tab.
        self._open_new_tab(filepath)

    def _do_extend(self, value: tuple) -> None:
        if not value or len(value) != 3:
            return
        na, nb, nc = value
        if na == 0 and nb == 0 and nc == 0:
            return  # no-op
        if self._edit_running():
            return
        self._start_edit(self._do_extend_async(na, nb, nc))

    async def _do_extend_async(self, na: int, nb: int, nc: int) -> None:
        """Add na, nb, nc extra copies along a, b, c (classic-menu counting)."""
        size = f"{na + 1}x{nb + 1}x{nc + 1}"
        target = self.molecule
        self._save_undo()
        from pdb_wizard.operations import extend_axis
        self._show_progress(f"Extending to {size}...")
        await asyncio.sleep(0)

        def _extend_all(mol):
            # All three axes inside one thread — main loop only sees the
            # final state, never the intermediate (renderer doesn't render
            # the partial 2x1x1 / 2x2x1 frames).
            if na > 0:
                extend_axis(mol, 0, na)
            if nb > 0:
                extend_axis(mol, 1, nb)
            if nc > 0:
                extend_axis(mol, 2, nc)
            return mol

        await asyncio.to_thread(_extend_all, target)
        await self._refresh_molecule(f"Extended to {size}", target)

    def _confirm_overwrite(self, filepath: str, callback) -> None:
        """If filepath exists, prompt to overwrite. Otherwise call callback directly."""
        if Path(filepath).exists():
            self.push_screen(
                ConfirmModal(
                    "Overwrite?",
                    f"{filepath}\nalready exists. Overwrite?",
                    yes_no=True,
                ),
                callback=lambda ok: callback(filepath) if ok else None,
            )
        else:
            callback(filepath)

    # The writers take the molecule the Save dialog was opened for (``mol``); the active
    # tab can change while the dialog is up. Without it they write the active molecule.

    def _do_write_xyz(self, value: str, mol=None) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), lambda p: self._write_xyz_file(p, mol))

    def _write_xyz_file(self, filepath: str, mol=None) -> None:
        from pdb_wizard.io import write_xyz
        mol = self.molecule if mol is None else mol
        try:
            with open(filepath, "w") as f:
                write_xyz(mol, f)
            self.notify(f"Wrote {filepath}", timeout=3)
        except OSError as e:
            self.notify(f"Error: {e}", timeout=3)

    def _do_write_pdb(self, value: str, mol=None) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), lambda p: self._write_pdb_file(p, mol))

    def _write_pdb_file(self, filepath: str, mol=None) -> None:
        from pdb_wizard.io import write_standard_pdb
        mol = self.molecule if mol is None else mol
        self._write_text_file(
            filepath, lambda f: write_standard_pdb(mol, f), "Writing PDB",
        )

    def _do_write_cif(self, value: str, mol=None) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), lambda p: self._write_cif_file(p, mol))

    def _write_cif_file(self, filepath: str, mol=None) -> None:
        from pdb_wizard.formats import write_cif
        mol = self.molecule if mol is None else mol
        self._write_text_file(filepath, lambda f: write_cif(mol, f), "Writing CIF")

    def _do_write_poscar(self, value: str, mol=None) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), lambda p: self._write_poscar_file(p, mol))

    def _write_poscar_file(self, filepath: str, mol=None) -> None:
        from pdb_wizard.formats import write_poscar
        mol = self.molecule if mol is None else mol
        self._write_text_file(
            filepath, lambda f: write_poscar(mol, f), "Writing POSCAR",
        )

    def _crystal_step_group(self, value: str | None, mol=None) -> None:
        if value is None:
            return
        from pdb_wizard.crystal import crystal_reduction_choice
        mol = self.molecule if mol is None else mol
        try:
            reduction = crystal_reduction_choice(mol, value)
        except ValueError as e:
            self.notify(str(e), timeout=12, severity="warning")
            return
        stem = Path(self.filepath).stem
        default = str(Path(self.filepath).parent / f"{stem}.d12")
        self.push_screen(
            FileSaveModal("Save CRYSTAL23 input as:", default=default),
            callback=lambda path: self._do_write_crystal(path, reduction, mol),
        )

    def _do_write_crystal(self, value: str, reduction, mol=None) -> None:
        if not value or not value.strip():
            return
        from pdb_wizard.crystal import write_crystal_d12
        mol = self.molecule if mol is None else mol
        self._confirm_overwrite(
            value.strip(),
            lambda path: self._write_text_file(
                path, lambda f: write_crystal_d12(mol, f, reduction),
                "Writing CRYSTAL23 input"),
        )

    def _do_write_lammps(self, value: str, mol=None) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), lambda p: self._write_lammps_file(p, mol))

    def _write_lammps_file(self, filepath: str, mol=None) -> None:
        from pdb_wizard.formats import write_lammps_data
        mol = self.molecule if mol is None else mol
        self._write_text_file(
            filepath, lambda f: write_lammps_data(mol, f), "Writing LAMMPS data",
        )

    def _apply_reduced_cell(self, result: Molecule | None) -> None:
        if result is None or self._edit_running():
            return
        self._save_undo()
        self.molecule = result
        self._tabs[self._active_tab].molecule = result
        view = self.query_one(MoleculeView)
        view.set_molecule(result, keep_camera=False)
        view._invalidate_cache()
        panel = self.query_one(GeometryPanel)
        panel.set_molecule(result)
        self._update_title()

    def _do_substitute(self, value: str) -> None:
        if not value.strip():
            return
        parts = value.strip().split()
        if len(parts) != 2:
            self.notify("Enter: old_element new_element (e.g. Cu Zn)", timeout=3)
            return
        if self._edit_running():
            return
        old_el, new_el = parts[0], parts[1]
        from pdb_wizard.constants import get_element
        new_elem = get_element(new_el)
        if new_elem.symbol == "X":
            self.notify(f"Unknown element: {new_el}", timeout=3)
            return
        self._save_undo()
        count = 0
        for atom in self.molecule.atoms:
            if atom.element.symbol == old_el:
                atom._element = new_elem
                atom.name = new_elem.symbol
                count += 1
        if count == 0:
            self._discard_last_undo()
            self.notify(f"No {old_el} atoms found", timeout=2)
            return
        # Bond radii changed with the element: re-detect bonds and rebuild
        # the geometry tables like every other mutating operation does.
        self._start_edit(
            self._refresh_molecule(f"Substituted {count} {old_el} -> {new_el}")
        )

    def _do_edit_h(self, value: str) -> None:
        if not value.strip():
            return
        try:
            parts = value.strip().split()
            element = parts[0].strip().capitalize()
            distance = float(parts[1])
            if not np.isfinite(distance) or distance <= 0:
                self.notify("The distance must be a positive number, e.g. O 1.0", timeout=3)
                return
            if self._edit_running():
                return
            self._start_edit(self._do_edit_h_async(element, distance))
        except (ValueError, IndexError):
            self.notify("Invalid input. Use: O 1.0, C 1.09, etc.", timeout=3)

    async def _do_edit_h_async(self, element: str, distance: float) -> None:
        mol = self.molecule
        self._save_undo()
        undo = self._undo_stack
        from pdb_wizard.geometry import adjust_h_distances
        self._show_progress(f"Editing {element}-H distances...")
        await asyncio.sleep(0)
        _, messages = await asyncio.to_thread(
            adjust_h_distances, mol, element, distance,
        )
        if not messages:
            self._discard_last_undo(undo)
            self._hide_progress()
            self.notify(f"No bonded {element}-H pairs found; nothing changed", timeout=4)
            return
        n = len(messages)
        await self._refresh_molecule(
            f"Set {n} {element}-H distance{'s' if n != 1 else ''} to {distance} A", mol
        )

    def _do_qeq_charges(self, confirmed: bool) -> None:
        if not confirmed or self._edit_running():
            return
        self._start_edit(self._do_qeq_async())

    async def _do_qeq_async(self) -> None:
        from pdb_wizard.charges import apply_qeq_charges
        mol = self.molecule
        self._save_undo()
        self._show_progress("Computing QEq charges...", total=100)
        await asyncio.sleep(0)

        def _progress(frac: float) -> None:
            self.call_from_thread(self._update_progress, int(frac * 100))

        try:
            await asyncio.to_thread(apply_qeq_charges, mol, progress_callback=_progress)
        except Exception as e:
            self._hide_progress()
            self.notify(f"QEq failed: {e}", timeout=5)
            return
        self._hide_progress()

        # Show summary
        from pdb_wizard.analyses import charge_report
        self.push_screen(InfoModal("QEq Charges", charge_report(mol)))

    # MSD and Density are now launched as full screens (MsdScreen, DensityScreen)

    def _show_density3d(self) -> None:
        """Prompt for element and compute 3D density map overlay."""
        elements = sorted(set(a.element.symbol for a in self.molecule.atoms))
        items = [(el, f"3D density: {el}") for el in elements]
        self.push_screen(
            MenuDropdown(items, offset=20),
            callback=self._do_density3d,
        )

    def _do_density3d(self, element: str) -> None:
        if not element:
            return
        self._track_task(self._compute_density3d(element))

    async def _compute_density3d(self, element: str) -> None:
        from pdb_wizard.density3d import compute_density_3d
        self._show_progress(f"Computing 3D density for {element}...", total=100)
        _prog = [0.0]
        def _cb(f):
            _prog[0] = f
        timer = self.set_interval(0.1, lambda: self._update_progress(int(_prog[0] * 100)))
        await asyncio.sleep(0)

        try:
            positions, densities, pbc = await asyncio.to_thread(
                compute_density_3d, self._frames, element, n_bins=15, progress_callback=_cb,
            )
        finally:
            timer.stop()
        self._hide_progress()

        if len(positions) == 0:
            self.notify(f"No {element} atoms found in trajectory", timeout=3)
            return

        view = self.query_one(MoleculeView)
        view._density_positions = positions
        view._density_values = densities
        view._density_element = element
        view._invalidate_cache()
        self.notify(f"3D density: {len(positions)} bins for {element} (toggle with View menu)", timeout=3)

    def _plot_isotherm(self, dirpath: str) -> None:
        if not dirpath.strip():
            return
        import os
        dirpath = dirpath.strip()
        if not os.path.isdir(dirpath):
            self.notify(f"Not a directory: {dirpath}", timeout=3)
            return
        from pdb_wizard.analyses import run_isotherm
        res = run_isotherm(dirpath)
        if res.empty:
            self.notify(res.summary, timeout=3)
            return
        pressures, uptakes = res.x, res.y

        class IsothermPlotScreen(_AnalysisScreen):
            _prefix = "iso"
            DEFAULT_CSS = _ANALYSIS_CSS.format(cls="IsothermPlotScreen") + """
            IsothermPlotScreen #iso-status { height: 1; color: $text-muted; }
            """
            def __init__(self, p, u, path, log_text=""):
                ModalScreen.__init__(self)
                self._molecule = None
                self._frames = None
                self._elements = []
                self._compute_task = None
                self._p, self._u, self._path = p, u, path
                self._log_text = log_text  # MPMC's own averages and unfinished runs
            def compose(self_inner) -> ComposeResult:
                with Vertical():
                    with Horizontal():
                        yield Button("CSV", id="iso-csv")
                        if self_inner._log_text:
                            yield Button("MPMC averages", id="iso-log")
                        yield Button("Close", id="iso-close")
                    yield PlotWidget(id="iso-plot")
                    yield Label("", id="iso-status")
            def on_mount(self_inner) -> None:
                plot = self_inner.query_one("#iso-plot", PlotWidget)
                y_max = float(self_inner._u.max()) * 1.1 if self_inner._u.max() > 0 else 1.0
                plot.set_data(self_inner._p, self_inner._u,
                              title="Adsorption Isotherm",
                              x_label="pressure (atm)", y_min=0.0, y_max=y_max)
                n = len(self_inner._p)
                self_inner.query_one("#iso-status", Label).update(
                    f"{n} pressure points from {self_inner._path}"
                )
            def on_button_pressed(self_inner, event: Button.Pressed) -> None:
                if event.button.id == "iso-close":
                    self_inner.dismiss(None)
                elif event.button.id == "iso-log":
                    self_inner.app.push_screen(
                        InfoModal("MPMC averages", self_inner._log_text))
                elif event.button.id == "iso-csv":
                    self_inner.app.push_screen(
                        FileSaveModal("Save isotherm CSV:", default="isotherm.csv"),
                        callback=lambda fp: self_inner._do_export_csv(
                            fp, "pressure_atm,avg_molecules", self_inner._p, self_inner._u),
                    )

        self.push_screen(IsothermPlotScreen(pressures, uptakes, dirpath, res.text))
        if "has not finished" in res.text:
            self.notify("Some pressure points have not finished; see MPMC averages.",
                        severity="warning", timeout=8)

    def _show_energy_plot(self) -> None:
        from pathlib import Path
        # Look for energy.dat in same directory as the loaded file
        base = Path(self.filepath).parent
        candidates = [base / "energy.dat", base / "energy.csv"]
        energy_file = None
        for c in candidates:
            if c.exists():
                energy_file = str(c)
                break
        if energy_file is None:
            # Ask user for file
            self.push_screen(
                FileSaveModal("Open energy file:", default=str(base / "energy.dat"), button_label="Open"),
                callback=self._load_energy_plot,
            )
        else:
            self._load_energy_plot(energy_file)

    def _load_energy_plot(self, filepath: str) -> None:
        if not filepath.strip():
            return
        from pdb_wizard.energy import read_energy_dat
        try:
            data = read_energy_dat(filepath.strip())
        except (OSError, ValueError, IndexError) as e:  # a folder, not text, ragged rows
            self.notify(f"Could not read {filepath}: {e}", timeout=5, severity="error")
            return
        if not data:
            self.notify(f"Could not read {filepath}", timeout=3)
            return
        if not [k for k in data if k != "step"]:
            self.notify(f"Nothing to plot in {filepath}: it has only a step column",
                        timeout=5, severity="warning")
            return
        self.push_screen(EnergyPlotScreen(data, filepath.strip()))

    def _save_undo(self, frame_cells: bool = False, mol=None, undo=None, redo=None) -> None:
        """Snapshot the molecule for Undo. ``frame_cells`` also keeps every trajectory
        frame's cell (for an edit that changes them all). ``mol``, ``undo`` and ``redo``
        name another tab's molecule and stacks (default: the active tab's)."""
        import copy
        snap = copy.deepcopy(self.molecule if mol is None else mol)
        if self._frames and (mol is None or mol is self.molecule):
            snap._undo_frame_idx = self._frame_idx
        if frame_cells and self._frames:
            snap._undo_frame_cells = [copy.deepcopy(f.pbc) for f in self._frames]
        undo = self._undo_stack if undo is None else undo
        undo.append(snap)
        if len(undo) > 10:
            undo.pop(0)
        (self._redo_stack if redo is None else redo).clear()

    def _swap_in_snapshot(self, source: list, target: list) -> None:
        """Make the top of ``source`` the molecule; push the current one on ``target``."""
        import copy
        snap = source.pop()
        idx = snap.__dict__.pop("_undo_frame_idx", None)
        if (idx is not None and self._frames and idx != self._frame_idx
                and 0 <= idx < len(self._frames)):
            # The snapshot is of another frame than the one on screen: go back to it, or
            # the frame on screen would be overwritten with that frame's atoms.
            self._frame_idx = idx
            self.molecule = self._frames[idx]
        current = copy.deepcopy(self.molecule)
        if self._frames:
            current._undo_frame_idx = self._frame_idx
        cells = snap.__dict__.pop("_undo_frame_cells", None)
        if cells is not None and self._frames and len(cells) == len(self._frames):
            current._undo_frame_cells = [copy.deepcopy(f.pbc) for f in self._frames]
            for frame, cell in zip(self._frames, cells):
                frame.pbc = cell
        target.append(current)
        self.molecule = snap
        if self._frames and 0 <= self._frame_idx < len(self._frames):
            # The shown frame is this molecule: without this the edit comes back as soon
            # as the user steps to another frame and returns.
            self._frames[self._frame_idx] = snap

    def _do_undo(self) -> None:
        if not self._undo_stack:
            self.notify("Nothing to undo", timeout=2)
            return
        if self._edit_running():
            return
        self._swap_in_snapshot(self._undo_stack, self._redo_stack)
        view = self.query_one(MoleculeView)
        view.set_molecule(self.molecule)
        self.query_one(GeometryPanel).set_molecule(self.molecule)
        if self._frames:
            self._update_title()
            self._setup_traj_bar()
        view._invalidate_cache()
        self.notify("Undone", timeout=1)

    def _do_redo(self) -> None:
        if not self._redo_stack:
            self.notify("Nothing to redo", timeout=2)
            return
        if self._edit_running():
            return
        self._swap_in_snapshot(self._redo_stack, self._undo_stack)
        view = self.query_one(MoleculeView)
        view.set_molecule(self.molecule)
        self.query_one(GeometryPanel).set_molecule(self.molecule)
        if self._frames:
            self._update_title()
            self._setup_traj_bar()
        view._invalidate_cache()
        self.notify("Redone", timeout=1)

    # Density profile is now launched as DensityScreen

    def _do_update_cell(self, value: str) -> None:
        if value is None or not value.strip():
            return
        text = value.strip().lower()
        if self._edit_running():
            return
        # A trajectory has one cell per frame; the new cell applies to all of them (as in
        # the classic menu), or the analyses over frames would use the old ones.
        others = [f for f in (self._frames or []) if f is not self.molecule]
        # Sentinel words to remove the box entirely (turn off PBC)
        if text in ("none", "off", "remove", "delete", "no", "0"):
            if self.molecule.pbc is None:
                self.notify("This system already has no PBC", timeout=2)
                return
            self._save_undo(frame_cells=True)
            self.molecule.pbc = None
            for frame in others:
                frame.pbc = None
            self._start_edit(self._refresh_molecule("PBC removed"))
            return
        try:
            parts = value.strip().split()
            a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
            alpha, beta, gamma = float(parts[3]), float(parts[4]), float(parts[5])
            from pdb_wizard.pbc import PBC, cell_problem
            problem = cell_problem(a, b, c, alpha, beta, gamma)
            if problem:
                self.notify(f"{problem}. The cell was not changed.", timeout=5,
                            severity="warning")
                return
            self._save_undo(frame_cells=True)
            if self.molecule.pbc is not None:
                self.molecule.pbc.update(a, b, c, alpha, beta, gamma)
            else:
                self.molecule.pbc = PBC(a, b, c, alpha, beta, gamma)
            for frame in others:
                frame.pbc = PBC(a, b, c, alpha, beta, gamma)
            self._start_edit(
                self._refresh_molecule(f"Cell: {a:.3f} {b:.3f} {c:.3f} {alpha:.1f} {beta:.1f} {gamma:.1f}")
            )
        except (ValueError, IndexError):
            self.notify(
                "Invalid input. Need: a b c alpha beta gamma  (or 'none' to remove)",
                timeout=4,
            )

    def _mpmc_step_keep_charges(self, keep: bool) -> None:
        if keep:
            self._mpmc_state["write_charges"] = True
            self._mpmc_step_ask_ff()
        else:
            # Existing charges stay until a new source actually succeeds, so
            # cancelling the next step doesn't leave the system with all-zero
            # charges. Both the file loader and QEq overwrite every framework atom.
            self._mpmc_step_charge_source()

    def _mpmc_step_charge_source(self) -> None:
        self.push_screen(
            ConfirmModal(
                "Charge Source",
                "How would you like to assign charges?\n\n"
                "Yes = Load from file (.resp or raw)\n"
                "No = Generate QEq charges automatically",
                yes_no=True,
            ),
            callback=self._mpmc_step_charge_choice,
        )

    def _mpmc_step_charge_choice(self, from_file: bool) -> None:
        if from_file:
            self._mpmc_step_charges_file_prompt()
        else:
            # Run QEq
            if self._edit_running():
                return
            self._mpmc_state["write_charges"] = True
            self._start_edit(self._mpmc_run_qeq())

    async def _mpmc_run_qeq(self) -> None:
        from pdb_wizard.charges import apply_qeq_charges
        mol = _mpmc_wizard_molecule(self)
        self._show_progress("Computing QEq charges...", total=100)
        await asyncio.sleep(0)

        def _progress(frac: float) -> None:
            self.call_from_thread(self._update_progress, int(frac * 100))

        try:
            await asyncio.to_thread(apply_qeq_charges, mol, progress_callback=_progress)
            self._hide_progress()
            total_q = sum(a.charge for a in mol.atoms)
            self.notify(f"QEq charges applied (total: {total_q:.4f} e)", timeout=2)
        except Exception as e:
            # No charges were made, so the export stops here rather than writing a file
            # whose charge column was never filled.
            self._hide_progress()
            self.notify(f"QEq failed: {e}. Nothing was written.", timeout=8, severity="error")
            return
        self._mpmc_step_ask_ff()

    def _mpmc_step_charges_file_prompt(self) -> None:
        self._mpmc_state["write_charges"] = True
        self.push_screen(
            ChargesFileModal(),
            callback=self._mpmc_step_charges_file,
        )

    def _mpmc_step_charges_file(self, result: dict) -> None:
        if not result:
            return  # cancelled
        mol = _mpmc_wizard_molecule(self)
        try:
            from pdb_wizard.io import assign_framework_charges, read_charges_file
            charges = read_charges_file(
                result["path"],
                skip_first=result.get("skip_first", 0),
                skip_last=result.get("skip_last", 0),
            )

            n = sum(1 for a in mol.atoms if a.mpmc_flag != "M")
            problem = assign_framework_charges(mol, charges)
            if problem:
                self.notify(problem, timeout=5)
                return
            if len(charges) != n:
                self.notify(
                    f"Applied {len(charges)} charges recursively "
                    f"({n // len(charges)}x)", timeout=3,
                )
            self.notify(f"Loaded {len(charges)} charges", timeout=2)
        except (OSError, ValueError) as e:
            self.notify(f"Error reading charges: {e}", timeout=5)
            return
        self._mpmc_step_ask_ff()

    def _mpmc_step_ask_ff(self) -> None:
        self.push_screen(
            ForceFieldModal(_mpmc_wizard_molecule(self)),
            callback=self._mpmc_step_ff_apply,
        )

    def _mpmc_step_ff_apply(self, ff_name: str | None) -> None:
        if ff_name is None:
            return  # Cancel: abort the MPMC export
        if not ff_name:
            self._mpmc_state["write_ff"] = False
        else:
            from pdb_wizard.forcefields import (
                apply_ff_to_system_report,
                get_forcefield,
                missing_ff_messages,
            )
            ff_map = {"OPLSAA": 0, "PHAHST": 1}
            ff_idx = ff_map.get(ff_name, 0)
            mol = _mpmc_wizard_molecule(self)
            _, missing = apply_ff_to_system_report(mol.atoms, get_forcefield(ff_idx))
            label = ["OPLS-AA/UFF", "PHAHST"][ff_idx]
            self.notify(f"Applied {label} force field", timeout=2)
            for msg in missing_ff_messages(missing):
                self.notify(msg.strip("! "), timeout=8, severity="warning")
        self._mpmc_step_sorbate()

    # --- Edit > Pack Sorbates: add movable sorbate molecules to the structure ---

    def _pack_step_model(self, choice: str | None) -> None:
        if not choice:
            return  # Cancel
        if choice == SorbateModal.FROM_FILE:
            self.push_screen(
                FileSaveModal("Sorbate template (MPMC .pqr):", default="", button_label="Open"),
                callback=self._pack_step_template,
            )
            return
        from pdb_wizard.sorbates import SORBATE_MODELS
        self._pack_ask_count(SORBATE_MODELS[choice])

    def _pack_step_template(self, path: str | None) -> None:
        if not path or not path.strip():
            return
        from pdb_wizard.sorbates import read_sorbate_template
        try:
            model = read_sorbate_template(path.strip())
        except (OSError, ValueError) as e:
            self.notify(f"Could not read the template: {e}", severity="error", timeout=6)
            return
        self._pack_ask_count(model)

    def _pack_ask_count(self, model) -> None:
        self.push_screen(
            InputModal(f"How many {model.label} molecules?", default="1",
                       hint="A number, or 'fill' to pack until nothing else fits. "
                            "Positions and orientations are random, with no overlaps "
                            "(minimum image in the real cell)."),
            callback=lambda value: self._pack_step_count(model, value),
        )

    def _pack_step_count(self, model, value: str | None) -> None:
        if not value or not value.strip():
            return  # Cancel / Escape
        from pdb_wizard.sorbates import parse_sorbate_count
        try:
            n = parse_sorbate_count(value)
        except ValueError as e:
            self.notify(str(e), severity="warning")
            self._pack_ask_count(model)
            return
        if self._edit_running():
            return
        self._start_edit(self._pack_sorbates_async(model, n))

    async def _pack_sorbates_async(self, model, n: int | None) -> None:
        from pdb_wizard.io import next_mpmc_mol_id
        from pdb_wizard.sorbates import insert_sorbates, sorbate_atoms
        mol = self.molecule
        undo, redo = self._undo_stack, self._redo_stack
        self._show_progress(f"Packing {model.label}...")
        await asyncio.sleep(0)
        placed = await asyncio.to_thread(insert_sorbates, mol, model, n)
        if not placed:
            self._hide_progress()
            self.notify("No room for even one molecule with these settings.",
                        severity="error", timeout=5)
            return
        self._save_undo(mol=mol, undo=undo, redo=redo)
        mol.atoms.extend(sorbate_atoms(model, placed, next_mpmc_mol_id(mol)))
        full = "" if n is None or len(placed) == n else f" (asked for {n}; the cell is full)"
        await self._refresh_molecule(
            f"Packed {len(placed)} {model.label}{full}. Save as MPMC PDB for MPMC.", mol)

    def _mpmc_step_sorbate(self) -> None:
        self.push_screen(
            SorbateModal(),
            callback=self._mpmc_step_sorbate_result,
        )

    def _mpmc_step_sorbate_result(self, model_name: str | None) -> None:
        if model_name is None:
            return  # Cancel: abort the MPMC export
        self._mpmc_state["sorbate"] = model_name if model_name else None
        self._mpmc_state["sorbate_n"] = 1
        if not model_name:
            self._mpmc_step_filename()
            return
        self.push_screen(
            InputModal("How many molecules?", default="1",
                       hint="A number, or 'fill' to pack until nothing else fits. "
                            "Positions and orientations are random, with no overlaps "
                            "(minimum image in the real cell)."),
            callback=self._mpmc_step_sorbate_count,
        )

    def _mpmc_step_sorbate_count(self, value: str | None) -> None:
        if not value or not value.strip():
            return  # Cancel / Escape (InputModal returns ""): abort the MPMC export
        from pdb_wizard.sorbates import parse_sorbate_count
        try:
            self._mpmc_state["sorbate_n"] = parse_sorbate_count(value)
        except ValueError as e:
            self.notify(str(e), severity="warning")
            self._mpmc_step_sorbate_result(self._mpmc_state["sorbate"])
            return
        self._mpmc_step_filename()

    def _mpmc_step_filename(self) -> None:
        filepath = getattr(self, "_mpmc_state", {}).get("filepath", self.filepath)
        stem = Path(filepath).stem
        default = str(Path(filepath).parent / f"{stem}_out_mpmc.pqr")
        self.push_screen(
            FileSaveModal("Save MPMC PDB as:", default=default),
            callback=self._mpmc_step_write,
        )

    def _mpmc_step_write(self, value: str) -> None:
        if not value.strip():
            return
        self._confirm_overwrite(value.strip(), self._write_mpmc_file)

    def _write_mpmc_file(self, filepath: str) -> None:

        from pdb_wizard.io import write_mpmc_pdb

        sorbate_lines = None
        sorbate_name = self._mpmc_state.get("sorbate")
        mol = _mpmc_wizard_molecule(self)
        if sorbate_name:
            from pdb_wizard.io import next_mpmc_mol_id
            from pdb_wizard.sorbates import format_inserted_sorbates, insert_sorbates
            n_want = self._mpmc_state.get("sorbate_n", 1)
            placed = insert_sorbates(mol, sorbate_name, n_want)
            if not placed:
                self.notify("No room for even one molecule; nothing written",
                            severity="error")
                return
            if n_want is None or len(placed) < n_want:
                self.notify(f"Placed {len(placed)} molecules (cell full)", timeout=5)
            sorbate_lines = format_inserted_sorbates(
                sorbate_name, placed, start_atom_id=len(mol.atoms) + 1,
                start_mol_id=next_mpmc_mol_id(mol),
            )

        self._write_text_file(
            filepath,
            lambda f: write_mpmc_pdb(
                mol, f,
                write_charges=self._mpmc_state["write_charges"],
                write_params=self._mpmc_state["write_ff"],
                sorbate_lines=sorbate_lines,
            ),
            "Writing MPMC PDB",
        )
