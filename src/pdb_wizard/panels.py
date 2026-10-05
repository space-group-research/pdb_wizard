"""TUI sidebar panels: OperationsPanel, GeometryPanel, and VisualPanel."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Checkbox,
    DataTable,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from pdb_wizard.molecule import Molecule

# ---------------------------------------------------------------------------
# OperationsPanel
# ---------------------------------------------------------------------------

# Mirrors the top-bar menus (File, Edit, Analysis) so users have one source
# of truth for every available tool. View-only display toggles live in the
# top-bar View menu and the VisualPanel; they are intentionally NOT in the
# Operations panel because they're not "run-once" actions.
OPERATIONS = [
    # ---- Analysis (mirrors top bar Analysis menu) ----
    ("info",           "System Info",              "analysis"),
    ("void_vol",       "Void Volume",              "analysis"),
    ("surface_area",   "Surface Area",             "analysis"),
    ("pore_size",      "Pore Size Distribution",   "analysis"),
    ("rdf",            "Radial Distribution",      "analysis"),
    ("coordination",   "Coordination Number",      "analysis"),
    ("hbonds",         "Hydrogen Bonds",           "analysis"),
    ("msd",            "Mean Sq. Displacement",    "analysis"),
    ("rmsd",           "RMSD vs Time",             "analysis"),
    ("gyration",       "Radius of Gyration",       "analysis"),
    ("density",        "Density Profile",          "analysis"),
    ("density3d",      "3D Density Map",           "analysis"),
    ("pxrd",           "Powder XRD",               "analysis"),
    ("energy_plot",    "Energy Plot",              "analysis"),
    ("input_generator","Input Generator",          "analysis"),
    ("isotherm_plot",  "Plot Isotherm Results",    "analysis"),
    ("db_search",      "Fetch from Database",      "analysis"),
    # ---- Edit (mirrors top bar Edit menu) ----
    ("undo",           "Undo",                     "modify"),
    ("wrap_center",    "Wrap (centered)",          "modify"),
    ("wrap_forward",   "Wrap (forward)",           "modify"),
    ("sort",           "Sort Atoms",               "modify"),
    ("extend",         "Extend Axis",              "modify"),
    ("del_lone",       "Delete Lone Atoms",        "modify"),
    ("edit_h",         "Edit H Distances",         "modify"),
    ("update_cell",    "Update Unit Cell",         "modify"),
    ("substitute",     "Substitute Element",       "modify"),
    ("reduce_cell",    "Reduce Supercell",         "modify"),
    ("qeq_charges",    "Generate QEq Charges",     "modify"),
    # ---- File (mirrors top bar File menu) ----
    ("open_file",      "Open File...",             "export"),
    ("write_xyz",      "Save as XYZ",              "export"),
    ("write_pdb",      "Save as PDB",              "export"),
    ("write_cif",      "Save as CIF",              "export"),
    ("write_poscar",   "Save as POSCAR",           "export"),
    ("write_lammps",   "Save as LAMMPS Data",      "export"),
    ("write_mpmc",     "Save as MPMC PDB",         "export"),
    ("frame_to_tab",   "Open Frame in New Tab",    "export"),
    ("export_png",     "Export PNG",               "export"),
    ("export_gif",     "Export Rotation GIF",      "export"),
    ("quit",           "Quit",                     "export"),
]


class OperationsPanel(Widget):
    BINDINGS = [
        Binding("enter", "run_selected", "Run", show=True),
    ]
    DEFAULT_CSS = """
    OperationsPanel {
        dock: left;
        width: 30;
        display: none;
        border-right: solid $accent;
    }
    OperationsPanel.visible {
        display: block;
    }
    OperationsPanel DataTable {
        height: 1fr;
    }
    """

    class RunCommand(Message):
        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    def __init__(self) -> None:
        super().__init__()
        self._populated = False

    def compose(self) -> ComposeResult:
        yield DataTable(id="ops-table", cursor_type="row")

    def on_mount(self) -> None:
        if not self._populated:
            self._populate()

    def _populate(self) -> None:
        table = self.query_one("#ops-table", DataTable)
        table.clear(columns=True)
        table.add_columns("", "Command")
        section = ""
        for cmd_id, label, group in OPERATIONS:
            if group != section:
                section = group
                header = {"analysis": "--- Analysis ---", "modify": "--- Edit ---", "export": "--- File ---"}
                table.add_row("", header.get(group, ""), key=f"_header_{group}")
            table.add_row(chr(0x25b6), label, key=cmd_id)
        self._populated = True

    def action_run_selected(self) -> None:
        table = self.query_one("#ops-table", DataTable)
        if table.row_count == 0:
            return
        row_keys = list(table.rows.keys())
        row = max(0, min(table.cursor_row, len(row_keys) - 1))
        key = row_keys[row].value
        if key and not key.startswith("_header_"):
            self.post_message(self.RunCommand(key))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value and not event.row_key.value.startswith("_header_"):
            self.post_message(self.RunCommand(event.row_key.value))


# ---------------------------------------------------------------------------
# GeometryPanel
# ---------------------------------------------------------------------------

class GeometryPanel(Widget):
    BINDINGS = [
        Binding("tab", "next_tab", "Tab", show=True),
        Binding("shift+tab", "prev_tab", "Prev tab", show=False),
        Binding("s", "toggle_sort", "Sort", show=True),
    ]

    _TAB_IDS = [
        "tab-bonds", "tab-angles", "tab-dihedrals",
        "tab-contacts", "tab-molecules", "tab-lone", "tab-coords",
    ]

    DEFAULT_CSS = """
    GeometryPanel {
        dock: right;
        width: 50;
        display: none;
        border-left: solid $accent;
    }
    GeometryPanel.visible {
        display: block;
    }
    GeometryPanel DataTable {
        height: 1fr;
    }
    """

    class HighlightAtoms(Message):
        def __init__(self, atom_indices: tuple[int, ...]) -> None:
            super().__init__()
            self.atom_indices = atom_indices

    class EditCoord(Message):
        def __init__(self, atom_index: int, x: float, y: float, z: float) -> None:
            super().__init__()
            self.atom_index = atom_index
            self.x = x
            self.y = y
            self.z = z

    def __init__(self) -> None:
        super().__init__()
        self._molecule: Molecule | None = None
        self._populating = False
        self._sort_ascending: dict[str, bool] = {t: False for t in self._TAB_IDS}

    def set_molecule(self, molecule: Molecule) -> None:
        self._molecule = molecule
        # Mark all tabs as needing refresh; compute lazily on demand
        self._populated_tabs: set[str] = set()
        if self.is_mounted and self.has_class("visible"):
            self._populate_active_tab()

    def on_mount(self) -> None:
        if not hasattr(self, "_populated_tabs"):
            self._populated_tabs = set()
        if self._molecule is not None and self.has_class("visible"):
            self._populate_active_tab()

    def on_show(self) -> None:
        """Populate the active tab when the panel becomes visible."""
        if self._molecule is not None:
            self._populate_active_tab()

    def _populate_active_tab(self) -> None:
        """Populate only the currently active tab's data."""
        if self._molecule is None:
            return
        try:
            tabs = self.query_one(TabbedContent)
            active = tabs.active
        except Exception:
            return
        self._populate_tab(active)

    def _populate_tab(self, tab_id: str, force: bool = False) -> None:
        """Populate a single tab's data on demand."""
        if self._molecule is None:
            return
        if not force and tab_id in self._populated_tabs:
            return
        self._populated_tabs.add(tab_id)
        self._populating = True
        mol = self._molecule
        if tab_id == "tab-bonds":
            self._populate_bonds(mol)
        elif tab_id == "tab-angles":
            self._populate_angles(mol)
        elif tab_id == "tab-dihedrals":
            self._populate_dihedrals(mol)
        elif tab_id == "tab-contacts":
            self._populate_contacts(mol)
        elif tab_id == "tab-molecules":
            self._populate_submols(mol)
        elif tab_id == "tab-lone":
            self._populate_lone(mol)
        elif tab_id == "tab-coords":
            self._populate_coords(mol)
        self._populating = False

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        current = tabs.active
        idx = self._TAB_IDS.index(current) if current in self._TAB_IDS else 0
        tabs.active = self._TAB_IDS[(idx + 1) % len(self._TAB_IDS)]

    def action_prev_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        current = tabs.active
        idx = self._TAB_IDS.index(current) if current in self._TAB_IDS else 0
        tabs.active = self._TAB_IDS[(idx - 1) % len(self._TAB_IDS)]

    def action_toggle_sort(self) -> None:
        if self._molecule is None:
            return
        tabs = self.query_one(TabbedContent)
        active = tabs.active
        selected_row_key = self._current_row_key_value(self._table_for_tab(active))
        asc = self._sort_ascending.get(active, False)
        self._sort_ascending[active] = not asc
        self._populate_tables({active: selected_row_key} if selected_row_key is not None else None)
        table = self._table_for_tab(active)
        table.focus()
        self._emit_current_highlight(table)

    def compose(self) -> ComposeResult:
        with TabbedContent("Bonds", "Angles", "Dihedrals", "Contacts", "Mols", "Lone", "Coords"):
            with TabPane("Bonds", id="tab-bonds"):
                yield DataTable(id="bonds-table", cursor_type="row")
            with TabPane("Angles", id="tab-angles"):
                yield DataTable(id="angles-table", cursor_type="row")
            with TabPane("Dihedrals", id="tab-dihedrals"):
                yield DataTable(id="dihedrals-table", cursor_type="row")
            with TabPane("Contacts", id="tab-contacts"):
                yield DataTable(id="contacts-table", cursor_type="row")
            with TabPane("Mols", id="tab-molecules"):
                yield DataTable(id="molecules-table", cursor_type="row")
            with TabPane("Lone", id="tab-lone"):
                yield DataTable(id="lone-table", cursor_type="row")
            with TabPane("Coords", id="tab-coords"):
                yield DataTable(id="coords-table", cursor_type="row")

    def _atom_label(self, idx: int) -> str:
        if self._molecule is None:
            return str(idx + 1)
        return f"{idx + 1}:{self._molecule.atoms[idx].element.symbol}"

    def _table_for_tab(self, tab_id: str) -> DataTable:
        table_id = {
            "tab-bonds": "#bonds-table",
            "tab-angles": "#angles-table",
            "tab-dihedrals": "#dihedrals-table",
            "tab-contacts": "#contacts-table",
            "tab-molecules": "#molecules-table",
            "tab-lone": "#lone-table",
            "tab-coords": "#coords-table",
        }.get(tab_id, "#bonds-table")
        return self.query_one(table_id, DataTable)

    def _current_row_key_value(self, dt: DataTable) -> str | None:
        if dt.row_count == 0:
            return None
        row_keys = list(dt.rows.keys())
        if not row_keys:
            return None
        row = max(0, min(dt.cursor_row, len(row_keys) - 1))
        return row_keys[row].value

    def _restore_cursor(self, dt: DataTable, row_key: str | None) -> None:
        if row_key is None or dt.row_count == 0:
            return
        row_keys = list(dt.rows.keys())
        for row, key in enumerate(row_keys):
            if key.value == row_key:
                dt.move_cursor(row=row, scroll=False)
                return

    def _populate_tables(self, selected_row_keys: dict[str, str | None] | None = None) -> None:
        """Force-repopulate all tabs (used by sort and explicit refresh)."""
        if self._molecule is None:
            return
        self._populated_tabs = set()
        for tid in self._TAB_IDS:
            self._populate_tab(tid, force=True)

    def _populate_bonds(self, mol) -> None:
        bonds = getattr(mol, '_cached_bond_lengths', None) or mol.get_bond_lengths()
        if self._sort_ascending.get("tab-bonds"):
            bonds.sort(key=lambda x: x[2])
        table = self.query_one("#bonds-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Length (\u00c5)")
        for i, j, dist in bonds:
            table.add_row(self._atom_label(i), self._atom_label(j), f"{dist:.4f}", key=f"{i}-{j}")

    def _populate_angles(self, mol) -> None:
        angles = getattr(mol, '_cached_angles', None) or mol.get_angles()
        if self._sort_ascending.get("tab-angles"):
            angles.sort(key=lambda x: x[3])
        table = self.query_one("#angles-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Vertex", "Atom 3", "Angle (\u00b0)")
        for i, j, k, angle in angles:
            table.add_row(
                self._atom_label(i), self._atom_label(j), self._atom_label(k),
                f"{angle:.3f}", key=f"{i}-{j}-{k}",
            )

    def _populate_dihedrals(self, mol) -> None:
        dihedrals = getattr(mol, '_cached_dihedrals', None) or mol.get_dihedrals()
        if self._sort_ascending.get("tab-dihedrals"):
            dihedrals.sort(key=lambda x: x[4])
        table = self.query_one("#dihedrals-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle (\u00b0)")
        for i, j, k, l_idx, angle in dihedrals:
            table.add_row(
                self._atom_label(i), self._atom_label(j),
                self._atom_label(k), self._atom_label(l_idx),
                f"{angle:.3f}", key=f"{i}-{j}-{k}-{l_idx}",
            )

    def _populate_contacts(self, mol) -> None:
        contacts = getattr(mol, '_cached_contacts', None)
        if contacts is None:
            from pdb_wizard.geometry import get_close_contacts
            from pdb_wizard.molecule import set_atom_ids
            set_atom_ids(mol.atoms)
            contacts = get_close_contacts(mol)
        table = self.query_one("#contacts-table", DataTable)
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

    def _populate_submols(self, mol) -> None:
        submols = getattr(mol, '_cached_molecules', None) or mol.find_molecules()
        table = self.query_one("#molecules-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Atoms", "Formula")
        from collections import Counter
        for mi, sm in enumerate(submols):
            sm.atoms.sort(key=lambda a: a.atomic_number, reverse=True)
            counts = Counter(a.element.symbol for a in sm.atoms)
            formula = "".join(f"{el}{n}" if n > 1 else el for el, n in counts.items())
            atom_indices = "-".join(str(mol.atoms.index(a)) for a in sm.atoms)
            table.add_row(str(mi + 1), str(len(sm.atoms)), formula, key=atom_indices)

    def _populate_lone(self, mol) -> None:
        lone = getattr(mol, '_cached_lone_atoms', None)
        if lone is None:
            from pdb_wizard.geometry import get_lone_atoms
            lone = get_lone_atoms(mol)
        table = self.query_one("#lone-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Element", "x", "y", "z")
        for a in lone:
            idx = mol.atoms.index(a)
            table.add_row(
                str(a.id), a.element.symbol,
                f"{a.x[0]:.4f}", f"{a.x[1]:.4f}", f"{a.x[2]:.4f}",
                key=str(idx),
            )

    def _populate_coords(self, mol) -> None:
        table = self.query_one("#coords-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "El", "x", "y", "z")
        for idx, a in enumerate(mol.atoms):
            table.add_row(
                str(a.id), a.element.symbol,
                f"{a.x[0]:.4f}", f"{a.x[1]:.4f}", f"{a.x[2]:.4f}",
                key=str(idx),
            )

    def _emit_current_highlight(self, dt: DataTable) -> None:
        if not self.has_class("visible") or dt.row_count == 0:
            return
        rk = list(dt.rows.keys())[dt.cursor_row]
        if rk.value is not None:
            indices = tuple(int(x) for x in rk.value.split("-"))
            self.post_message(self.HighlightAtoms(indices))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # Lazily populate the activated tab
        if event.pane.id and self._molecule is not None and self.has_class("visible"):
            self._populate_tab(event.pane.id)
        for dt in event.pane.query(DataTable):
            dt.focus()
            self._emit_current_highlight(dt)
            break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._populating or not self.has_class("visible"):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        indices = tuple(int(x) for x in event.row_key.value.split("-"))
        self.post_message(self.HighlightAtoms(indices))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "coords-table":
            return
        if event.row_key is None or event.row_key.value is None:
            return
        if self._molecule is None:
            return
        idx = int(event.row_key.value)
        a = self._molecule.atoms[idx]
        self.post_message(self.EditCoord(idx, float(a.x[0]), float(a.x[1]), float(a.x[2])))


# ---------------------------------------------------------------------------
# Slider widget
# ---------------------------------------------------------------------------

class _NavRadioSet(RadioSet):
    BINDINGS = [
        Binding("tab", "next_and_toggle", "Next", show=False),
        Binding("shift+tab", "prev_and_toggle", "Prev", show=False),
    ]

    def action_next_and_toggle(self) -> None:
        self.action_next_button()
        self.action_toggle_button()

    def action_prev_and_toggle(self) -> None:
        self.action_previous_button()
        self.action_toggle_button()


class Slider(Static, can_focus=True):
    BINDINGS = [
        Binding("tab", "increase", "Increase", show=False),
        Binding("shift+tab", "decrease", "Decrease", show=False),
    ]
    DEFAULT_CSS = """
    Slider {
        height: 1;
        width: 1fr;
        padding: 0 1;
    }
    Slider:focus {
        background: $accent 30%;
        text-style: bold;
    }
    """

    def __init__(
        self, label: str, value: float = 0.5,
        min_val: float = 0.0, max_val: float = 1.0,
        step: float = 0.05, decimals: int = 2, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals

    class Changed(Message):
        def __init__(self, slider: Slider, value: float) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

    def render(self) -> str:
        bar_width = 10
        frac = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        filled = int(frac * bar_width)
        bar = chr(0x2588) * filled + chr(0x2591) * (bar_width - filled)
        prefix = chr(0x25b8) + " " if self.has_focus else "  "
        arrows = " " + chr(0x25c0) + chr(0x25b6) if self.has_focus else ""
        return f"{prefix}{self.label}: {self.value:.{self.decimals}f} [{bar}]{arrows}"

    def _adjust(self, delta: float) -> None:
        new = max(self.min_val, min(self.max_val, self.value + delta))
        if new != self.value:
            self.value = new
            self.refresh()
            self.post_message(self.Changed(self, self.value))

    def action_decrease(self) -> None:
        self._adjust(-self.step)

    def action_increase(self) -> None:
        self._adjust(self.step)


# ---------------------------------------------------------------------------
# VisualPanel (MO isovalue removed)
# ---------------------------------------------------------------------------

class VisualPanel(Widget):
    DEFAULT_CSS = """
    VisualPanel {
        dock: right;
        width: 30;
        display: none;
        border-left: solid $accent;
        padding: 1;
    }
    VisualPanel.visible {
        display: block;
    }
    VisualPanel Label {
        margin-top: 1;
        text-style: bold;
    }
    VisualPanel RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    VisualPanel #visual-help {
        dock: bottom;
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    class StyleChanged(Message):
        def __init__(self, licorice: bool, vdw: bool, ribbon: bool = False) -> None:
            super().__init__()
            self.licorice = licorice
            self.vdw = vdw
            self.ribbon = ribbon

    class LightingChanged(Message):
        def __init__(self, ambient: float, diffuse: float, specular: float, shininess: float) -> None:
            super().__init__()
            self.ambient = ambient
            self.diffuse = diffuse
            self.specular = specular
            self.shininess = shininess

    class SizeChanged(Message):
        def __init__(self, atom_scale: float, bond_radius: float) -> None:
            super().__init__()
            self.atom_scale = atom_scale
            self.bond_radius = bond_radius

    def __init__(self) -> None:
        super().__init__()
        self._licorice = False
        self._vdw = False
        self._ribbon = False

    def set_state(
        self, *, licorice: bool, vdw: bool = False, ribbon: bool = False,
        ambient: float, diffuse: float, specular: float, shininess: float,
        atom_scale: float, bond_radius: float,
    ) -> None:
        self._licorice = licorice
        self._vdw = vdw
        self._ribbon = ribbon
        if self.is_mounted:
            self._sync_widgets(
                ambient=ambient, diffuse=diffuse, specular=specular,
                shininess=shininess, atom_scale=atom_scale, bond_radius=bond_radius,
            )

    def _sync_widgets(
        self, *, ambient: float, diffuse: float, specular: float,
        shininess: float, atom_scale: float, bond_radius: float,
    ) -> None:
        radio_set = self.query_one(_NavRadioSet)
        idx = 1 if self._licorice else (2 if self._vdw else (3 if self._ribbon else 0))
        radio_set.query(RadioButton)[idx].value = True
        self.query_one("#slider-atom-scale", Slider).value = atom_scale
        self.query_one("#slider-bond-radius", Slider).value = bond_radius
        self.query_one("#slider-ambient", Slider).value = ambient
        self.query_one("#slider-diffuse", Slider).value = diffuse
        self.query_one("#slider-specular", Slider).value = specular
        self.query_one("#slider-shininess", Slider).value = shininess
        self._update_visibility()
        self.refresh()

    class ThemeChanged(Message):
        def __init__(self, dark: bool) -> None:
            super().__init__()
            self.dark = dark

    class ToggleChanged(Message):
        def __init__(self, toggle: str, value: bool) -> None:
            super().__init__()
            self.toggle = toggle
            self.value = value

    class ColorModeChanged(Message):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    def compose(self) -> ComposeResult:
        yield Label("Theme")
        with RadioSet(id="theme-set"):
            yield RadioButton("Dark", value=True, id="radio-dark")
            yield RadioButton("Light", id="radio-light")
        yield Label("Display")
        yield Checkbox("Show bonds", value=True, id="chk-bonds")
        yield Checkbox("Atom numbers", value=False, id="chk-atomnums")
        yield Checkbox("Hide water", value=False, id="chk-hidewater")
        yield Checkbox("Polyhedra", value=False, id="chk-polyhedra")
        yield Label("Color by")
        yield Select(
            [
                ("Element", "element"),
                ("Charge", "charge"),
                ("Residue", "residue"),
                ("Chain", "chain"),
                ("Index", "index"),
            ],
            value="element",
            id="sel-colormode",
        )
        yield Label("Style")
        with _NavRadioSet():
            yield RadioButton("CPK", value=True, id="radio-cpk")
            yield RadioButton("Licorice", id="radio-licorice")
            yield RadioButton("VDW", id="radio-vdw")
            yield RadioButton("Ribbon", id="radio-ribbon")
        yield Label("Sizes", id="label-sizes")
        yield Slider("Atom scale", value=0.35, min_val=0.10, max_val=1.00, id="slider-atom-scale")
        yield Slider("Bond radius", value=0.08, min_val=0.02, max_val=0.30, step=0.02, id="slider-bond-radius")
        yield Label("Lighting")
        yield Slider("Ambient", value=0.35, min_val=0.0, max_val=1.0, id="slider-ambient")
        yield Slider("Diffuse", value=0.60, min_val=0.0, max_val=1.0, id="slider-diffuse")
        yield Slider("Specular", value=0.40, min_val=0.0, max_val=1.0, id="slider-specular")
        yield Slider("Shininess", value=32.0, min_val=1.0, max_val=128.0, step=4.0, id="slider-shininess")
        yield Static("n/p nav; (shift-)tab toggle", id="visual-help")

    def _update_visibility(self) -> None:
        self.query_one("#slider-atom-scale", Slider).display = not self._licorice and not self._vdw
        self.query_one("#slider-bond-radius", Slider).display = not self._vdw
        self.query_one("#label-sizes", Label).display = not self._vdw

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        cid = event.checkbox.id or ""
        if cid == "chk-bonds":
            self.post_message(self.ToggleChanged("bonds", event.value))
        elif cid == "chk-atomnums":
            self.post_message(self.ToggleChanged("atom_numbers", event.value))
        elif cid == "chk-hidewater":
            self.post_message(self.ToggleChanged("hide_water", event.value))
        elif cid == "chk-polyhedra":
            self.post_message(self.ToggleChanged("polyhedra", event.value))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sel-colormode" and event.value is not None:
            self.post_message(self.ColorModeChanged(str(event.value)))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.id in ("radio-dark", "radio-light"):
            self.post_message(self.ThemeChanged(event.pressed.id == "radio-dark"))
        else:
            self._licorice = event.pressed.id == "radio-licorice"
            self._vdw = event.pressed.id == "radio-vdw"
            self._ribbon = event.pressed.id == "radio-ribbon"
            self._update_visibility()
            self.post_message(self.StyleChanged(self._licorice, self._vdw, self._ribbon))

    def on_slider_changed(self, event: Slider.Changed) -> None:
        sid = event.slider.id or ""
        if sid.startswith("slider-atom") or sid.startswith("slider-bond"):
            self.post_message(
                self.SizeChanged(
                    atom_scale=self.query_one("#slider-atom-scale", Slider).value,
                    bond_radius=self.query_one("#slider-bond-radius", Slider).value,
                )
            )
        else:
            self.post_message(
                self.LightingChanged(
                    ambient=self.query_one("#slider-ambient", Slider).value,
                    diffuse=self.query_one("#slider-diffuse", Slider).value,
                    specular=self.query_one("#slider-specular", Slider).value,
                    shininess=self.query_one("#slider-shininess", Slider).value,
                )
            )
