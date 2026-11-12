"""Classic text menu: the analyses, edits, exports and generators the viewers have.

Everything here is reached through menu numbers that the original pdb_wizard never used
(11 and up in the main menus, 10 and up in the geometry menu), so input scripts written
for the original numbering keep working. The numerical work is done by the same shared
modules the terminal viewer and the desktop GUI call (``analyses``, ``inputgen``,
``database``, ...), so all three front ends report the same numbers.
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from pdb_wizard.analyses import (
    ALL_ATOMS,
    AnalysisResult,
    charge_report,
    elements_of,
    energy_columns,
    energy_n_steps,
    find_energy_file,
    run_coordination,
    run_density_3d,
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
    surface_area_text,
    system_info_text,
    unavailable,
    void_volume_text,
    write_csv,
    write_energy_csv,
)
from pdb_wizard.charges import apply_qeq_charges
from pdb_wizard.constants import get_element
from pdb_wizard.database import BACKENDS, modes_for, placeholder_for, search_backends
from pdb_wizard.energy import read_energy_dat
from pdb_wizard.formats import write_cif, write_lammps_data, write_poscar
from pdb_wizard.inputgen import (
    check_engine,
    default_engine_outdir,
    generate_engine_inputs,
    parse_form_value,
)
from pdb_wizard.io import next_mpmc_mol_id
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.operations import sort_system
from pdb_wizard.reduce_cell import identify_framework_and_sorbates, reduce_to_primitive
from pdb_wizard.renderer import render_scene, rotation_matrix
from pdb_wizard.sorbates import (
    SORBATE_MODELS,
    insert_sorbates,
    parse_sorbate_count,
    read_sorbate_template,
    sorbate_atoms,
)
from pdb_wizard.templates import ENGINES

# Where each viewer feature (the terminal viewer's Operations-panel keys) lives in the
# classic menu. tests/test_classic_parity.py fails if a viewer feature is missing here.
CLASSIC_FEATURES: Dict[str, str] = {
    "info": "main 11 > 1",
    "void_vol": "main 11 > 2",
    "surface_area": "main 11 > 3",
    "pore_size": "main 11 > 4",
    "rdf": "main 11 > 5",
    "coordination": "main 11 > 6",
    "hbonds": "main 11 > 7",
    "msd": "main 11 > 8",
    "rmsd": "main 11 > 9",
    "gyration": "main 11 > 10",
    "density": "main 11 > 11",
    "density3d": "main 11 > 12",
    "pxrd": "main 11 > 13",
    "energy_plot": "main 11 > 14",
    "input_generator": "main 14",
    "isotherm_plot": "main 15",
    "db_search": "main 16",
    "undo": "main 17",
    "wrap_center": "main 4",
    "wrap_forward": "main 3",
    "sort": "main 12 > 1",
    "extend": "main 2",
    "del_lone": "main 1 > 5",
    "edit_h": "main 1 > 7",
    "update_cell": "main 5",
    "substitute": "main 12 > 2",
    "reduce_cell": "main 12 > 3",
    "pack_sorbates": "main 12 > 4",
    "qeq_charges": "main 12 > 5",
    "open_file": "main 18",
    "write_xyz": "main 6",
    "write_pdb": "main 8",
    "write_cif": "main 13 > 1",
    "write_poscar": "main 13 > 2",
    "write_lammps": "main 13 > 3",
    "write_mpmc": "main 7",
    "export_png": "main 13 > 4",
    "export_gif": "main 13 > 5",
    "quit": "main 9",
}
# Viewer features with no meaning in a text menu.
CLASSIC_NOT_APPLICABLE = frozenset({"frame_to_tab"})

CM_ANALYSES = [
    ("info", "system info"),
    ("void_vol", "void volume"),
    ("surface_area", "surface area"),
    ("pore_size", "pore size distribution"),
    ("rdf", "radial distribution function"),
    ("coordination", "coordination number"),
    ("hbonds", "hydrogen bonds"),
    ("msd", "mean squared displacement"),
    ("rmsd", "RMSD vs time"),
    ("gyration", "radius of gyration"),
    ("density", "density profile"),
    ("density3d", "3D density map"),
    ("pxrd", "powder XRD"),
    ("energy_plot", "energy plot"),
]

_CM_QUIT = ("q", "quit")
CM_UNDO_MAX_ATOMS = 300_000  # atoms (over all frames) above which undo is not recorded


class _CmCancel(Exception):
    """The user typed q at a prompt."""


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

def _cm_ask(label: str, default: object = None) -> str:
    """One line of input; blank returns the default, q cancels the whole action."""
    hint = f" [{default}]" if default not in (None, "") else ""
    raw = input(f"{label}{hint} > ").strip()
    if raw.lower() in _CM_QUIT:
        raise _CmCancel
    return raw if raw else ("" if default is None else str(default))


def _cm_ask_number(label: str, default: float, cast: Callable = float, low: float = None):
    while True:
        raw = _cm_ask(label, default)
        try:
            value = cast(raw)
            if low is not None and value < low:
                raise ValueError
            return value
        except ValueError:
            print(f"!!! Error: enter a number{'' if low is None else f' of at least {low:g}'} !!!")


def _cm_ask_choice(label: str, choices: List[str], default: str) -> str:
    """One of ``choices`` (case-insensitive)."""
    lookup = {c.lower(): c for c in choices}
    while True:
        raw = _cm_ask(f"{label} ({', '.join(choices)})", default)
        if raw.lower() in lookup:
            return lookup[raw.lower()]
        print(f"!!! Error: choose one of {', '.join(choices)} !!!")


def _cm_ask_yes(label: str, default: bool = False) -> bool:
    while True:
        raw = _cm_ask(f"{label} (y/n)", "y" if default else "n").lower()
        if raw in ("yes", "y", "1"):
            return True
        if raw in ("no", "n", "0"):
            return False
        print("!!! Error reading input !!!")


def _cm_ask_option(prompt: str) -> int:
    """A menu number; 0 for anything that is not a number."""
    try:
        return int(input(prompt))
    except ValueError:
        print("!!! Error converting input to int !!!")
        return 0


def _cm_ask_element(label: str, mol: Molecule, default: str, allow_all: bool = False) -> str:
    elements = elements_of(mol)
    while True:
        raw = _cm_ask(f"{label} ({', '.join(elements)}"
                      f"{', or all' if allow_all else ''})", default)
        if allow_all and raw.lower() in ("all", ALL_ATOMS):
            return ALL_ATOMS
        symbol = get_element(raw).symbol
        if symbol in elements:
            return symbol
        print(f"!!! No {raw} atoms in this structure !!!")


def _cm_ask_name(label: str = "Atom name filter (blank = all)") -> str:
    return _cm_ask(label) or ALL_ATOMS


def _cm_ask_range(frames: Optional[list]) -> tuple:
    """(start, end) frames, 1-based inclusive; (1, None) without a trajectory."""
    if not frames or len(frames) < 2:
        return 1, None
    n = len(frames)
    start = _cm_ask_number("First frame", 1, int, 1)
    end = _cm_ask_number("Last frame", n, int, 1)
    return start, end


# ----------------------------------------------------------------------
# Showing results
# ----------------------------------------------------------------------

def cm_ascii_plot(x, y, width: int = 64, height: int = 12) -> str:
    """A small text plot of y against x (finite points only)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return ""
    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0
    grid = [[" "] * width for _ in range(height)]
    cols = np.minimum(((x - x_lo) / x_span * (width - 1)).round().astype(int), width - 1)
    rows = np.minimum(((y - y_lo) / y_span * (height - 1)).round().astype(int), height - 1)
    for c, r in zip(cols, rows):
        grid[height - 1 - r][c] = "*"
    lines = []
    for i, row in enumerate(grid):
        label = f"{y_hi:>10.4g}" if i == 0 else f"{y_lo:>10.4g}" if i == height - 1 else " " * 10
        lines.append(f"{label} |{''.join(row).rstrip()}")
    lines.append(" " * 11 + "+" + "-" * width)
    lines.append(" " * 12 + f"{x_lo:<.4g}".ljust(width - 10) + f"{x_hi:>10.4g}")
    return "\n".join(lines)


def _cm_show(res: AnalysisResult) -> None:
    """Print a result the way the viewers show it: title, curve, summary, then offer CSV."""
    if res.empty:
        print(f"\n{res.summary}\n")
        return
    print(f"\n{res.title}\n")
    if len(res.x) > 1:
        print(cm_ascii_plot(res.x, res.y))
        print(" " * 12 + res.x_label)
    print(f"\n{res.summary}")
    if res.text:
        print(f"\n{res.text}")
    path = _cm_ask("\nCSV filename (blank to skip)")
    if path:
        try:
            write_csv(path, res.csv_header, *res.csv_arrays())
            print(f"wrote {path}")
        except OSError as e:
            print(f"!!! Error: {e} !!!")


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def _cm_run_analysis(key: str, mol: Molecule, frames: Optional[list], filename: str) -> None:
    problem = unavailable(key, mol, frames)
    if problem:
        print(f"\n{problem}\n")
        return
    traj = bool(frames and len(frames) > 1)
    elements = elements_of(mol) if mol is not None and mol.atoms else []
    first = elements[0] if elements else ""
    if key == "info":
        print("\n" + system_info_text(mol, filename) + "\n")
    elif key == "void_vol":
        print("\n" + void_volume_text(mol) + "\n")
    elif key == "surface_area":
        print("\n" + surface_area_text(mol) + "\n")
    elif key == "pore_size":
        _cm_show(run_pore_size(mol))
    elif key == "rdf":
        el1 = _cm_ask_element("First element", mol, first)
        el2 = _cm_ask_element("Second element", mol, elements[1] if len(elements) > 1 else first)
        name1 = _cm_ask_name("First atom name filter (blank = all)")
        name2 = _cm_ask_name("Second atom name filter (blank = all)")
        r_max = _cm_ask_number("Maximum r (A)", 13.0, float, 0.1)
        n_bins = _cm_ask_number("Number of bins", 100, int, 1)
        start, end = _cm_ask_range(frames)
        _cm_show(run_rdf(mol, el1, el2, name1, name2, r_max, n_bins, frames, start, end))
    elif key == "coordination":
        center = _cm_ask_element("Centre element", mol, first)
        neighbor = _cm_ask_element("Neighbour element", mol,
                                   elements[1] if len(elements) > 1 else first)
        r_max = _cm_ask_number("Maximum r (A)", 5.0, float, 0.1)
        use_traj = traj and _cm_ask_yes("Average over the trajectory?")
        start, end = _cm_ask_range(frames) if use_traj else (1, None)
        _cm_show(run_coordination(mol, center, neighbor, r_max, frames, start, end, use_traj))
    elif key == "hbonds":
        cutoff = _cm_ask_number("Donor-acceptor cutoff (A)", 3.5, float, 0.1)
        angle = _cm_ask_number("Minimum D-H...A angle (deg)", 120.0, float, 0.0)
        start, end = _cm_ask_range(frames)
        _cm_show(run_hbonds(mol, cutoff, angle, frames, start, end))
    elif key == "msd":
        element = _cm_ask_element("Element", mol, first)
        name = _cm_ask_name()
        dt = _cm_ask_number("Time between frames (ps)", 1.0, float, 0.0)
        start, end = _cm_ask_range(frames)
        _cm_show(run_msd(frames, element, name, dt, start, end))
    elif key == "rmsd":
        element = _cm_ask_element("Element", mol, "all", allow_all=True)
        name = _cm_ask_name()
        dt = _cm_ask_number("Time between frames (ps)", 1.0, float, 0.0)
        start, end = _cm_ask_range(frames)
        _cm_show(run_rmsd(frames, element, name, dt, start, end))
    elif key == "gyration":
        element = _cm_ask_element("Element", mol, "all", allow_all=True)
        dt = _cm_ask_number("Time between frames (ps)", 1.0, float, 0.0)
        start, end = _cm_ask_range(frames)
        _cm_show(run_gyration(frames, element, dt, start, end))
    elif key == "density":
        element = _cm_ask_element("Element", mol, first)
        name = _cm_ask_name()
        axis = "abc".index(_cm_ask_choice("Axis", ["a", "b", "c"], "c"))
        start, end = _cm_ask_range(frames)
        _cm_show(run_density_profile(mol, element, name, axis, frames, start, end))
    elif key == "density3d":
        element = _cm_ask_element("Element", mol, first)
        n_bins = _cm_ask_number("Bins per axis", 15, int, 1)
        res = run_density_3d(frames, element, n_bins)
        print(f"\n{res.summary}")
        if res.extra:
            path = _cm_ask("\nCSV filename for the grid (blank to skip)")
            if path:
                pos = res.extra["positions"]
                write_csv(path, "x_A,y_A,z_A,density", pos[:, 0], pos[:, 1], pos[:, 2],
                          res.extra["values"])
                print(f"wrote {path}")
    elif key == "pxrd":
        wavelength = _cm_ask_number("Wavelength (A; Cu Ka 1.5406, Mo Ka 0.7107, Co Ka 1.7903)",
                                    1.5406, float, 0.01)
        tt_max = _cm_ask_number("Maximum 2-theta (deg)", 50.0, float, 5.1)
        width = _cm_ask_number("Peak width (deg)", 0.10, float, 0.001)
        _cm_show(run_pxrd(mol, wavelength, tt_max, width))
    elif key == "energy_plot":
        _cm_energy(filename)


def _cm_energy(filename: str) -> None:
    path = _cm_ask("MPMC energy file", find_energy_file(filename))
    data = read_energy_dat(path) if path else {}
    if not data:
        print(f"\nNo energy data in {path or 'that file'}\n")
        return
    columns, default = energy_columns(data)
    column = _cm_ask_choice("Column", columns, default)
    n = energy_n_steps(data)
    start = _cm_ask_number("First step", 1, int, 1)
    end = _cm_ask_number("Last step", n, int, 1)
    res = run_energy(data, column, start, end)
    if res.empty:
        print(f"\n{res.summary}\n")
        return
    print(f"\n{res.title}\n")
    print(cm_ascii_plot(res.x, res.y))
    print(" " * 12 + res.x_label)
    print(f"\n{res.summary}")
    out = _cm_ask("\nCSV filename (blank to skip)")
    if out:
        write_energy_csv(data, out, res.extra["slice"])
        print(f"wrote {out}")


def cm_menu_analysis(mol: Molecule, frames: Optional[list], filename: str) -> None:
    """The Analysis menu. ``frames`` is the whole trajectory, or None for one structure."""
    listing = "".join(f"{i} = {label}\n" for i, (_, label) in enumerate(CM_ANALYSES, 1))
    back = len(CM_ANALYSES) + 1
    while True:
        option = _cm_ask_option(f"\nWhich analysis?\n\n{listing}{back} = back to main menu\n\n> ")
        if option == back:
            return
        if 1 <= option <= len(CM_ANALYSES):
            try:
                _cm_run_analysis(CM_ANALYSES[option - 1][0], mol, frames, filename)
            except _CmCancel:
                print("\nCancelled")
            except (OSError, ValueError) as e:
                print(f"!!! Error: {e} !!!")
        elif option:
            print("\nInvalid option!")


def cm_menu_isotherm() -> None:
    try:
        directory = _cm_ask("Isotherm folder (the one holding the pressure folders)", ".")
        _cm_show(run_isotherm(directory))
    except _CmCancel:
        print("\nCancelled")
    except (OSError, ValueError) as e:
        print(f"!!! Error: {e} !!!")


# ----------------------------------------------------------------------
# Edit
# ----------------------------------------------------------------------

def cm_prompt_sorbate_model():
    """Pick a built-in sorbate model or an MPMC .pqr template; None if cancelled."""
    names = list(SORBATE_MODELS)
    print("\nSorbate models:")
    for i, name in enumerate(names):
        print(f"{i:>3} = {name:<14} {SORBATE_MODELS[name].description}")
    while True:
        choice = input("\nModel number or name, or a path to an MPMC .pqr template "
                       "(q to cancel)\n\n> ").strip()
        if choice.lower() in _CM_QUIT:
            return None
        try:
            if choice.isdigit() and int(choice) < len(names):
                return SORBATE_MODELS[names[int(choice)]]
            if choice in SORBATE_MODELS:
                return SORBATE_MODELS[choice]
            return read_sorbate_template(choice)
        except (OSError, ValueError) as e:
            print(f"!!! {e} !!!")


def cm_prompt_sorbate_count() -> Optional[int]:
    """How many molecules to place; None means fill the cell."""
    while True:
        raw = input("\nHow many molecules? (a number, or 'fill' to pack until "
                    "nothing else fits)\n\n> ").strip().lower()
        try:
            return parse_sorbate_count(raw)
        except ValueError:
            print("!!! Enter a positive number or 'fill' !!!")


def cm_place_sorbates(mol: Molecule, model, n: Optional[int]) -> list:
    placed = insert_sorbates(mol, model, n,
                             progress_callback=lambda k: print(f"\rplaced {k}", end=""))
    print(f"\nPlaced {len(placed)} {model.label}"
          + (f" (asked for {n}; the cell is full)" if n is not None and len(placed) < n else ""))
    return placed


def _cm_sort(mol: Molecule) -> Molecule:
    if mol.pbc is None:
        print("\nSorting atoms into molecules needs a unit cell")
        return mol
    sort_system(mol, False)
    print("\nSorted atoms")
    return mol


def _cm_substitute(mol: Molecule) -> Molecule:
    parts = _cm_ask("Substitute which element with which? (old new, e.g. Cu Zn)").split()
    if len(parts) != 2:
        print("!!! Enter: old_element new_element (e.g. Cu Zn) !!!")
        return mol
    old, new = parts
    new_el = get_element(new)
    if new_el.symbol == "X":
        print(f"!!! Unknown element: {new} !!!")
        return mol
    count = 0
    for atom in mol.atoms:
        if atom.element.symbol == old:
            atom._element = new_el
            atom.name = new_el.symbol
            count += 1
    if not count:
        print(f"\nNo {old} atoms found")
        return mol
    mol.bonds = []
    mol.detect_bonds()
    print(f"\nReplaced {count} {old} atom{'s' if count != 1 else ''} with {new_el.symbol}")
    return mol


def _cm_reduce(mol: Molecule) -> Molecule:
    if mol.pbc is None:
        print("\nNeed PBC for supercell reduction")
        return mol
    print("\nHow many unit cells does the supercell hold along each axis?")
    na = _cm_ask_number("a", 2, int, 1)
    nb = _cm_ask_number("b", 2, int, 1)
    nc = _cm_ask_number("c", 2, int, 1)
    framework, sorbates = identify_framework_and_sorbates(mol, na, nb, nc)
    print(f"\nFramework: {len(framework)} unique sites ({len(framework) * na * nb * nc} total "
          f"atoms)  |  Sorbate molecules: {len(sorbates)}")
    keep = list(range(len(sorbates)))
    if sorbates:
        print("\n  #  atoms  formula      centre")
        for i, indices in enumerate(sorbates, 1):
            counts: Dict[str, int] = {}
            for idx in indices:
                symbol = mol.atoms[idx].element.symbol
                counts[symbol] = counts.get(symbol, 0) + 1
            centre = np.mean([mol.atoms[idx].x for idx in indices], axis=0)
            formula = " ".join(f"{el}{n}" for el, n in sorted(counts.items()))
            print(f"{i:>3}  {len(indices):>5}  {formula:<12} "
                  f"{centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}")
        while True:
            raw = _cm_ask("\nSorbate molecules to keep (all, none, or numbers like 1,3,4)",
                          "all").lower()
            try:
                if raw == "all":
                    break
                keep = [] if raw == "none" else sorted(
                    {int(p) - 1 for p in raw.replace(",", " ").split()})
                if any(k < 0 or k >= len(sorbates) for k in keep):
                    raise ValueError
                break
            except ValueError:
                print(f"!!! Enter all, none, or numbers from 1 to {len(sorbates)} !!!")
    result = reduce_to_primitive(mol, na, nb, nc, keep)
    print(f"\nReduced to primitive cell: {len(result.atoms)} atoms, "
          f"kept {len(keep)}/{len(sorbates)} sorbate molecules")
    return result


def _cm_pack(mol: Molecule) -> Molecule:
    if mol.pbc is None:
        print("\n!!! Packing sorbates needs a unit cell (update cell dimensions first) !!!")
        return mol
    model = cm_prompt_sorbate_model()
    if model is None:
        return mol
    placed = cm_place_sorbates(mol, model, cm_prompt_sorbate_count())
    if not placed:
        print("No room for even one molecule.")
        return mol
    mol.atoms.extend(sorbate_atoms(model, placed, next_mpmc_mol_id(mol)))
    set_atom_ids(mol.atoms)
    mol.bonds = []
    print("Added them to the structure. Write an MPMC .pdb (main menu 7) to use them in MPMC.")
    return mol


def _cm_qeq(mol: Molecule) -> Molecule:
    if not mol.atoms:
        print("\nNo atoms loaded")
        return mol
    print("\nComputing QEq charges...")
    try:
        apply_qeq_charges(mol)
    except Exception as e:  # too many atoms, singular matrix, ...
        print(f"!!! QEq failed: {e} !!!")
        return mol
    print("\n" + charge_report(mol) + "\n")
    return mol


def cm_menu_edit(mol: Molecule) -> Molecule:
    """The Edit menu. Returns the (possibly new) structure."""
    actions = {1: _cm_sort, 2: _cm_substitute, 3: _cm_reduce, 4: _cm_pack, 5: _cm_qeq}
    while True:
        option = _cm_ask_option(
            "\nWhat would you like to do?\n\n"
            "1 = sort atoms into molecules\n"
            "2 = substitute an element\n"
            "3 = reduce a supercell to one unit cell\n"
            "4 = pack sorbates into the structure\n"
            "5 = generate QEq charges\n"
            "6 = back to main menu\n\n> "
        )
        if option == 6:
            return mol
        if option in actions:
            try:
                mol = actions[option](mol)
            except _CmCancel:
                print("\nCancelled")
        elif option:
            print("\nInvalid option!")


# ----------------------------------------------------------------------
# Undo
# ----------------------------------------------------------------------

def cm_signature(mol: Molecule) -> tuple:
    """Everything an edit can change, for telling whether an action did anything."""
    pbc = mol.pbc
    cell = None if pbc is None else (pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma)
    return (cell, tuple((a.element.symbol, a.name, float(a.x[0]), float(a.x[1]), float(a.x[2]),
                         a.charge) for a in mol.atoms))


class CmHistory:
    """Undo for the classic menu: ``before`` an action, ``after`` it, ``undo`` to go back."""

    def __init__(self, depth: int = 20):
        self._depth = depth
        self._stack: list = []
        self._pending = None

    def before(self, state) -> None:
        # Copying a very large trajectory would stall the menu, so those are not recorded.
        frames = state if isinstance(state, list) else [state]
        if sum(len(m.atoms) for m in frames) > CM_UNDO_MAX_ATOMS:
            self._pending = None
            return
        self._pending = (copy.deepcopy(state), self._sig(state))

    def after(self, state) -> None:
        if self._pending is not None and self._pending[1] != self._sig(state):
            self._stack.append(self._pending[0])
            del self._stack[:-self._depth]
        self._pending = None

    def undo(self):
        """The previous state, or None when there is nothing to undo."""
        return self._stack.pop() if self._stack else None

    @staticmethod
    def _sig(state) -> tuple:
        if isinstance(state, list):
            return tuple(cm_signature(m) for m in state)
        return cm_signature(state)


# ----------------------------------------------------------------------
# Measuring
# ----------------------------------------------------------------------

def cm_measure(mol: Molecule, ids: List[int]) -> str:
    """Distance (2 atoms), angle (3) or dihedral (4) between 1-based atom numbers,
    using the minimum image when there is a cell."""
    if len(ids) not in (2, 3, 4) or len(set(ids)) != len(ids):
        raise ValueError("Enter 2, 3 or 4 different atom numbers")
    if any(i < 1 or i > len(mol.atoms) for i in ids):
        raise ValueError(f"Atom numbers run from 1 to {len(mol.atoms)}")
    atoms = [mol.atoms[i - 1] for i in ids]
    label = "-".join(f"{a.element.symbol}{i}" for a, i in zip(atoms, ids))

    def vec(a, b) -> np.ndarray:
        d = b.x - a.x
        return mol.pbc.wrap(d) if mol.pbc is not None else d

    if len(ids) == 2:
        return f"{label}   r = {np.linalg.norm(vec(atoms[0], atoms[1])):.4f} A"
    if len(ids) == 3:
        v1, v2 = vec(atoms[1], atoms[0]), vec(atoms[1], atoms[2])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return f"{label}   angle = {np.degrees(np.arccos(np.clip(cos, -1, 1))):.2f} deg"
    # unsigned (0-180), as Molecule.get_dihedrals: the dihedrals list and the GUI agree
    b1, b2, b3 = (vec(atoms[0], atoms[1]), vec(atoms[1], atoms[2]), vec(atoms[2], atoms[3]))
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    cos_d = np.dot(n1, n2) / ((np.linalg.norm(n1) + 1e-10) * (np.linalg.norm(n2) + 1e-10))
    angle = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    return f"{label}   dihedral = {angle:.2f} deg"


def cm_menu_measure(mol: Molecule) -> None:
    while True:
        raw = input("\nAtom numbers: 2 for a distance, 3 for an angle, 4 for a dihedral "
                    "(q to go back)\n\n> ").strip()
        if raw.lower() in _CM_QUIT or not raw:
            return
        try:
            print("\n" + cm_measure(mol, [int(p) for p in raw.replace(",", " ").split()]))
        except ValueError as e:
            message = str(e)
            print(f"!!! {message if 'atom' in message.lower() else 'Enter atom numbers'} !!!")


def cm_dihedrals_list(mol: Molecule) -> List[str]:
    if not mol.bonds:
        mol.detect_bonds()
    atoms = mol.atoms
    return [
        f"{'-'.join(atoms[n].element.symbol for n in (i, j, k, m)):<11} "
        f"{i + 1:>5} {j + 1:>5} {k + 1:>5} {m + 1:>5}   dihedral = {round(angle, 4)}"
        for i, j, k, m, angle in mol.get_dihedrals()
    ]


# ----------------------------------------------------------------------
# Other file formats and pictures
# ----------------------------------------------------------------------

def _cm_render(mol: Molecule, width: int, height: int, turn: float = 0.0) -> np.ndarray:
    """The viewers' default view, turned about the vertical axis by ``turn`` radians."""
    if not mol.bonds:
        mol.detect_bonds()
    rot = rotation_matrix(0, turn, 0) @ rotation_matrix(-0.2, -0.5, 0.0)
    pixels, _ = render_scene(width, height, mol, rot, 4.0, bg_color=(0, 0, 0), pbc=mol.pbc,
                             ssaa=2, ambient=0.31, diffuse=0.72, specular=0.42, shininess=96.0)
    return pixels


def _cm_write_picture(mol: Molecule, path: str, animated: bool) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("!!! Pillow required: pip install Pillow !!!")
        return
    if not animated:
        Image.fromarray(_cm_render(mol, 1600, 1200)).save(path)
        print(f"wrote {path}")
        return
    n = 36
    images = []
    for i in range(n):
        images.append(Image.fromarray(_cm_render(mol, 800, 600, 2 * np.pi * i / n)))
        print(f"\rrendered {i + 1}/{n}", end="")
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0)
    print(f"\nwrote {path} ({n} frames)")


def cm_menu_write_other(mol: Molecule, filename: str) -> None:
    stem = os.path.splitext(filename)[0]
    writers = {1: (write_cif, ".cif"), 2: (write_poscar, ".vasp"), 3: (write_lammps_data, ".data")}
    while True:
        option = _cm_ask_option(
            "\nWhat would you like to write?\n\n"
            "1 = .cif\n"
            "2 = VASP POSCAR\n"
            "3 = LAMMPS data file\n"
            "4 = PNG picture\n"
            "5 = rotating GIF\n"
            "6 = back to main menu\n\n> "
        )
        if option == 6:
            return
        if option not in (1, 2, 3, 4, 5):
            if option:
                print("\nInvalid option!")
            continue
        if not mol.atoms:
            print("\nNo atoms loaded")
            continue
        try:
            if option in writers:
                writer, _ = writers[option]
                path = input("\noutput filename > ").strip()
                with open(path, "w") as out:
                    writer(mol, out)
                print(f"wrote {path}")
            else:
                path = _cm_ask("\noutput filename", stem + (".png" if option == 4 else ".gif"))
                _cm_write_picture(mol, path, animated=option == 5)
        except _CmCancel:
            print("\nCancelled")
        except (OSError, ValueError) as e:
            print(f"!!! Error: {e} !!!")


# ----------------------------------------------------------------------
# Simulation inputs
# ----------------------------------------------------------------------

def _cm_format_default(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{v:g}" for v in value)
    return str(value)


def cm_menu_inputs(mol: Molecule, filename: str) -> None:
    """Generate MPMC, CP2K or OpenMM inputs from the same forms the viewers show."""
    keys = list(ENGINES)
    listing = "".join(f"{i} = {ENGINES[k].label}\n" for i, k in enumerate(keys, 1))
    back = len(keys) + 1
    option = _cm_ask_option(f"\nWhich inputs?\n\n{listing}{back} = back to main menu\n\n> ")
    if not 1 <= option <= len(keys):
        if option not in (0, back):
            print("\nInvalid option!")
        return
    engine = keys[option - 1]
    spec = ENGINES[engine]
    print(f"\n{spec.description}\nPress Enter to keep a default; q cancels.\n")
    values: Dict[str, object] = {}
    try:
        for var in spec.user_vars:
            while True:
                if var.kind == "select" and var.choices:
                    values[var.name] = _cm_ask_choice(var.label, list(var.choices), var.default)
                    break
                raw = _cm_ask(var.label, _cm_format_default(var.default))
                try:
                    values[var.name] = parse_form_value(var, raw)
                    break
                except ValueError as e:
                    print(f"!!! Error: {e} !!!")
        check = check_engine(mol, engine, values)
        print(f"\n{check.preview}\n\n{check.status}")
        if not check.ok:
            return
        outdir = _cm_ask("\nOutput folder", default_engine_outdir(filename, engine))
    except _CmCancel:
        print("\nCancelled")
        return
    outcome = generate_engine_inputs(mol, engine, values, outdir)
    if outcome.preview:
        print(f"\n{outcome.preview}")
    print(f"\n{outcome.status}\n")


# ----------------------------------------------------------------------
# Databases
# ----------------------------------------------------------------------

def cm_menu_database() -> Optional[str]:
    """Search COD, RCSB, PubChem or the Materials Project and download a structure.

    Returns the downloaded file's path when the user wants to open it, else None."""
    sources = ["all"] + list(BACKENDS)
    try:
        source = _cm_ask_choice("Database", sources, "all")
        modes = list(modes_for(source))
        mode = _cm_ask_choice("Search by", modes, modes[0]) if len(modes) > 1 else modes[0]
        query = _cm_ask(f"Search for (e.g. {placeholder_for(source, mode)})")
        if not query:
            return None
        if mode == "id":
            picked = {"source": source, "id": query}
        else:
            print("\nSearching...")
            results, message = search_backends(source, mode, query)
            print(message)
            if not results:
                return None
            shown = results[:50]
            print()
            for i, r in enumerate(shown, 1):
                print(f"{i:>3}  {BACKENDS[r['source']]['label']:<17} {str(r.get('id') or ''):<12} "
                      f"{str(r.get('formula') or ''):<16} {str(r.get('title') or '')}")
            if len(results) > len(shown):
                print(f"... and {len(results) - len(shown)} more; narrow the search to see them")
            number = _cm_ask_number("\nDownload which one? (number)", 1, int, 1)
            if number > len(shown):
                print("\nInvalid option!")
                return None
            picked = shown[number - 1]
        outdir = _cm_ask("Download into folder", ".")
        print("\nDownloading...")
        path = BACKENDS[picked["source"]]["fetch"](picked["id"], outdir)
        print(f"wrote {path}")
        return path if _cm_ask_yes("Open it now? (replaces the current structure)") else None
    except _CmCancel:
        print("\nCancelled")
    except Exception as e:  # network errors, bad ids, unwritable folder
        print(f"!!! Error: {e} !!!")
    return None
