"""Guard rails: things worth knowing about a structure or trajectory before simulating it.

``check_structure`` looks for the mistakes that give a wrong simulation without any error
message: a CIF read as its asymmetric unit, disordered sites kept as full atoms, missing
hydrogens, atoms on top of each other, no charges, a cell too small for the cutoff, and so
on. It never changes the structure. The terminal viewer, the desktop GUI and the classic
menu all show the same report (``structure_check_text``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pdb_wizard.geometry import _pairs_within, _same_movable_molecule, get_lone_atoms
from pdb_wizard.io import is_fallback_cell
from pdb_wizard.molecule import Molecule

CHECK_ERROR, CHECK_WARNING, CHECK_NOTE = "error", "warning", "note"
_CHECK_MARKS = {CHECK_ERROR: "✗", CHECK_WARNING: "⚠", CHECK_NOTE: "•"}

CHECK_CUTOFF = 10.0  # A, the cutoff the cell is judged against (half the smallest width)
_CLOSE_CONTACT = 0.7  # A, closer than any bond: overlapping or disordered atoms
# Elements usually open-shell in MOF nodes; CP2K needs UKS and a multiplicity for them.
OPEN_SHELL_METALS = frozenset({"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "Ru",
                               "Rh", "W", "Re", "Os", "Ir", "Ce", "Gd", "Eu", "Tb", "Dy"})
_ORGANIC_NEEDS_H = 4  # carbons before "no hydrogens at all" is worth a warning


@dataclass
class CheckFinding:
    level: str  # CHECK_ERROR, CHECK_WARNING or CHECK_NOTE
    title: str
    detail: str = ""


def load_notes(mol: Optional[Molecule]) -> List[str]:
    """What the file reader had to assume (set by ``read_file``; empty for most files)."""
    return list(getattr(mol, "load_notes", None) or [])


def open_shell_metals(mol: Molecule) -> List[str]:
    return sorted({a.element.symbol for a in mol.atoms} & OPEN_SHELL_METALS)


def lone_atom_summary(mol: Molecule) -> str:
    """What Delete Lone Atoms would remove, e.g. '48 Cu, 2 O' ('' when nothing)."""
    if mol.pbc is None:
        return ""
    counts: dict = {}
    for atom in get_lone_atoms(mol):
        counts[atom.element.symbol] = counts.get(atom.element.symbol, 0) + 1
    return ", ".join(f"{n} {el}" for el, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def lone_atom_warning(mol: Molecule) -> str:
    """A caution when Delete Lone Atoms would remove metal centres ('' otherwise)."""
    if mol.pbc is None:
        return ""
    metals = sorted({a.element.symbol for a in get_lone_atoms(mol) if a.element.symbol
                     not in ("H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "He", "Ne",
                             "Ar", "Kr", "Xe", "B", "Si")})
    if not metals:
        return ""
    return (f"This includes metal atoms ({', '.join(metals)}). Metal-ligand bonds are longer "
            f"than the contact distance this test uses, so framework metals can look "
            f"'lone'. Deleting them would break the framework.")


_BIO_ELEMENTS = frozenset({"H", "C", "N", "O", "S", "P", "Se", "Na", "K", "Cl", "Mg", "Ca"})


def openmm_template_warning(mol: Molecule, forcefield: str) -> str:
    """Why a stock biomolecular force field will not load this structure ('' if it may)."""
    if not any(key in forcefield.lower() for key in ("amber", "charmm", "tip", "spce")):
        return ""  # the user's own force-field file
    other = sorted({a.element.symbol for a in mol.atoms} - _BIO_ELEMENTS)
    names = {a.name.strip().upper() for a in mol.atoms}
    protein = {"CA", "N", "C", "O"} <= names
    water_only = {a.element.symbol for a in mol.atoms} <= {"H", "O"}
    if not other and (protein or water_only):
        return ""
    what = f"contains {', '.join(other)}" if other else "is not a protein or water"
    return (f"The force field ({forcefield}) only has templates for proteins, nucleic acids, "
            f"water and common ions, and this structure {what}. OpenMM will stop with "
            f"'No template found for residue'. Give your own force-field XML for a framework.")


def supercell_for_cutoff(mol: Molecule, cutoff: float = CHECK_CUTOFF) -> tuple:
    """Cells needed along a, b, c for half the smallest width to reach ``cutoff``."""
    widths = mol.pbc.perpendicular_widths()
    return tuple(int(np.ceil(2.0 * cutoff / w - 1e-9)) for w in widths)


def mc_trajectory_reason(frames: Optional[list]) -> str:
    """Why a trajectory looks like Monte Carlo output ('' if it does not).

    Monte Carlo frames have no time between them, and in grand-canonical runs molecules
    appear and disappear, so atom i is not the same molecule from frame to frame."""
    if not frames or len(frames) < 2:
        return ""
    counts = {len(m.atoms) for m in frames}
    if len(counts) > 1:
        return (f"the number of atoms changes between frames ({min(counts)} to "
                f"{max(counts)}), as in a grand-canonical (μVT) run")
    first = frames[0]
    flags = {getattr(a, "mpmc_flag", "") for a in first.atoms}
    if "M" in flags and "F" in flags:
        return "the frames are MPMC output (frozen framework 'F' and movable 'M' molecules)"
    return ""


def mc_trajectory_warning(frames: Optional[list], what: str) -> str:
    reason = mc_trajectory_reason(frames)
    if not reason:
        return ""
    return (f"⚠ This looks like a Monte Carlo trajectory: {reason}. Monte Carlo has no time, "
            f"so {what} here is not a physical result.")


def check_structure(mol: Optional[Molecule], frames: Optional[list] = None,
                    cutoff: float = CHECK_CUTOFF) -> List[CheckFinding]:
    """Everything worth a look before simulating ``mol``. Most serious first."""
    out: List[CheckFinding] = []
    if mol is None or not mol.atoms:
        return [CheckFinding(CHECK_ERROR, "No atoms loaded")]
    atoms = mol.atoms
    symbols = [a.element.symbol for a in atoms]
    n = len(atoms)

    for note in load_notes(mol):
        out.append(CheckFinding(CHECK_WARNING, "From the file reader", note))

    unknown = sum(s == "X" for s in symbols)
    if unknown:
        out.append(CheckFinding(CHECK_WARNING, f"{unknown} atoms have an unknown element",
                                "Their masses and force-field parameters will be wrong."))

    # ---- cell
    has_cell = mol.pbc is not None and not is_fallback_cell(mol.pbc)
    if not has_cell:
        out.append(CheckFinding(
            CHECK_WARNING, "No unit cell",
            "Periodic simulations (MPMC, CP2K) need one. Set it with Update Unit Cell."))
    else:
        half = float(mol.pbc.perpendicular_widths().min()) / 2.0
        if half < cutoff:
            na, nb, nc = supercell_for_cutoff(mol, cutoff)
            out.append(CheckFinding(
                CHECK_WARNING, f"Cell is small for a {cutoff:g} Å cutoff",
                f"Half the smallest cell width is {half:.2f} Å. A {na} × {nb} × {nc} supercell "
                f"({n * na * nb * nc:,} atoms) reaches {cutoff:g} Å."))
        else:
            out.append(CheckFinding(CHECK_NOTE, f"Cell allows a cutoff up to {half:.2f} Å"))

    # ---- atoms on top of each other
    if has_cell and n > 1:
        i, j, d = _pairs_within(mol, _CLOSE_CONTACT)
        # sites of one rigid sorbate model (and its massless sites) sit close by design
        keep = np.array([atoms[p].mass > 0 and atoms[q].mass > 0
                         and not _same_movable_molecule(atoms[p], atoms[q])
                         for p, q in zip(i, j)], dtype=bool)
        i, j, d = i[keep], j[keep], d[keep]
        if len(i):
            k = int(np.argmin(d))
            out.append(CheckFinding(
                CHECK_ERROR, f"{len(i)} pairs of atoms are closer than {_CLOSE_CONTACT} Å",
                f"Closest: {symbols[i[k]]}{i[k] + 1}-{symbols[j[k]]}{j[k] + 1} at {d[k]:.2f} Å. "
                f"This usually means disordered sites, a duplicated molecule, or the wrong "
                f"cell."))

    # ---- hydrogens
    n_c = symbols.count("C")
    if n_c >= _ORGANIC_NEEDS_H and "H" not in symbols:
        out.append(CheckFinding(
            CHECK_WARNING, "No hydrogen atoms",
            f"There are {n_c} carbons and no hydrogens. Crystal structures often leave "
            f"them out; add them before assigning charges or a force field."))

    # ---- unbonded atoms
    if has_cell:
        summary = lone_atom_summary(mol)
        if summary:
            out.append(CheckFinding(
                CHECK_NOTE, f"Atoms with no close neighbour: {summary}",
                lone_atom_warning(mol) or "These may be leftover solvent or counter-ions."))

    # ---- charges and force-field parameters (framework atoms only)
    frame_atoms = [a for a in atoms if getattr(a, "mpmc_flag", "") != "M"] or atoms
    charges = np.array([a.charge for a in frame_atoms])
    if not np.any(np.abs(charges) > 1e-9):
        out.append(CheckFinding(
            CHECK_WARNING, "No charges",
            "Every charge is zero. Load fitted charges or generate QEq charges."))
    else:
        net = float(np.array([a.charge for a in atoms]).sum())
        if abs(net) > 1e-3:
            out.append(CheckFinding(
                CHECK_WARNING, f"Net charge {net:+.4f} e",
                "A periodic simulation cell should be neutral. Check for missing "
                "counter-ions or rescale the charges."))
        else:
            out.append(CheckFinding(CHECK_NOTE, f"Charges present, net {net:+.4f} e"))
    eps = np.array([a.epsilon for a in frame_atoms])
    n_eps = int((eps > 0).sum())
    if n_eps == 0:
        out.append(CheckFinding(CHECK_WARNING, "No force-field parameters",
                                "Every ε is zero. Apply a force field before writing MPMC "
                                "input."))
    elif n_eps < len(frame_atoms):
        missing = sorted({a.element.symbol for a in frame_atoms if a.epsilon <= 0
                          and a.mass > 0})
        if missing:
            out.append(CheckFinding(
                CHECK_WARNING, f"No force-field parameters for {', '.join(missing)}",
                "The chosen force field has no entry for these elements."))

    # ---- things to remember later
    metals = open_shell_metals(mol)
    if metals:
        out.append(CheckFinding(
            CHECK_NOTE, f"Open-shell metals: {', '.join(metals)}",
            "For CP2K add UKS and a MULTIPLICITY; the generated input is closed-shell."))
    reason = mc_trajectory_reason(frames)
    if reason:
        out.append(CheckFinding(
            CHECK_NOTE, "Monte Carlo trajectory",
            f"{reason[0].upper()}{reason[1:]}. MSD and RMSD have no physical meaning here; "
            f"RDF, density and coordination are fine."))

    order = {CHECK_ERROR: 0, CHECK_WARNING: 1, CHECK_NOTE: 2}
    out.sort(key=lambda f: order[f.level])
    return out


def format_findings(findings: List[CheckFinding]) -> str:
    n_err = sum(f.level == CHECK_ERROR for f in findings)
    n_warn = sum(f.level == CHECK_WARNING for f in findings)
    if n_err or n_warn:
        parts = []
        if n_err:
            parts.append(f"{n_err} problem{'s' if n_err != 1 else ''}")
        if n_warn:
            parts.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}")
        lines = ["Structure check: " + ", ".join(parts), ""]
    else:
        lines = ["Structure check: nothing suspicious found", ""]
    for f in findings:
        lines.append(f"{_CHECK_MARKS[f.level]} {f.title}")
        if f.detail:
            lines.append(f"    {f.detail}")
    lines += ["", "This is a set of heuristics, not a proof that the structure is right."]
    return "\n".join(lines)


def structure_check_text(mol: Optional[Molecule], frames: Optional[list] = None) -> str:
    return format_findings(check_structure(mol, frames))
