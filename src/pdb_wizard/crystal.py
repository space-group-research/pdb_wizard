"""CRYSTAL23 input and output, and the symmetry search that makes the input small.

Writing: ``write_crystal_d12`` produces a complete ``.d12`` (the group's cell-optimisation
template from common-scripts) whose geometry block follows the CRYSTAL manual's "Geometry
input for crystalline compounds":

    CRYSTAL
    0 0 0                 IFLAG IFHR IFSO: group by number, hexagonal axes, origin
                          setting (1 = first setting, for groups that have two)
    225                   space group number
    26.343                the minimal set of cell parameters for that crystal system
    7                     atoms in the asymmetric unit
    29 0.2850 0.2850 0.0  atomic number and fractional coordinates

Without a symmetry reduction the structure is written in P1 with every atom (always valid,
but the most expensive way to run CRYSTAL). With one, only the asymmetric unit of the
conventional cell is written. The reduction uses spglib, whose standard setting is the one
CRYSTAL assumes for a group given by number (unique axis b, hexagonal axes for
rhombohedral groups, centred conventional cell) except for the origin: spglib uses the
first origin setting where there are two and CRYSTAL defaults to the second, so IFSO=1 is
written for those groups.

Symmetry: ``find_symmetry`` runs spglib at a ladder of tolerances and ranks the space
groups it finds by how far the atoms have to move to fit each one exactly.

Reading: ``read_crystal_output`` takes the last geometry printed in a CRYSTAL ``.out``
(the optimised one after OPTGEOM) as a P1 structure in the primitive cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TextIO

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.constants import get_element, get_element_by_number
from pdb_wizard.io import is_fallback_cell, site_element
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC

try:  # optional: only the symmetry search needs it
    import spglib as _spglib
except ImportError:  # pragma: no cover - exercised by the no-spglib tests via monkeypatch
    _spglib = None

SPGLIB_HINT = ("The symmetry search needs spglib (pip install spglib). Without it the "
               "structure can still be written in P1.")
SYM_TOLERANCES = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)  # A
SYM_SUGGEST_DRIFT = 0.1  # A: the suggested group is the most symmetric one within this
_SYM_MATCH_CHUNK = 2_000_000

# The group's cell-optimisation settings (common-scripts/CRYSTAL/geo-opt.d12).
CRYSTAL_D12_TAIL = """OPTGEOM
FULLOPTG
ENDOPT
BASISSET
POB-TZVP-REV2
DFT
HSE06
XLGRID
ENDDFT
SHRINK
4 8
TOLINTEG
7 7 7 7 14
TOLDEE
7
MAXCYCLE
200
FMIXING
70
DIIS
END
"""

SPACE_GROUP_231 = ("Space group 231 is not in the International Tables: there are only 230. "
                   "Number 231 is the Space group at NC State, point group Brian, with no "
                   "known symmetry operations except showing up to group meeting.")


class SymmetryUnavailable(RuntimeError):
    """spglib is not installed."""


@dataclass
class SymmetryCandidate:
    """One space group found by the search."""

    number: int
    symbol: str
    tolerance: float  # the smallest tolerance (A) at which it was found
    max_drift: float  # the furthest any atom moves to fit the group exactly (A)
    rms_drift: float
    n_operations: int  # symmetry operations of the cell that was searched
    n_asymmetric: int  # atoms CRYSTAL needs in the input
    suggested: bool = False


@dataclass
class SymmetryReduction:
    """The asymmetric unit of the conventional cell, ready for a CRYSTAL input."""

    number: int
    symbol: str
    tolerance: float
    cell: tuple  # a, b, c, alpha, beta, gamma of the conventional cell
    numbers: List[int] = field(default_factory=list)  # atomic numbers
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # fractional
    n_conventional: int = 0  # atoms in the conventional cell
    max_drift: float = 0.0
    origin_flag: int = 0  # CRYSTAL's IFSO: 1 when the cell is in the first origin setting


# ----------------------------------------------------------------------
# Symmetry search
# ----------------------------------------------------------------------

def has_spglib() -> bool:
    return _spglib is not None


def space_group_name(number: int) -> str:
    """'F m -3 m'-style symbol for a space group number; 231 has its own entry."""
    if number == 231:
        return SPACE_GROUP_231
    if not 1 <= number <= 230:
        raise ValueError("Space groups are numbered 1 to 230")
    if _spglib is None:
        raise SymmetryUnavailable(SPGLIB_HINT)
    for hall in range(1, 531):
        info = _spglib.get_spacegroup_type(hall)
        if _field(info, "number") == number:
            return str(_field(info, "international_short"))
    raise ValueError(f"No space group {number}")  # pragma: no cover


def _field(obj, name: str):
    """spglib returned dicts before 2.5 and objects after."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _real_atoms(mol: Molecule) -> list:
    """(atom, atomic number) for every atom with a nucleus. Massless model sites (water M,
    H2 centres) are left out, and sorbate sites get their element from their mass, not
    their MPMC name (see ``io.site_element``)."""
    out = []
    for atom in mol.atoms:
        element = site_element(atom)
        if element is not None and element.symbol != "X" and element.atomic_number > 0:
            out.append((atom, element.atomic_number))
    return out


def _spglib_cell(mol: Molecule) -> tuple:
    if mol.pbc is None or is_fallback_cell(mol.pbc):
        raise ValueError("A symmetry search needs a unit cell")
    atoms = _real_atoms(mol)
    if not atoms:
        raise ValueError("No atoms to search")
    cart = np.array([a.x for a, _ in atoms], dtype=float)
    frac = cart @ mol.pbc.reciprocal_basis_matrix
    frac -= np.floor(frac)
    numbers = [z for _, z in atoms]
    return np.array(mol.pbc.basis_matrix, dtype=float), frac, numbers


def _dataset(cell: tuple, tolerance: float):
    if _spglib is None:
        raise SymmetryUnavailable(SPGLIB_HINT)
    return _spglib.get_symmetry_dataset(cell, symprec=tolerance)


def _drifts(cell: tuple, ds) -> np.ndarray:
    """How far (A) each input atom is from its place in spglib's idealised cell.

    spglib gives x_std = P x + p for the input atoms; the idealised structure has an atom
    of the same element at (or a lattice vector from) that point, moved by the drift."""
    _, frac, numbers = cell
    P = np.array(_field(ds, "transformation_matrix"), dtype=float)
    p = np.array(_field(ds, "origin_shift"), dtype=float)
    moved = frac @ P.T + p
    std_lat = np.array(_field(ds, "std_lattice"), dtype=float)
    std_pos = np.array(_field(ds, "std_positions"), dtype=float)
    std_types = np.array(_field(ds, "std_types"))
    numbers = np.array(numbers)
    out = np.zeros(len(moved))
    for z in np.unique(numbers):
        rows = np.where(numbers == z)[0]
        targets = std_pos[std_types == z]
        if len(targets) == 0:  # cannot happen for a consistent dataset
            out[rows] = np.inf
            continue
        step = max(1, _SYM_MATCH_CHUNK // max(len(targets), 1))
        for start in range(0, len(rows), step):
            part = rows[start:start + step]
            d = moved[part][:, None, :] - targets[None, :, :]
            d -= np.round(d)
            dist = np.linalg.norm(d @ std_lat, axis=2)
            out[part] = dist.min(axis=1)
    return out


def _asymmetric_unit(ds, tolerance: float) -> tuple:
    """(atomic numbers, fractional positions) of one atom per orbit of the idealised
    conventional cell, and the size of that cell."""
    std_lat = np.array(_field(ds, "std_lattice"), dtype=float)
    std_pos = np.array(_field(ds, "std_positions"), dtype=float)
    std_types = [int(z) for z in _field(ds, "std_types")]
    inner = _dataset((std_lat, std_pos, std_types), min(tolerance, 1e-3))
    if inner is None or int(_field(inner, "number")) != int(_field(ds, "number")):
        inner = _dataset((std_lat, std_pos, std_types), tolerance)
    if inner is None or int(_field(inner, "number")) != int(_field(ds, "number")):
        raise ValueError("The idealised cell did not keep its space group; try another "
                         "tolerance")
    reps = sorted(set(int(i) for i in _field(inner, "equivalent_atoms")))
    pos = std_pos[reps] - np.floor(std_pos[reps] + 1e-9)
    pos[np.abs(pos) < 5e-10] = 0.0
    return [std_types[i] for i in reps], pos, len(std_types)


def _cell_parameters(lattice: np.ndarray) -> tuple:
    a, b, c = (float(np.linalg.norm(v)) for v in lattice)

    def angle(u, v) -> float:
        cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    return (a, b, c, angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]),
            angle(lattice[0], lattice[1]))


def find_symmetry(mol: Molecule, tolerances=SYM_TOLERANCES) -> List[SymmetryCandidate]:
    """The space groups spglib finds as the tolerance is loosened, most symmetric last.

    Each group is listed once, at the smallest tolerance that finds it, with the atomic
    drift needed to make the structure fit it exactly."""
    cell = _spglib_cell(mol)
    found: List[SymmetryCandidate] = []
    seen = set()
    for tol in tolerances:
        ds = _dataset(cell, tol)
        if ds is None:
            continue
        number = int(_field(ds, "number"))
        if number in seen:
            continue
        seen.add(number)
        drift = _drifts(cell, ds)
        try:
            n_asym = len(_asymmetric_unit(ds, tol)[0])
        except ValueError:
            continue
        found.append(SymmetryCandidate(
            number, str(_field(ds, "international")), tol, float(drift.max()),
            float(np.sqrt((drift ** 2).mean())), len(_field(ds, "rotations")), n_asym))
    ok = [c for c in found if c.max_drift <= SYM_SUGGEST_DRIFT]
    if ok:
        max(ok, key=lambda c: (c.n_operations, -c.max_drift)).suggested = True
    return found


def reduce_symmetry(mol: Molecule, tolerance: float) -> SymmetryReduction:
    """The asymmetric unit of the conventional cell at this tolerance."""
    cell = _spglib_cell(mol)
    ds = _dataset(cell, tolerance)
    if ds is None:
        raise ValueError(f"No symmetry found at {tolerance:g} Å")
    numbers, positions, n_conv = _asymmetric_unit(ds, tolerance)
    lattice = np.array(_field(ds, "std_lattice"), dtype=float)
    return SymmetryReduction(int(_field(ds, "number")), str(_field(ds, "international")),
                             tolerance, _cell_parameters(lattice), numbers, positions, n_conv,
                             float(_drifts(cell, ds).max()),
                             _origin_flag(int(_field(ds, "number")), str(_field(ds, "choice"))))


def _origin_flag(number: int, choice: str) -> int:
    """CRYSTAL's IFSO for spglib's standard setting, after checking the setting is one
    CRYSTAL assumes for a group given by number.

    spglib's ``choice`` is '1' or '2' for the groups with two origins (it standardises to
    the first; CRYSTAL's default, IFSO=0, is the second, so IFSO=1 is written), 'H' for
    rhombohedral groups on hexagonal axes, and starts with 'b' for monoclinic unique
    axis b."""
    choice = choice.strip()
    if 3 <= number <= 15 and not choice.startswith("b"):
        raise ValueError(f"Monoclinic setting '{choice}' is not the unique-axis-b setting "
                         f"CRYSTAL expects")
    if choice == "R":
        raise ValueError("Rhombohedral axes are not supported; expected hexagonal axes")
    return 1 if choice == "1" else 0


def reduction_for_group(mol: Molecule, number: int,
                        tolerances=SYM_TOLERANCES) -> SymmetryReduction:
    """The reduction for a space group picked from ``find_symmetry``'s list."""
    if number == 231:
        raise ValueError(SPACE_GROUP_231)
    for cand in find_symmetry(mol, tolerances):
        if cand.number == number:
            return reduce_symmetry(mol, cand.tolerance)
    raise ValueError(f"Space group {number} was not found for this structure at any "
                     f"tolerance up to {max(tolerances):g} Å")


def symmetry_report(mol: Molecule) -> str:
    """The ranking shown by Find Symmetry in every front end."""
    if _spglib is None:
        return SPGLIB_HINT
    try:
        found = find_symmetry(mol)
    except ValueError as e:
        return str(e)
    n = len(_real_atoms(mol))
    lines = [f"Space groups found for {n} atoms, loosening the tolerance step by step:", "",
             f"{'group':>5}  {'symbol':<12} {'tolerance':>9}  {'max drift':>9}  "
             f"{'rms drift':>9}  {'atoms in input':>14}"]
    for c in found:
        lines.append(f"{c.number:>5}  {c.symbol:<12} {c.tolerance:>7.3f} Å  "
                     f"{c.max_drift:>7.4f} Å  {c.rms_drift:>7.4f} Å  {c.n_asymmetric:>14}"
                     + ("   <- suggested" if c.suggested else ""))
    lines += ["",
              "tolerance: the smallest search tolerance that finds the group.",
              "drift: how far atoms move when the structure is made to fit it exactly.",
              f"suggested: the most symmetric group within {SYM_SUGGEST_DRIFT:g} Å of drift. "
              f"A large drift means the",
              "symmetry is only approximate (or a guest breaks it); a higher group that "
              "needs it is not real.",
              "", "Use the group number when saving a CRYSTAL23 .d12 to write only its "
              "asymmetric unit."]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def crystal_lattice_line(number: int, cell: tuple) -> str:
    """The minimal set of cell parameters CRYSTAL reads for a space group number."""
    a, b, c, alpha, beta, gamma = cell

    def fmt(*values) -> str:
        return " ".join(f"{v:.6f}" for v in values)

    if number >= 195:  # cubic
        return fmt(a)
    if number >= 75:  # tetragonal, trigonal and hexagonal (hexagonal axes)
        return fmt(a, c)
    if number >= 16:  # orthorhombic
        return fmt(a, b, c)
    if number >= 3:  # monoclinic, unique axis b
        return fmt(a, b, c, beta)
    return fmt(a, b, c, alpha, beta, gamma)


def crystal_geometry(mol: Molecule, reduction: Optional[SymmetryReduction] = None) -> str:
    """The geometry block, from ``CRYSTAL`` to the last atom (no END)."""
    unknown = []
    for atom in mol.atoms:
        element = site_element(atom)  # None: a massless model site, left out by design
        if element is not None and (element.symbol == "X" or element.atomic_number <= 0):
            unknown.append(atom.name)
    if unknown:
        raise ValueError(
            "A CRYSTAL input needs an atomic number for every atom; these atom names have "
            f"no known element: {', '.join(sorted(set(unknown)))}")
    if reduction is None:
        if mol.pbc is None or is_fallback_cell(mol.pbc):
            raise ValueError("A CRYSTAL input needs a unit cell")
        atoms = _real_atoms(mol)
        if not atoms:
            raise ValueError("No atoms to write")
        pbc = mol.pbc
        frac = np.array([a.x for a, _ in atoms]) @ pbc.reciprocal_basis_matrix
        frac -= np.floor(frac + 1e-9)
        number = 1
        cell = (pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma)
        numbers = [z for _, z in atoms]
    else:
        number, cell = reduction.number, reduction.cell
        numbers, frac = reduction.numbers, reduction.positions
    origin = 0 if reduction is None else reduction.origin_flag
    lines = ["CRYSTAL", f"0 0 {origin}", str(number), crystal_lattice_line(number, cell),
             str(len(numbers))]
    for z, (x, y, zc) in zip(numbers, frac):
        lines.append(f"{z:<3d} {x + 0.0:15.10f} {y + 0.0:15.10f} {zc + 0.0:15.10f}")
    return "\n".join(lines)


def write_crystal_d12(mol: Molecule, out: TextIO,
                      reduction: Optional[SymmetryReduction] = None,
                      title: str = "") -> None:
    """A complete CRYSTAL23 cell-optimisation input (HSE06/POB-TZVP-REV2, the group's
    template). ``reduction`` None writes every atom in P1."""
    if not title:
        what = "P1" if reduction is None else f"space group {reduction.number}"
        title = f"PDB Wizard - cell optimization, HSE06/POB-TZVP-REV2 ({what})"
    out.write(title.splitlines()[0] + "\n")
    out.write(crystal_geometry(mol, reduction) + "\n")
    out.write(CRYSTAL_D12_TAIL)


def crystal_reduction_choice(mol: Molecule, answer: str) -> Optional[SymmetryReduction]:
    """Turn what the user typed at "space group?" into a reduction.

    Blank, 1 or P1 means no reduction (None). A number means that group from the Find
    Symmetry list. Raises ValueError with the message to show otherwise."""
    text = answer.strip().lower().replace(" ", "")
    if text in ("", "1", "p1"):
        return None
    if not text.isdigit():
        raise ValueError("Enter a space group number from Find Symmetry, or 1 for P1")
    number = int(text)
    if number == 231:
        raise ValueError(SPACE_GROUP_231)
    if not 1 <= number <= 230:
        raise ValueError("Space groups are numbered 1 to 230")
    if _spglib is None:
        raise ValueError(SPGLIB_HINT)
    return reduction_for_group(mol, number)


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def read_crystal_output(file: TextIO) -> tuple:
    """(atoms, PBC) of the last geometry in a CRYSTAL output: the optimised structure
    after OPTGEOM, or the input one for a single point. It is the primitive cell with all
    of its atoms, so the result is a complete P1 structure."""
    lines = file.read().splitlines()
    cell = None
    atoms_at = None
    for i, line in enumerate(lines):
        if "ATOMS IN THE ASYMMETRIC UNIT" in line and "ATOMS IN THE UNIT CELL" in line:
            # the cell printed just above belongs to this block
            for j in range(i - 1, max(i - 12, -1), -1):
                if "ALPHA" in lines[j] and "BETA" in lines[j] and "GAMMA" in lines[j]:
                    try:
                        values = [float(v) for v in lines[j + 1].split()[:6]]
                    except (ValueError, IndexError):
                        break
                    if len(values) == 6:
                        cell, atoms_at = values, i
                    break
    if cell is None or atoms_at is None:
        raise ValueError("No CRYSTAL geometry found (expected 'ATOMS IN THE ASYMMETRIC "
                         "UNIT' after a lattice-parameter table)")
    pbc = PBC(*cell)
    atoms: List[Atom] = []
    for line in lines[atoms_at + 1:]:
        tokens = line.split()
        if len(tokens) == 7 and tokens[0].isdigit() and tokens[1] in ("T", "F"):
            try:
                frac = np.array([float(t) for t in tokens[4:7]])
                z = int(tokens[2]) % 100  # "conventional" atomic numbers carry a basis tag
            except ValueError:
                break
            symbol = tokens[3].capitalize()
            if get_element(symbol).symbol == "X" or get_element(symbol).atomic_number != z:
                symbol = _symbol_for(z)
            xyz = frac @ pbc.basis_matrix
            atom = Atom(xyz[0], xyz[1], xyz[2], symbol)
            atoms.append(atom)
        elif atoms and not line.strip().startswith("*"):
            break
    if not atoms:
        raise ValueError("The CRYSTAL geometry block has no atoms")
    set_atom_ids(atoms)
    return atoms, pbc


def _symbol_for(atomic_number: int) -> str:
    return get_element_by_number(atomic_number).symbol
