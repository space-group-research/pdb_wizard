"""File I/O for PDB, XYZ, Zmat, CIF formats — readers, writers, trajectory support."""

from __future__ import annotations

import copy
import io as _io
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Optional, TextIO

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-frame readers
# ---------------------------------------------------------------------------

def _split_packed_floats(s: str) -> list[str]:
    """Split a string of packed floats that may lack spaces (e.g. '100.123-100.456')."""
    import re
    return re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", s)


# An MPMC atom record carries x y z mass charge alpha epsilon sigma (and
# optionally omega gwp_alpha c6 c8 c10) after the six leading fields, so it has
# at least 14 whitespace tokens. A standard PDB line has at most 12 (record,
# serial, name, resName, chainID, resSeq, x, y, z, occupancy, B-factor,
# element, formal charge) and a pdb2pqr .pqr line 10-11.
MPMC_MIN_TOKENS = 14


def _is_decimal(tok: str) -> bool:
    if "." not in tok:
        return False
    try:
        float(tok)
        return True
    except ValueError:
        return False


# PDB coordinates are %8.3f: when two overflow into each other the only
# unambiguous way to split "-1234.5002345.250" is on exactly three decimals.
_PDB_COORD_RE = re.compile(r"[+-]?\d+\.\d{3}")


def _expand_numeric_tokens(tokens: list[str]) -> list[str]:
    """Split tokens that pack several numbers together ('-1000.000-1000.000').

    Packed tokens are split on the %8.3f pattern first (the only way packing
    arises), then on a general float pattern. Non-numeric tokens are kept
    as-is so a run of coordinates can still be located relative to them.
    """
    out: list[str] = []
    for tok in tokens:
        if _is_decimal(tok) or not any(ch.isdigit() for ch in tok):
            out.append(tok)
            continue
        parts = _PDB_COORD_RE.findall(tok)
        if len(parts) > 1 and "".join(parts) == tok:
            out.extend(parts)
            continue
        parts = _split_packed_floats(tok)
        if len(parts) > 1 and "".join(parts) == tok:
            out.extend(parts)
        else:
            out.append(tok)
    return out


def _coords_from_tokens(tokens: list[str]) -> tuple[float, float, float, list[str]]:
    """(x, y, z, tokens after z) from the first run of three decimal numbers
    after the residue fields.

    Used when fixed columns are not reliable: free-form spacing, or values too
    wide for their 8-character field that ran into the neighbouring one.
    Residue sequence numbers are integers, so requiring a decimal point skips
    them.
    """
    toks = _expand_numeric_tokens(tokens[4:])
    for i in range(len(toks) - 2):
        if _is_decimal(toks[i]) and _is_decimal(toks[i + 1]) and _is_decimal(toks[i + 2]):
            return float(toks[i]), float(toks[i + 1]), float(toks[i + 2]), toks[i + 3:]
    raise ValueError("no x y z coordinates found")


def _is_mpmc_record(tokens: list[str]) -> bool:
    """MPMC atom record: 14+ tokens with x y z mass charge alpha epsilon sigma
    all numeric. A standard PDB line with a segment ID, element and formal
    charge can also reach 14 tokens, but its token 11 is the segment ID.
    """
    if len(tokens) < MPMC_MIN_TOKENS:
        return False
    try:
        for tok in tokens[6:14]:
            float(tok)
    except ValueError:
        return False
    return True


# Marker lines/fields that identify a whole file as MPMC format even when
# some atom records are short (MPMC tolerates missing trailing columns).
_MPMC_FILE_MARKERS = ("REMARK BOX BASIS",)
_MPMC_FLAGS = ("F", "M", "A", "S", "T")  # frozen / movable / adiabatic / spectre / target


def _is_mpmc_dummy_site(atom: Atom) -> bool:
    """A massless model site from an MPMC file: water M/LP charge sites, H2
    centers of mass (H2G), and so on. Shown as element X, but expected."""
    return atom.file_mass is not None and atom.file_mass < 0.5


_SITE_MASS_TOLERANCE = 0.02  # amu


def site_element(atom: Atom):
    """The chemical element an atom really is, or None for a massless model site.

    Sites of MPMC sorbate models are named after the model, not the element ('COG' and
    'COE' are the carbon and oxygens of CO2, 'CHE' a methane hydrogen), so ``atom.element``,
    which is read from the name, can be wrong for them. Their mass column is reliable: it
    is matched to the periodic table here. Framework atoms, atoms without an MPMC mass,
    and united-atom sites whose mass is no single element's keep ``atom.element``."""
    if atom.file_mass is None or atom.mpmc_flag != "M":
        return atom.element  # framework atoms are named after their element
    if atom.file_mass < 0.5:
        return None
    from pdb_wizard.constants import ELEMENTS
    best = min((el for el in ELEMENTS.values() if el.atomic_number > 0),
               key=lambda el: abs(el.mass - atom.file_mass))
    if abs(best.mass - atom.file_mass) <= _SITE_MASS_TOLERANCE:
        return best
    return atom.element


def _mpmc_mass(atom: Atom) -> float:
    """Mass to write for a movable site: the file's own value when it came from
    MPMC (dummy sites are massless). Framework atoms keep element masses."""
    return atom.file_mass if atom.file_mass is not None else atom.mass


def assign_framework_charges(mol: Molecule, charges: list[float]) -> str | None:
    """Put a list of charges on the framework atoms, in order.

    Movable (M) molecules read from an MPMC file keep their model charges, so
    the charges file only has to cover the framework. A list that divides the
    framework count is repeated (charges for one cell of a supercell). Returns
    None on success or a message saying why nothing was assigned.
    """
    framework = [a for a in mol.atoms if a.mpmc_flag != "M"]
    n = len(framework)
    if not charges or n % len(charges) != 0:
        return f"Charge count ({len(charges)}) doesn't match framework atom count ({n})"
    for ind, atom in enumerate(framework):
        atom.charge = charges[ind % len(charges)]
    return None


def mpmc_movable_count(mol: Molecule) -> int:
    """Number of movable (M) molecules read from an MPMC file and still in ``mol``."""
    return len({a.mpmc_mol for a in mol.atoms if a.mpmc_flag == "M"})


def next_mpmc_mol_id(mol: Molecule) -> int:
    """Molecule id for the next sorbate added to ``mol``: one past the highest movable id
    (the framework is molecule 1). Counting molecules is not enough: supercells and
    deletions leave gaps, and a reused id merges two sorbates into one MPMC molecule."""
    ids = [a.mpmc_mol for a in mol.atoms if a.mpmc_flag == "M" and isinstance(a.mpmc_mol, int)]
    return max(ids) + 1 if ids else 2


def _file_is_mpmc(lines: list[str]) -> bool:
    for line in lines:
        if line.startswith(_MPMC_FILE_MARKERS):
            return True
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            tokens = line.split()
            if len(tokens) > 5 and tokens[3] == "BOX" and tokens[4] in _MPMC_FLAGS:
                return True
            if _is_mpmc_record(tokens):
                return True
    return False


def _parse_mpmc_atom(tokens: list[str]) -> Atom:
    """MPMC dialect: whitespace tokens, free field widths.

    Layout: ATOM serial name resName flag molID x y z mass charge alpha epsilon
    sigma [omega gwp_alpha c6 c8 c10]. MPMC itself reads this with
    ``sscanf("%s %s ...")``, so columns carry no meaning here.
    """
    name = tokens[2]
    atom = Atom(float(tokens[6]), float(tokens[7]), float(tokens[8]), name)
    if len(tokens) > 5 and tokens[4] in _MPMC_FLAGS:
        atom.mpmc_flag = tokens[4]
        atom.mpmc_res = tokens[3]
        if tokens[5].isdigit():
            atom.mpmc_mol = int(tokens[5])
    try:
        atom.file_mass = float(tokens[9])
    except (ValueError, IndexError):
        pass
    try:
        atom.charge = float(tokens[10])
    except (ValueError, IndexError):
        pass
    try:
        atom.alpha = float(tokens[11])
        atom.epsilon = float(tokens[12])
        atom.sigma = float(tokens[13])
    except (ValueError, IndexError):
        pass
    if len(tokens) > 18:
        try:
            atom.c6 = float(tokens[16])
            atom.c8 = float(tokens[17])
            atom.c10 = float(tokens[18])
        except (ValueError, IndexError):
            pass
    return atom


def _parse_formal_charge(field: str) -> Optional[float]:
    """PDB columns 79-80: '2+', '1-' (or the lenient '+2' / '-1')."""
    f = field.strip()
    if len(f) != 2:
        return None
    if f[0].isdigit() and f[1] in "+-":
        return float(f[0]) * (1.0 if f[1] == "+" else -1.0)
    if f[1].isdigit() and f[0] in "+-":
        return float(f[1]) * (1.0 if f[0] == "+" else -1.0)
    return None


def _coordinate_columns_aligned(line: str) -> bool:
    """True if the x/y/z fields (cols 31-54) are bounded by blanks.

    A value too wide for its 8-character field runs into the neighbouring
    column; the fixed-column slice would then still parse, but to a wrong
    (truncated or sign-flipped) number. Column 30 is blank by specification
    and column 55 is the first character of the occupancy field.
    """
    if len(line) > 29 and line[29] != " ":
        return False
    if len(line) > 54 and line[54] not in " \r\n":
        return False
    return True


def _element_column(line: str) -> str:
    """Element symbol from PDB columns 77-78, tolerating an off-by-one shift.

    The symbol is right-justified in columns 77-78 (0-based 76:78), but many
    writers put it one column early or late. Take the alphabetic run touching
    those columns; give up (return '') if it is longer than two letters, which
    means it merged with a segment identifier.
    """
    n = len(line)
    for start in (76, 77, 75):
        if start < n and line[start].isalpha():
            left = start
            while left > 72 and line[left - 1].isalpha():
                left -= 1
            right = start
            while right + 1 < n and line[right + 1].isalpha():
                right += 1
            tok = line[left:right + 1]
            return tok if len(tok) <= 2 else ""
    return ""


def _parse_standard_atom(line: str, tokens: list[str], pqr: bool = False) -> Atom:
    """Standard PDB (fixed columns) or pdb2pqr .pqr dialect.

    Coordinates come from columns 31-54 as the format specifies; a line whose
    columns are misaligned falls back to the first run of three decimals.
    The element column (77-78) overrides name-based inference, so 'CA' is an
    alpha carbon and 'HO' a hydrogen, not calcium and holmium. The charge
    comes from the formal-charge columns (79-80) or, for pdb2pqr files, from
    the token after z — never from the B-factor.
    """
    name = line[12:16].strip() or tokens[2]
    after_z: Optional[list[str]] = None
    try:
        if not _coordinate_columns_aligned(line):
            raise ValueError("coordinate fields overflow their columns")
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        x, y, z, after_z = _coords_from_tokens(tokens)
        name = tokens[2]
    atom = Atom(x, y, z, name)

    elem_col = _element_column(line)
    if elem_col and not pqr:
        from pdb_wizard.constants import get_element
        if elem_col.upper() == "D":
            elem_col = "H"  # deuterium: no separate entry in the element table
        el = get_element(elem_col)
        if el.symbol != "X":
            atom._element = el

    if pqr:
        # pdb2pqr / APBS: ... x y z charge radius
        if after_z is None:
            try:
                _, _, _, after_z = _coords_from_tokens(tokens)
            except ValueError:
                after_z = None
        if after_z:
            try:
                atom.charge = float(after_z[0])
            except ValueError:
                pass
    else:
        fc = _parse_formal_charge(line[78:80])
        if fc is not None:
            atom.charge = fc
    return atom


def _parse_pdb_atom(line: str, pqr: bool = False, mpmc: bool = False) -> Atom:
    """Parse an ATOM/HETATM line, detecting the dialect per line.

    1. Serial overflow (``HETATM99999``): fixed-width name + regex coordinates.
    2. MPMC (14+ numeric-tailed tokens, or ``mpmc=True`` because the file
       carries MPMC markers): whitespace tokens, extended force-field columns.
    3. Standard PDB / pdb2pqr: fixed columns with a token fallback.

    ``pqr`` says the file has a .pqr suffix, which only matters for the
    pdb2pqr charge/radius convention; MPMC files also use .pqr but are
    recognised by their content.
    """
    tokens = line.split()

    if len(tokens[0]) > 6:
        name = line[12:16].strip()
        nums = [t for t in _expand_numeric_tokens(line[30:].split()) if _is_decimal(t)]
        if len(nums) < 3:
            raise ValueError("no x y z coordinates found")
        return Atom(float(nums[0]), float(nums[1]), float(nums[2]), name)

    if mpmc or _is_mpmc_record(tokens):
        return _parse_mpmc_atom(tokens)

    return _parse_standard_atom(line, tokens, pqr=pqr)


def read_pdb(file: TextIO, filename: str = "") -> tuple[list[Atom], Optional[PBC]]:
    """Read atoms and cell from a PDB / MPMC PDB / pdb2pqr PQR file.

    MPMC box-corner pseudo-atoms (element X in residue BOX) are dropped. Other
    atoms with an unrecognised element are kept as element 'X' and reported
    through the module logger, instead of vanishing silently.
    """
    system: list[Atom] = []
    pbc: Optional[PBC] = None
    pqr = filename.lower().endswith(".pqr")
    unknown_names: dict[str, int] = {}
    skipped: list[int] = []

    lines = file.readlines()
    # A file with MPMC markers (box corners, basis remarks, or any full MPMC
    # record) is parsed as MPMC throughout, so short records that only carry
    # mass and charge are not mistaken for standard PDB occupancy/B-factor.
    mpmc_file = _file_is_mpmc(lines)
    for lineno, line in enumerate(lines, 1):
        if len(line) < 4:
            continue
        # Fast record detection without full split
        is_atom = line[0] == "A" and line[1:4] == "TOM"
        is_hetatm = line[0] == "H" and line[1:6] == "ETATM"
        if is_atom or is_hetatm:
            try:
                atom = _parse_pdb_atom(line, pqr=pqr, mpmc=mpmc_file)
            except (ValueError, IndexError):
                skipped.append(lineno)
                continue
            if atom.atomic_number == 0:
                tokens = line.split()
                if len(tokens) > 3 and tokens[3] == "BOX":
                    continue  # MPMC box corner, not an atom
                if not _is_mpmc_dummy_site(atom):
                    unknown_names[atom.name] = unknown_names.get(atom.name, 0) + 1
            system.append(atom)
            continue
        if line[0] == "R" and line[:6] == "REMARK":
            tokens = line.split()
            if len(tokens) >= 8 and tokens[1] == "carbasis":
                try:
                    pbc = PBC(
                        float(tokens[2]), float(tokens[3]), float(tokens[4]),
                        float(tokens[5]), float(tokens[6]), float(tokens[7]),
                    )
                except (ValueError, IndexError):
                    pass
            continue
        if line[0] == "C" and line[:6] == "CRYST1":
            try:
                tokens = line.split()
                pbc = PBC(
                    float(tokens[1]), float(tokens[2]), float(tokens[3]),
                    float(tokens[4]), float(tokens[5]), float(tokens[6]),
                )
            except (ValueError, IndexError):
                pass
            continue
        if line[:3] == "END":
            break

    set_atom_ids(system)
    where = f" in {filename}" if filename else ""
    if skipped:
        shown = ", ".join(str(n) for n in skipped[:5])
        more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        _log.warning(f"Skipped {len(skipped)} unparsable atom line(s){where}: {shown}{more}")
    if unknown_names:
        names = ", ".join(f"{n} x{c}" for n, c in sorted(unknown_names.items()))
        _log.warning(f"Unrecognised element names{where} kept as 'X': {names}")
    return system, pbc


def _xyz_atom(tokens: list[str]) -> Atom:
    """Atom from an XYZ record; the symbol column may hold an atomic number."""
    symbol = tokens[0]
    if symbol.isdigit():
        from pdb_wizard.constants import get_element_by_number
        el = get_element_by_number(int(symbol))
        atom = Atom(tokens[1], tokens[2], tokens[3], el.symbol)
        atom._element = el
        return atom
    return Atom(tokens[1], tokens[2], tokens[3], symbol)


def read_xyz(file: TextIO, filename: str = "") -> tuple[list[Atom], Optional[PBC]]:
    first_line = file.readline().strip()
    pbc = None
    system: list[Atom] = []

    # First line is atom count
    try:
        n_atoms = int(first_line)
    except ValueError:
        n_atoms = None

    line = file.readline()
    try:
        tokens = line.split()
        if len(tokens) != 6:
            raise ValueError
        pbc = PBC(
            float(tokens[0]), float(tokens[1]), float(tokens[2]),
            float(tokens[3]), float(tokens[4]), float(tokens[5]),
        )
    except ValueError:
        pass

    try:
        if n_atoms is not None:
            # Read exactly n_atoms lines (handles multi-frame files)
            for _ in range(n_atoms):
                line = file.readline()
                if not line:
                    break
                tokens = line.split()
                atom = _xyz_atom(tokens)
                try:
                    atom.charge = float(tokens[4])
                except (ValueError, IndexError):
                    pass
                system.append(atom)
        else:
            for line in file.readlines():
                if line == "" or line == "\n":
                    continue
                tokens = line.split()
                atom = _xyz_atom(tokens)
                try:
                    atom.charge = float(tokens[4])
                except (ValueError, IndexError):
                    pass
                system.append(atom)
    except (ValueError, IndexError):
        pass  # stop reading on malformed line

    set_atom_ids(system)
    return system, pbc


# ---------------------------------------------------------------------------
# Z-matrix (from moltui)
# ---------------------------------------------------------------------------

def _zmat_to_cartesian(
    symbols: list[str],
    refs: list[tuple[int, ...]],
    values: list[tuple[float, ...]],
) -> list[np.ndarray]:
    coords: list[np.ndarray] = []
    for i in range(len(symbols)):
        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0]))
        elif i == 1:
            r = values[i][0]
            coords.append(np.array([r, 0.0, 0.0]))
        elif i == 2:
            r = values[i][0]
            angle = np.radians(values[i][1])
            ref_a, ref_b = refs[i][0], refs[i][1]
            d = coords[ref_a] - coords[ref_b]
            d_norm = d / (np.linalg.norm(d) + 1e-15)
            if abs(d_norm[1]) < 0.9:
                perp = np.cross(d_norm, np.array([0.0, 1.0, 0.0]))
            else:
                perp = np.cross(d_norm, np.array([1.0, 0.0, 0.0]))
            perp /= np.linalg.norm(perp) + 1e-15
            pos = coords[ref_a] + r * (-d_norm * np.cos(angle) + perp * np.sin(angle))
            coords.append(pos)
        else:
            r = values[i][0]
            angle = np.radians(values[i][1])
            dihedral = np.radians(values[i][2])
            ref_a, ref_b, ref_c = refs[i][0], refs[i][1], refs[i][2]

            ab = coords[ref_b] - coords[ref_a]
            ab /= np.linalg.norm(ab) + 1e-15
            bc = coords[ref_c] - coords[ref_b]

            n = ab
            bc_perp = bc - np.dot(bc, n) * n
            bc_perp_norm = np.linalg.norm(bc_perp)
            if bc_perp_norm < 1e-10:
                if abs(n[1]) < 0.9:
                    d2 = np.cross(n, np.array([0.0, 1.0, 0.0]))
                else:
                    d2 = np.cross(n, np.array([1.0, 0.0, 0.0]))
                d2 /= np.linalg.norm(d2)
            else:
                d2 = bc_perp / bc_perp_norm
            d3 = np.cross(n, d2)

            pos = coords[ref_a] + r * (
                -n * np.cos(angle)
                + d2 * np.sin(angle) * np.cos(dihedral)
                + d3 * np.sin(angle) * np.sin(dihedral)
            )
            coords.append(pos)
    return coords


# ---------------------------------------------------------------------------
# CIF reader
# ---------------------------------------------------------------------------

def _strip_cif_uncertainty(s: str) -> float:
    """Strip parenthesized uncertainty and parse as float: '25.832(10)' -> 25.832."""
    return float(re.sub(r"\([^)]*\)", "", s))


def _tokenize_cif_line(line: str) -> list[str]:
    """Split a CIF line into tokens, respecting single- and double-quoted strings."""
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] in (" ", "\t"):
            i += 1
            continue
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < n and line[i] != quote:
                i += 1
            tokens.append(line[start:i])
            i += 1  # skip closing quote
        else:
            start = i
            while i < n and line[i] not in (" ", "\t"):
                i += 1
            tokens.append(line[start:i])
    return tokens


def _parse_symop(expr: str) -> tuple[np.ndarray, float]:
    """Parse one component of a symmetry operation string.

    E.g. '-x+1/2' -> ([-1, 0, 0], 0.5)
    Returns (coefficients_xyz, translation).
    """
    coeffs = np.zeros(3)
    translation = 0.0
    expr = expr.strip().replace(" ", "")

    i = 0
    n = len(expr)
    while i < n:
        # Determine sign
        sign = 1.0
        if expr[i] == "+":
            sign = 1.0
            i += 1
        elif expr[i] == "-":
            sign = -1.0
            i += 1

        if i >= n:
            break

        # Check for x, y, z variable
        if expr[i] in ("x", "y", "z"):
            idx = {"x": 0, "y": 1, "z": 2}[expr[i]]
            coeffs[idx] = sign
            i += 1
        # Check for fraction or integer
        elif expr[i].isdigit() or expr[i] == ".":
            # Read the number (could be fraction like 1/2 or decimal)
            num_start = i
            while i < n and expr[i] not in ("+", "-", "x", "y", "z"):
                i += 1
            num_str = expr[num_start:i]
            if "/" in num_str:
                num, den = num_str.split("/")
                translation += sign * float(num) / float(den)
            else:
                translation += sign * float(num_str)
        else:
            i += 1  # skip unexpected character

    return coeffs, translation


def _parse_symop_full(op_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a full symmetry operation string like 'x, y, z' or '-x+1/2, y, -z+1/2'.

    Returns (rotation_matrix (3x3), translation_vector (3,)).
    """
    parts = op_string.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 components in symmetry op, got {len(parts)}: {op_string!r}")

    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    for i, part in enumerate(parts):
        coeffs, trans = _parse_symop(part)
        rotation[i] = coeffs
        translation[i] = trans

    return rotation, translation


def _apply_symops(
    frac_coords: np.ndarray,
    labels: list[str],
    symbols: list[str],
    symops: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float = 0.01,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply symmetry operations to generate all atoms in the unit cell.

    Args:
        frac_coords: (N, 3) fractional coordinates of asymmetric unit.
        labels: atom labels for each atom.
        symbols: element symbols for each atom.
        symops: list of (rotation, translation) from _parse_symop_full.
        tolerance: fractional distance below which two atoms are duplicates.

    Returns:
        (all_frac, all_labels, all_symbols) with duplicates removed.
    """
    all_frac: list[np.ndarray] = []
    all_labels: list[str] = []
    all_symbols: list[str] = []

    for rot, trans in symops:
        for j in range(len(frac_coords)):
            new_frac = rot @ frac_coords[j] + trans
            # Wrap into [0, 1)
            new_frac = new_frac % 1.0

            # Check for duplicates
            is_duplicate = False
            for existing in all_frac:
                diff = new_frac - existing
                # Handle periodic boundary: distance across 0/1 boundary
                diff = diff - np.round(diff)
                if np.linalg.norm(diff) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_frac.append(new_frac)
                all_labels.append(labels[j])
                all_symbols.append(symbols[j])

    return np.array(all_frac), all_labels, all_symbols


def _frac_to_cart(frac_coords: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian using the PBC basis matrix.

    The basis_matrix rows are the cell vectors: frac @ basis_matrix = cart.
    """
    return frac_coords @ basis_matrix


def read_cif(file: TextIO, filename: str = "",
             notes: Optional[list] = None) -> tuple[list[Atom], Optional[PBC]]:
    """Read a CIF file, returning atoms (in Cartesian) and PBC.

    Handles:
    - P1 CIF files with fractional coordinates (most common for MOFs)
    - CIF files with symmetry operations that need expansion
    - Both _atom_site_fract_x and _atom_site_Cartn_x

    ``notes`` (a list) receives a sentence for each thing the reader could not honour: a
    space group named without its symmetry operators (the atoms are then only the
    asymmetric unit) and partially occupied or disordered sites (all kept as full atoms).
    """
    lines = file.readlines()
    _sg_number = ""

    # --- Phase 1: Parse all key-value pairs and loops ---
    cell_params: dict[str, float] = {}
    symops: list[str] = []
    atom_loop_tags: list[str] = []
    atom_loop_rows: list[list[str]] = []
    _space_group: str = ""

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Skip comments and blank lines
        if not line or line.startswith("#"):
            i += 1
            continue

        # Skip data_ block headers
        if line.startswith("data_"):
            i += 1
            continue

        # Handle semicolon text fields (skip them)
        if line.startswith(";"):
            i += 1
            while i < n and not lines[i].strip().startswith(";"):
                i += 1
            i += 1  # skip closing semicolon
            continue

        # Handle loops
        if line.lower() == "loop_":
            i += 1
            loop_tags: list[str] = []
            loop_rows: list[list[str]] = []

            # Read tags
            while i < n:
                tag_line = lines[i].strip()
                if not tag_line or tag_line.startswith("#"):
                    i += 1
                    continue
                if tag_line.startswith("_"):
                    loop_tags.append(tag_line.lower())
                    i += 1
                else:
                    break

            # Read data rows
            while i < n:
                data_line = lines[i].strip()
                if not data_line or data_line.startswith("#"):
                    i += 1
                    continue
                if (data_line.startswith("_") or data_line.lower() == "loop_"
                        or data_line.startswith("data_")):
                    break
                # Handle semicolon text fields in loop data
                if data_line.startswith(";"):
                    # Collect multi-line text as single value
                    text_val = ""
                    i += 1
                    while i < n and not lines[i].strip().startswith(";"):
                        text_val += lines[i].strip() + " "
                        i += 1
                    i += 1  # skip closing semicolon
                    # This text value belongs to the current row context, but for
                    # atom loops this is rare; skip for simplicity
                    continue
                tokens = _tokenize_cif_line(data_line)
                if tokens:
                    loop_rows.append(tokens)
                i += 1

            # Identify what this loop contains
            symop_tag = None
            for tag in loop_tags:
                if tag in (
                    "_symmetry_equiv_pos_as_xyz",
                    "_space_group_symop_operation_xyz",
                ):
                    symop_tag = tag
                    break

            if symop_tag is not None:
                tag_idx = loop_tags.index(symop_tag)
                for row in loop_rows:
                    if tag_idx < len(row):
                        symops.append(row[tag_idx])

            atom_tag = None
            for tag in loop_tags:
                if tag in ("_atom_site_label", "_atom_site_type_symbol"):
                    atom_tag = tag
                    break

            if atom_tag is not None:
                atom_loop_tags = loop_tags
                atom_loop_rows = loop_rows

            continue

        # Handle key-value pairs
        tokens = _tokenize_cif_line(line)
        if len(tokens) >= 2 and tokens[0].startswith("_"):
            tag = tokens[0].lower()
            value = tokens[1]

            cell_tag_map = {
                "_cell_length_a": "a",
                "_cell_length_b": "b",
                "_cell_length_c": "c",
                "_cell_angle_alpha": "alpha",
                "_cell_angle_beta": "beta",
                "_cell_angle_gamma": "gamma",
            }
            if tag in cell_tag_map:
                try:
                    cell_params[cell_tag_map[tag]] = _strip_cif_uncertainty(value)
                except ValueError:
                    pass

            if tag in ("_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt"):
                _space_group = value.strip().strip("'\"")
            if tag in ("_symmetry_int_tables_number", "_space_group_it_number"):
                _sg_number = value.strip()

        i += 1

    # --- Phase 2: Build PBC from cell parameters ---
    pbc: Optional[PBC] = None
    required_cell = {"a", "b", "c", "alpha", "beta", "gamma"}
    if required_cell.issubset(cell_params):
        pbc = PBC(
            cell_params["a"], cell_params["b"], cell_params["c"],
            cell_params["alpha"], cell_params["beta"], cell_params["gamma"],
        )

    # --- Phase 3: Parse symmetry operations ---
    if not symops:
        # Default: identity only (P1)
        symops = ["x,y,z"]

    parsed_symops = []
    for op_str in symops:
        try:
            parsed_symops.append(_parse_symop_full(op_str))
        except ValueError:
            continue

    if not parsed_symops:
        parsed_symops = [_parse_symop_full("x,y,z")]

    is_p1 = len(parsed_symops) == 1
    if notes is not None and is_p1:
        name = _space_group.replace(" ", "").replace("_", "").upper()
        named_higher = bool(name) and name not in ("P1", "?", ".")
        numbered_higher = _sg_number.isdigit() and int(_sg_number) > 1
        if named_higher or numbered_higher:
            which = _space_group if named_higher else f"number {_sg_number}"
            notes.append(
                f"The CIF names space group {which} but lists no symmetry operators, so it "
                f"was read as P1: the atoms are only the asymmetric unit, not the full cell. "
                f"Re-export it with symmetry operators, or as P1 with all atoms (Mercury, "
                f"VESTA), before using it.")

    # --- Phase 4: Extract atom data ---
    if not atom_loop_tags or not atom_loop_rows:
        return [], pbc

    # Build column index map
    col: dict[str, int] = {}
    for idx, tag in enumerate(atom_loop_tags):
        col[tag] = idx

    has_frac = all(
        k in col for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    )
    has_cart = all(
        k in col for k in ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z")
    )

    labels: list[str] = []
    symbols: list[str] = []
    frac_coords: list[np.ndarray] = []
    cart_coords: list[np.ndarray] = []

    for row in atom_loop_rows:
        # Get label and symbol
        label = ""
        symbol = ""
        if "_atom_site_label" in col and col["_atom_site_label"] < len(row):
            label = row[col["_atom_site_label"]]
        if "_atom_site_type_symbol" in col and col["_atom_site_type_symbol"] < len(row):
            symbol = row[col["_atom_site_type_symbol"]]

        # If no type_symbol, derive from label
        if not symbol and label:
            symbol = "".join(c for c in label if c.isalpha())[:2]
            # Normalize: capitalize first, lower rest
            if len(symbol) > 1:
                symbol = symbol[0].upper() + symbol[1].lower()
            else:
                symbol = symbol.upper()
        if not label and symbol:
            label = symbol

        if not label and not symbol:
            continue

        labels.append(label)
        symbols.append(symbol)

        if has_frac:
            try:
                fx = _strip_cif_uncertainty(row[col["_atom_site_fract_x"]])
                fy = _strip_cif_uncertainty(row[col["_atom_site_fract_y"]])
                fz = _strip_cif_uncertainty(row[col["_atom_site_fract_z"]])
                frac_coords.append(np.array([fx, fy, fz]))
            except (ValueError, IndexError):
                frac_coords.append(np.array([0.0, 0.0, 0.0]))
        if has_cart:
            try:
                cx = _strip_cif_uncertainty(row[col["_atom_site_cartn_x"]])
                cy = _strip_cif_uncertainty(row[col["_atom_site_cartn_y"]])
                cz = _strip_cif_uncertainty(row[col["_atom_site_cartn_z"]])
                cart_coords.append(np.array([cx, cy, cz]))
            except (ValueError, IndexError):
                cart_coords.append(np.array([0.0, 0.0, 0.0]))

    if not labels:
        return [], pbc

    if notes is not None:
        partial = 0
        groups = set()
        for row in atom_loop_rows:
            idx = col.get("_atom_site_occupancy")
            if idx is not None and idx < len(row):
                try:
                    if _strip_cif_uncertainty(row[idx]) < 0.999:
                        partial += 1
                except ValueError:
                    pass
            idx = col.get("_atom_site_disorder_group")
            if idx is not None and idx < len(row) and row[idx] not in (".", "?", "0", ""):
                groups.add(row[idx])
        if partial or groups:
            what = []
            if partial:
                what.append(f"{partial} site{'s' if partial != 1 else ''} with occupancy "
                            f"below 1")
            if groups:
                what.append(f"{len(groups)} disorder group{'s' if len(groups) != 1 else ''}")
            notes.append(
                f"The CIF has {' and '.join(what)}. Every site was kept as a full atom, so "
                f"alternative positions overlap and the composition is wrong. Remove the "
                f"extra sites before using it.")

    # --- Phase 5: Apply symmetry and convert to Cartesian ---
    system: list[Atom] = []

    if has_frac and pbc is not None:
        frac_arr = np.array(frac_coords)

        if not is_p1:
            frac_arr, labels, symbols = _apply_symops(
                frac_arr, labels, symbols, parsed_symops
            )

        # Convert fractional to Cartesian
        cart_arr = _frac_to_cart(frac_arr, pbc.basis_matrix)

        for j in range(len(labels)):
            atom = Atom(cart_arr[j, 0], cart_arr[j, 1], cart_arr[j, 2], symbols[j])
            atom.name = labels[j]
            system.append(atom)
    elif has_cart:
        for j in range(len(labels)):
            atom = Atom(
                cart_coords[j][0], cart_coords[j][1], cart_coords[j][2], symbols[j]
            )
            atom.name = labels[j]
            system.append(atom)
    elif has_frac and pbc is None:
        # Fractional coords but no cell — treat as-is (unusual edge case)
        for j in range(len(labels)):
            atom = Atom(
                frac_coords[j][0], frac_coords[j][1], frac_coords[j][2], symbols[j]
            )
            atom.name = labels[j]
            system.append(atom)

    set_atom_ids(system)
    return system, pbc


def read_zmat(filepath: str | Path) -> tuple[list[Atom], None]:
    filepath = Path(filepath)
    with open(filepath) as f:
        text = f.read()

    sections = text.strip().split("\n\n")
    atom_lines = [l.strip() for l in sections[0].strip().splitlines() if l.strip()]

    variables: dict[str, float] = {}
    for section in sections[1:]:
        for line in section.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, val = line.split("=", 1)
                variables[name.strip()] = float(val.strip())

    def _resolve(token: str) -> float:
        try:
            return float(token)
        except ValueError:
            if token.startswith("-") and token[1:] in variables:
                return -variables[token[1:]]
            return variables[token]

    symbols: list[str] = []
    refs: list[tuple[int, ...]] = []
    values: list[tuple[float, ...]] = []

    for i, line in enumerate(atom_lines):
        parts = line.split()
        sym = ""
        for ch in parts[0]:
            if ch.isalpha():
                sym += ch
            else:
                break
        symbols.append(sym)

        if i == 0:
            refs.append(())
            values.append(())
        elif i == 1:
            refs.append((int(parts[1]) - 1,))
            values.append((_resolve(parts[2]),))
        elif i == 2:
            refs.append((int(parts[1]) - 1, int(parts[3]) - 1))
            values.append((_resolve(parts[2]), _resolve(parts[4])))
        else:
            refs.append((int(parts[1]) - 1, int(parts[3]) - 1, int(parts[5]) - 1))
            values.append((_resolve(parts[2]), _resolve(parts[4]), _resolve(parts[6])))

    positions = _zmat_to_cartesian(symbols, refs, values)
    system = [Atom(pos[0], pos[1], pos[2], sym) for sym, pos in zip(symbols, positions)]
    set_atom_ids(system)
    return system, None


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _resolve_out(out: TextIO | str):
    """Return (file_handle, should_close). Accepts a path or open writable handle."""
    if isinstance(out, str):
        return open(out, "w"), True
    return out, False


def write_xyz(mol: Molecule, out: TextIO | str) -> None:
    fh, close = _resolve_out(out)
    try:
        fh.write(str(len(mol.atoms)))
        if mol.pbc is not None:
            pbc = mol.pbc
            fh.write(f"\n{pbc.a} {pbc.b} {pbc.c} {pbc.alpha} {pbc.beta} {pbc.gamma}\n")
        else:
            fh.write("\n\n")
        for atom in mol.atoms:
            fh.write(f"{atom.element.symbol} {atom.x[0]} {atom.x[1]} {atom.x[2]}\n")
    finally:
        if close:
            fh.close()


class CellRequiredError(ValueError):
    """Raised by writers for formats that cannot be written without a unit cell."""


def require_cell(mol: Molecule, what: str) -> None:
    if mol.pbc is None:
        raise CellRequiredError(
            f"{what} requires a unit cell (PBC); this structure has none. "
            "Set one with Update Cell first."
        )


def write_standard_pdb(mol: Molecule, out: TextIO | str, skip_mols_step: bool = False) -> None:
    """Write a standardized PDB (CRYST1, HETATM records, END).

    With ``skip_mols_step`` the atoms are written in their current order and
    positions; otherwise molecules are found, sorted, named and re-imaged whole.
    A structure without a cell is written without a CRYST1 record (and without
    molecule re-imaging) — a valid non-periodic PDB, as OpenMM NoCutoff setups
    need. Earlier versions silently wrote nothing in that case.
    """
    import copy

    from pdb_wizard.operations import sort_system

    # Work on a copy so we don't reorder the caller's molecule
    mol = copy.deepcopy(mol)
    pbc = mol.pbc
    if skip_mols_step:
        mols_list = [mol]
    else:
        mol, mols_list = sort_system(mol, output=False)

    fh, close = _resolve_out(out)
    out = fh
    out.write("MODEL        1\n")
    out.write("COMPND    " + " " * 69 + "\n")
    out.write("AUTHOR    GENERATED BY PDB WIZARD\n")
    if pbc is not None:
        out.write(
            f"CRYST1  {round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
            f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6} "
            "P 1           1\n"
        )

    atom_id = 1
    lines: list[str] = []
    for idx, submol in enumerate(mols_list):
        mol_name = "UNK"
        mol_elements = [a.element.symbol for a in submol.atoms]

        if len(mol_elements) == 1:
            mol_name = mol_elements[0].upper()

        if not skip_mols_step:
            submol.atoms.sort(key=lambda a: a.element.symbol)
            mol_elements = [a.element.symbol for a in submol.atoms]
            known = {
                ("H", "H", "O"): "HOH",
                ("H", "H"): "H2",
                ("H", "H", "H", "H", "C"): "MET",
                ("N", "N"): "N2",
                ("H", "H", "C", "C"): "ACE",
                ("H", "H", "H", "H", "C", "C"): "ENE",
                ("H", "H", "H", "H", "H", "H", "C", "C"): "ETH",
                ("Zn",): "ZNA",
                ("Cl", "Cl", "Cl", "Cl", "Zn"): "ZNC",
            }
            mol_name = known.get(tuple(mol_elements), mol_name)

        base_atom = submol.atoms[-1]

        # Pre-compute per-atom name suffix without rebuilding other_elements per atom (O(N²) → O(N)).
        # An atom needs a numeric suffix if its element appears more than once in the submol.
        if not skip_mols_step:
            from collections import Counter
            elem_counts = Counter(a.element.symbol for a in submol.atoms)
            elem_running: dict[str, int] = {}
            for atom in submol.atoms:
                if elem_counts[atom.element.symbol] > 1:
                    elem_running.setdefault(atom.element.symbol, 0)
                    elem_running[atom.element.symbol] += 1
                    atom.name = atom.element.symbol + str(elem_running[atom.element.symbol])
                else:
                    atom.name = atom.element.symbol
        else:
            for atom in submol.atoms:
                atom.name = atom.element.symbol

        for atom in submol.atoms:
            atom.id = atom_id
            if not skip_mols_step and pbc is not None:
                # Keep each molecule whole: image every atom next to its base atom.
                dx = pbc.wrap(atom.x - base_atom.x)
                atom.x = base_atom.x + dx

            # Exact PDB columns: serial 7-11, name 13-16 (short names start at
            # column 14), resName 18-20, chain 22, resSeq 23-26, x/y/z 31-54,
            # occupancy 55-60, B-factor 61-66, element 77-78.
            name = atom.name[:4]
            name_field = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"HETATM{atom.id:>5} {name_field} {mol_name:>3} A{idx + 1:>4}    "
                f"{atom.x[0]:8.3f}{atom.x[1]:8.3f}{atom.x[2]:8.3f}"
                f"  1.00  0.00          {atom.element.symbol:>2}\n"
            )
            atom_id += 1
    out.write("".join(lines))

    out.write("END\n")
    set_atom_ids(mol.atoms)
    if close:
        out.close()


def write_mpmc_pdb(
    mol: Molecule,
    filename: TextIO | str,
    write_charges: bool = False,
    write_params: bool = False,
    sorbate_lines: list[str] | None = None,
) -> None:
    """Write an MPMC-format PDB (what MPMC calls a .pqr).

    Every atom record is written complete — mass, charge, alpha, epsilon,
    sigma, c6, c8, c10 — using the values currently stored on the atoms, so the
    file is always loadable by MPMC. ``write_charges`` / ``write_params`` are
    kept for API compatibility and record whether the caller assigned charges /
    force-field parameters; they do not change the record layout (the original
    tool wrote ``xxxNAMExxx`` placeholders instead, which MPMC cannot read).
    Raises CellRequiredError when ``mol.pbc`` is None.
    """
    import copy

    from pdb_wizard.operations import sort_system

    require_cell(mol, "Writing an MPMC PDB")

    # Work on a copy so we don't reorder the caller's molecule
    mol = copy.deepcopy(mol)
    pbc = mol.pbc
    # Movable (M) molecules read from an MPMC file stay movable; everything
    # else is framework and gets sorted as before.
    movable = [a for a in mol.atoms if a.mpmc_flag == "M"]
    mol = Molecule(atoms=[a for a in mol.atoms if a.mpmc_flag != "M"], pbc=pbc)
    if mol.atoms:
        mol, _ = sort_system(mol)
    groups: dict[object, list[Atom]] = {}
    for a in movable:
        groups.setdefault(a.mpmc_mol, []).append(a)

    # Box molecule ID: one past the last movable/sorbate molecule (2 if none).
    # MPMC skips BOX records, but distinct IDs keep the file unambiguous.
    box_mol_id = 2 + len(groups)
    for sl in sorbate_lines or ():
        tok = sl.split()
        if len(tok) > 5 and tok[5].isdigit():
            box_mol_id = max(box_mol_id, int(tok[5]) + 1)

    fh, close_fh = _resolve_out(filename)
    try:
        parts: list[str] = [
            "MODEL        1\n",
            "COMPND    " + " " * 69 + "\n",
            "AUTHOR    GENERATED BY PDB WIZARD\n",
            (f"CRYST1  {round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
             f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6} P 1           1\n"),
        ]
        # MOF atoms (molecule 1, frozen)
        for idx, atom in enumerate(mol.atoms):
            parts.append(
                f"ATOM {idx + 1:>6} {atom.name:<4} MOF F    1    "
                f"{atom.x[0]:7.3f} {atom.x[1]:7.3f} {atom.x[2]:7.3f}"
                f" {atom.mass:>9.5f} {atom.charge:>9.5f}"
                f" {atom.alpha:>9.5f} {atom.epsilon:>9.5f} {atom.sigma:>9.5f}"
                # omega and gwp_alpha (unused) then the PHAHST dispersion
                # coefficients at full precision. The original wrote c10 with
                # only two significant digits ({:>10.2}), e.g. 13951.5 -> 1.4e+04,
                # which MPMC happily read as a 0.35% error.
                f" 0.0 0.0 {atom.c6:>10.5f} {atom.c8:>11.4f} {atom.c10:>12.3f}\n"
            )

        next_atom_id = len(mol.atoms) + 1

        # Movable molecules from the input file (molecules 2, 3, ...), each made
        # whole again in case a wrap split it across a cell face.
        for k, atoms in enumerate(groups.values()):
            ref = atoms[0].x
            for atom in atoms:
                x = ref + pbc.wrap(atom.x - ref)
                parts.append(
                    f"ATOM {next_atom_id:>6} {atom.name:<4} {(atom.mpmc_res or 'MOL'):<3} M"
                    f" {k + 2:>4}    "
                    f"{x[0]:>10.6f} {x[1]:>10.6f} {x[2]:>10.6f}"
                    f" {_mpmc_mass(atom):>9.5f} {atom.charge:>9.5f}"
                    f" {atom.alpha:>9.5f} {atom.epsilon:>9.5f} {atom.sigma:>9.5f}"
                    f" 0.0 0.0 {atom.c6:>10.5f} {atom.c8:>12.5f} {atom.c10:>14.5f}\n"
                )
                next_atom_id += 1

        # Sorbate atoms (molecule 2, movable) — before box
        if sorbate_lines:
            parts.extend(sl + "\n" for sl in sorbate_lines)
            next_atom_id += len(sorbate_lines)

        # Box vertices (last molecule, frozen)
        corners = pbc.corners()
        for ind, pos in enumerate(corners):
            parts.append(
                f"ATOM {next_atom_id + ind:>6} X    BOX F {box_mol_id:>4}    "
                f"{round(pos[0], 3):>7} {round(pos[1], 3):>7} {round(pos[2], 3):>7} 0.0 0.0 0.0 0.0 0.0\n"
            )
        parts.extend(f"CONECT {next_atom_id + i:>4} {next_atom_id + j:>4}\n"
                     for i, j in pbc.edges())
        # MPMC's read_pqr_box needs the "=": it checks that the 4th token is "=".
        parts.extend(f"REMARK BOX BASIS[{r_idx}] = "
                     f"{row[0]:20.14f} {row[1]:20.14f} {row[2]:20.14f}\n"
                     for r_idx, row in enumerate(pbc.basis_matrix))
        parts.append("END\n")
        fh.write("".join(parts))
    finally:
        if close_fh:
            fh.close()


# ---------------------------------------------------------------------------
# Trajectory support
# ---------------------------------------------------------------------------

def check_xyz_trajectory(filename: str) -> bool:
    try:
        with open(filename) as f:
            n_atoms = int(f.readline())
            f.readline()
            for _ in range(n_atoms):
                if f.readline() == "":
                    return False
            line = f.readline()
            if line == "":
                return False
            int(line)
            f.readline()
            for _ in range(n_atoms):
                if f.readline() == "":
                    return False
        return True
    except (ValueError, OSError):
        return False


def _is_atom_record(line: str) -> bool:
    return line[:4] == "ATOM" or line[:6] == "HETATM"


def _is_frame_end(line: str) -> bool:
    """END / ENDMDL record (VMD and many converters separate frames with END)."""
    rec = line[:6].rstrip()
    return rec in ("END", "ENDMDL")


def check_pdb_trajectory(filename: str) -> bool:
    """True if the file holds more than one frame.

    Frames may be introduced by ``MODEL`` or ``REMARK step=`` records, or
    simply separated by ``END``/``ENDMDL`` with further ATOM records after it.
    """
    try:
        with open(filename) as f:
            n_model = 0
            n_remark_step = 0
            atoms_in_block = False
            block_closed = False
            for line in f:
                if line[:6] == "MODEL ":
                    n_model += 1
                elif line.startswith("REMARK step="):
                    n_remark_step += 1
                elif _is_atom_record(line):
                    if block_closed:
                        return True
                    atoms_in_block = True
                elif _is_frame_end(line) and atoms_in_block:
                    block_closed = True
                if n_model > 1 or n_remark_step > 1:
                    return True
        return False
    except OSError:
        return False


def read_xyz_trajectory(
    file: TextIO, progress_callback=None,
) -> tuple[list[Molecule], list[Optional[PBC]]]:
    molecules: list[Molecule] = []
    pbcs: list[Optional[PBC]] = []
    default_pbc = None

    # Get file size for progress reporting
    file_size = 0
    if progress_callback:
        pos = file.tell()
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(pos)

    line = file.readline()

    try:
        while line != "" and line != "\n":
            n_atoms = int(line)
            pbc: Optional[PBC] = None
            system: list[Atom] = []
            line = file.readline()
            try:
                tokens = line.split()
                if len(tokens) != 6:
                    raise ValueError
                pbc = PBC(*[float(t) for t in tokens])
            except ValueError:
                if default_pbc is None:
                    default_pbc = fallback_cell()
                pbc = copy.deepcopy(default_pbc)

            for _ in range(n_atoms):
                line = file.readline()
                tokens = line.split()
                atom = _xyz_atom(tokens)
                try:
                    atom.charge = float(tokens[4])
                except (ValueError, IndexError):
                    pass
                system.append(atom)

            set_atom_ids(system)
            molecules.append(Molecule(atoms=system, pbc=pbc))
            pbcs.append(pbc)

            if progress_callback and file_size > 0:
                progress_callback(file.tell() / file_size)

            line = file.readline()
    except (ValueError, IndexError):
        sys.exit("Error reading XYZ trajectory")

    return molecules, pbcs


def _is_frame_boundary(line: str) -> bool:
    """Check if a line marks the start of a new PDB/PQR frame."""
    return line[:6] == "MODEL " or line.startswith("REMARK step=")


# Edge length (Å) of the box assumed for trajectories that carry no cell. It is
# large enough that minimum-image never wraps, i.e. the system is treated as
# non-periodic. Callers should warn the user when they see it.
NO_CELL_FALLBACK_LENGTH = 1000000.0


def fallback_cell() -> PBC:
    """A huge cubic box standing in for 'no cell' in trajectory frames."""
    return PBC(NO_CELL_FALLBACK_LENGTH, NO_CELL_FALLBACK_LENGTH, NO_CELL_FALLBACK_LENGTH,
               90.0, 90.0, 90.0)


def is_fallback_cell(pbc: Optional[PBC]) -> bool:
    """True if ``pbc`` is the stand-in box from fallback_cell()."""
    if pbc is None:
        return False
    return (abs(pbc.a - NO_CELL_FALLBACK_LENGTH) < 1.0
            and abs(pbc.b - NO_CELL_FALLBACK_LENGTH) < 1.0
            and abs(pbc.c - NO_CELL_FALLBACK_LENGTH) < 1.0)


def read_pdb_trajectory(file: TextIO, progress_callback=None) -> tuple[list[Molecule], list[Optional[PBC]]]:
    """Read every frame of a multi-frame PDB/PQR.

    Frames start at ``MODEL``/``REMARK step=`` records or after an
    ``END``/``ENDMDL`` that closes a block of atoms. Frames without a cell get
    the shared fallback_cell() so minimum-image code works (non-periodic).
    """
    molecules: list[Molecule] = []
    pbcs: list[Optional[PBC]] = []

    all_lines = file.readlines()
    total_lines = len(all_lines)

    frame_lines: list[str] = []
    atoms_in_frame = 0

    def _flush() -> None:
        nonlocal frame_lines, atoms_in_frame
        if atoms_in_frame > 0:
            buf = _io.StringIO("".join(frame_lines))
            system, pbc = read_pdb(buf)
            if system:
                molecules.append(Molecule(atoms=system, pbc=pbc))
                pbcs.append(pbc)
            frame_lines = []
        else:
            # A cell written before MODEL (CRYST1 per frame, or once at the top as VMD
            # and MDAnalysis do) belongs to the frame that follows; don't drop it.
            frame_lines = [ln for ln in frame_lines if ln.startswith(("CRYST1", "REMARK BOX"))]
        atoms_in_frame = 0

    for li, line in enumerate(all_lines):
        if _is_frame_boundary(line):
            _flush()
            if progress_callback and li % 2000 == 0:
                progress_callback(li / total_lines)
        frame_lines.append(line)
        if _is_atom_record(line):
            atoms_in_frame += 1
        elif _is_frame_end(line):
            _flush()

    _flush()  # last frame

    if progress_callback:
        progress_callback(1.0)

    for idx, pbc in enumerate(pbcs):
        if pbc is None:
            pbcs[idx] = fallback_cell() if idx == 0 else copy.copy(pbcs[0])
            molecules[idx].pbc = pbcs[idx]

    return molecules, pbcs


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

def detect_filetype(filepath: str) -> str:
    suffix = Path(filepath).suffix.lower()
    name = Path(filepath).name.upper()
    if suffix in (".pdb", ".ent", ".pqr"):
        return "pdb"
    if suffix == ".xyz":
        return "xyz"
    if suffix in (".zmat", ".zmatrix"):
        return "zmat"
    if suffix == ".cif":
        return "cif"
    if suffix in (".vasp",) or name in ("POSCAR", "CONTCAR"):
        return "poscar"
    if suffix in (".lmp", ".lammps", ".data"):
        return "lammps"
    if suffix == ".log":
        return "gaussian_log"
    if suffix in (".com", ".gjf"):
        return "gaussian_com"
    if suffix in (".sdf", ".mol"):
        return "sdf"
    if suffix == ".mol2":
        return "mol2"
    if suffix == ".dcd":
        return "dcd"
    if suffix == ".out":
        return "crystal_out"
    raise ValueError(f"Unsupported file format: {suffix}")


def read_file(
    filepath: str,
    remove_overlaps: bool = True,
    on_message: Optional[Callable[[str], None]] = None,
) -> Molecule:
    """Read a single-structure file into a Molecule.

    Like the original pdb_wizard, duplicate atoms sitting on top of each other
    (closer than ``geometry.OVERLAP_CUTOFF`` Å under minimum image) are deleted
    on load when the file carries a cell. Each deletion is reported through
    ``on_message``; when no callback is given the messages go to the
    ``pdb_wizard.io`` logger at WARNING level (stderr by default) so the
    deletion is never silent. Pass ``remove_overlaps=False`` to keep every atom.
    """
    ft = detect_filetype(filepath)
    load_notes: list[str] = []  # what the reader had to assume; kept on the Molecule
    if ft == "pdb":
        with open(filepath) as f:
            system, pbc = read_pdb(f, filepath)
    elif ft == "xyz":
        with open(filepath) as f:
            system, pbc = read_xyz(f, filepath)
    elif ft == "zmat":
        system, pbc = read_zmat(filepath)
    elif ft == "cif":
        with open(filepath) as f:
            system, pbc = read_cif(f, filepath, load_notes)
    elif ft == "poscar":
        from pdb_wizard.formats import read_poscar
        with open(filepath) as f:
            system, pbc = read_poscar(f)
    elif ft == "lammps":
        from pdb_wizard.formats import read_lammps_data
        with open(filepath) as f:
            system, pbc = read_lammps_data(f)
    elif ft == "gaussian_log":
        from pdb_wizard.formats import read_gaussian_log
        with open(filepath) as f:
            system, pbc = read_gaussian_log(f)
    elif ft == "gaussian_com":
        from pdb_wizard.formats import read_gaussian_com
        with open(filepath) as f:
            system, pbc = read_gaussian_com(f)
    elif ft == "sdf":
        from pdb_wizard.formats import read_sdf
        with open(filepath) as f:
            system, pbc = read_sdf(f)
    elif ft == "mol2":
        from pdb_wizard.formats import read_mol2
        with open(filepath) as f:
            system, pbc = read_mol2(f)
    elif ft == "crystal_out":
        from pdb_wizard.crystal import read_crystal_output
        with open(filepath, errors="replace") as f:
            system, pbc = read_crystal_output(f)
    elif ft == "dcd":
        raise ValueError(
            "DCD files are trajectories; open them with read_file_trajectory "
            "(or the TUI / --msd / --rdf)."
        )
    else:
        raise ValueError(f"Unknown filetype: {ft}")

    mol = Molecule(atoms=system, pbc=pbc)
    if load_notes:
        mol.load_notes = load_notes  # read by checks.load_notes and shown by the front ends
    if remove_overlaps:
        from pdb_wizard.geometry import remove_overlapping_atoms
        mol, messages = remove_overlapping_atoms(mol)
        report = on_message if on_message is not None else _log.warning
        for message in messages:
            report(message)
    if load_notes:
        mol.load_notes = load_notes
    mol.detect_bonds()
    return mol


def read_file_trajectory(filepath: str) -> list[Molecule] | None:
    """Read a trajectory file if it is one, otherwise return None."""
    ft = detect_filetype(filepath)
    is_traj = False
    if ft == "pdb":
        is_traj = check_pdb_trajectory(filepath)
    elif ft == "xyz":
        is_traj = check_xyz_trajectory(filepath)
    elif ft == "dcd":
        from pdb_wizard.dcd import check_dcd_trajectory
        is_traj = check_dcd_trajectory(filepath)
    if not is_traj:
        return None

    if ft == "pdb":
        with open(filepath) as f:
            mols, _ = read_pdb_trajectory(f)
    elif ft == "xyz":
        with open(filepath) as f:
            mols, _ = read_xyz_trajectory(f)
    elif ft == "dcd":
        from pdb_wizard.dcd import read_dcd_trajectory
        mols, _ = read_dcd_trajectory(filepath)
    else:
        return None

    for mol in mols:
        mol.detect_bonds()
    return mols


# ---------------------------------------------------------------------------
# Charge files (CP2K RESP and raw columns)
# ---------------------------------------------------------------------------

def parse_resp_charges(lines: list[str]) -> list[float]:
    """Extract partial charges from CP2K RESP output lines.

    CP2K writes a couple of header lines, one charge per atom (the charge is the
    last whitespace-separated token on the line), and a trailing
    ``Total charge of the system: ...`` line. Header lines are skipped because
    their last token is not a float; the ``Total`` line is skipped explicitly
    (its last token *is* a float and would otherwise be read as an atom charge).
    """
    charges: list[float] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        words = line.split()
        if words[0].lower() == "total":
            continue
        try:
            charges.append(float(words[-1]))
        except ValueError:
            continue  # header / non-data line
    return charges


def read_charges_file(
    path: str, skip_first: int = 0, skip_last: int = 0,
) -> list[float]:
    """Read partial charges from a file.

    CP2K ``.resp`` files are auto-detected and parsed via
    :func:`parse_resp_charges` (header/footer/``Total`` lines stripped, last
    column taken). For any other file the last whitespace-separated token of
    each non-empty line is taken as the charge, after trimming ``skip_first``
    leading and ``skip_last`` trailing lines.
    """
    with open(path) as f:
        lines = f.readlines()
    if path.lower().endswith(".resp"):
        return parse_resp_charges(lines)
    end = max(0, len(lines) - skip_last)
    used = lines[skip_first:end]
    charges: list[float] = []
    for raw in used:
        line = raw.strip()
        if not line:
            continue
        charges.append(float(line.split()[-1]))
    return charges
