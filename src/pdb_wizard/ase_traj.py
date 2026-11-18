"""Reader for ASE trajectory files (``.traj``), without needing ASE.

ASE writes trajectories in its ULM container: a 48-byte header (the magic ``- of Ulm``, a
16-byte tag, then the format version, the number of items and the position of the offset
table as little-endian 64-bit integers), a table of item offsets, and for each frame a
length-prefixed JSON document. Large arrays are not in the JSON: a key ending in a dot
(``positions.``) holds ``{"ndarray": [shape, dtype, offset]}`` pointing at raw
little-endian data elsewhere in the file. The first frame carries what does not change
(atomic numbers, periodic flags); later frames repeat only what does (positions, cell).

Read here: elements, positions, the cell (when the system is periodic) and, when the
trajectory was written with a calculator that reports them, per-atom charges.
"""

from __future__ import annotations

import json
import struct
from typing import List, Optional

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.constants import get_element_by_number
from pdb_wizard.io import fallback_cell
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC

ASE_TRAJ_MAGIC = b"- of Ulm"
_ASE_HEADER = struct.Struct("<8s16sqqq")


def _ase_header(f) -> tuple:
    """(tag, version, number of items, offset of the offset table), or raise ValueError."""
    raw = f.read(_ASE_HEADER.size)
    if len(raw) < _ASE_HEADER.size or raw[:8] != ASE_TRAJ_MAGIC:
        raise ValueError("not an ASE trajectory (the file does not start with '- of Ulm')")
    _, tag, version, n_items, table = _ASE_HEADER.unpack(raw)
    if n_items < 0 or table < 0:
        raise ValueError("ASE trajectory header is damaged")
    return tag.decode("ascii", "replace").strip(), version, n_items, table


def check_ase_trajectory(filepath: str) -> bool:
    """True when the file is an ASE trajectory with at least one frame."""
    try:
        with open(filepath, "rb") as f:
            return _ase_header(f)[2] > 0
    except (OSError, ValueError):
        return False


def _ase_array(f, ref) -> np.ndarray:
    shape, dtype, offset = ref["ndarray"]
    count = int(np.prod(shape)) if shape else 1
    dt = np.dtype(dtype).newbyteorder("<")
    f.seek(offset)
    data = f.read(count * dt.itemsize)
    if len(data) < count * dt.itemsize:
        raise ValueError("ASE trajectory ends in the middle of an array (still being written?)")
    return np.frombuffer(data, dtype=dt, count=count).reshape(shape)


def _ase_value(f, item: dict, key: str):
    """The value stored under ``key`` (inline) or ``key.`` (a reference to raw data)."""
    if key + "." in item:
        return _ase_array(f, item[key + "."])
    if key in item:
        return np.asarray(item[key])
    return None


def _ase_item(f, offset: int) -> dict:
    f.seek(offset)
    raw = f.read(8)
    if len(raw) < 8:
        raise ValueError("ASE trajectory ends before a frame (still being written?)")
    (length,) = struct.unpack("<q", raw)
    data = f.read(length)
    if length < 0 or len(data) < length:
        raise ValueError("ASE trajectory ends in the middle of a frame (still being written?)")
    return json.loads(data.decode("utf-8"))


def _ase_cell(cell, periodic) -> tuple:
    """(PBC or None, 3x3 lattice or None) for an ASE cell. A cell counts only when the
    system is periodic in some direction and the three vectors span a volume."""
    if cell is None or periodic is None or not np.any(periodic):
        return None, None
    lattice = np.asarray(cell, dtype=float).reshape(3, 3)
    if abs(np.linalg.det(lattice)) < 1e-9:
        return None, None
    a, b, c = (float(np.linalg.norm(v)) for v in lattice)

    def angle(u, v) -> float:
        cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    pbc = PBC(a, b, c, angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]),
              angle(lattice[0], lattice[1]))
    return pbc, lattice


def read_ase_trajectory(filepath: str, progress_callback=None) -> tuple:
    """Read an ASE ``.traj`` file. Returns (frames, cells), like the other trajectory
    readers: frames without a periodic cell get the stand-in box and ``None`` in ``cells``.

    Frames that were only partly written (a run still in progress) are left out; the
    complete ones before them are returned. Raises ValueError when no frame can be read.
    """
    frames: List[Molecule] = []
    cells: List[Optional[PBC]] = []
    with open(filepath, "rb") as f:
        _, _, n_items, table = _ase_header(f)
        f.seek(table)
        raw = f.read(8 * n_items)
        if len(raw) < 8 * n_items:
            raise ValueError("ASE trajectory ends before its table of frames")
        offsets = struct.unpack("<%dq" % n_items, raw)

        numbers = None
        periodic = None
        symbols: List[str] = []
        for index, offset in enumerate(offsets):
            try:
                item = _ase_item(f, offset)
                frame_numbers = _ase_value(f, item, "numbers")
                if frame_numbers is not None:
                    numbers = np.asarray(frame_numbers, dtype=int)
                    symbols = [get_element_by_number(int(z)).symbol for z in numbers]
                if "pbc" in item or "pbc." in item:
                    periodic = np.asarray(_ase_value(f, item, "pbc"), dtype=bool)
                positions = _ase_value(f, item, "positions")
                if numbers is None or positions is None:
                    raise ValueError("ASE trajectory frame has no atoms")
                positions = np.asarray(positions, dtype=float).reshape(-1, 3)
                if len(positions) != len(numbers):
                    raise ValueError("ASE trajectory frame does not match its atom list")
                pbc, lattice = _ase_cell(_ase_value(f, item, "cell"), periodic)
                charges = _ase_value(f, item.get("calculator.", item.get("calculator", {})),
                                     "charges")
            except ValueError:
                if frames:  # a run that is still writing: keep what is complete
                    break
                raise
            if pbc is not None:
                # ASE allows any orientation of the cell; this program keeps a along x and
                # b in the xy plane, so the atoms are carried over by fractional coordinates
                positions = (positions @ np.linalg.inv(lattice)) @ pbc.basis_matrix
            atoms = [Atom(float(x), float(y), float(z), symbol)
                     for (x, y, z), symbol in zip(positions, symbols)]
            if charges is not None and len(charges) == len(atoms):
                for atom, q in zip(atoms, np.asarray(charges, dtype=float)):
                    atom.charge = float(q)
            set_atom_ids(atoms)
            frames.append(Molecule(atoms=atoms, pbc=pbc if pbc is not None else fallback_cell()))
            cells.append(pbc)
            if progress_callback is not None:
                progress_callback((index + 1) / max(n_items, 1))
    if not frames:
        raise ValueError("ASE trajectory has no frames")
    return frames, cells
