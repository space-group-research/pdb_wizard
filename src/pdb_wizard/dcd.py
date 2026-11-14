"""CHARMM/NAMD/OpenMM DCD trajectory reader.

DCD is a Fortran-style binary format. Each record is bracketed by 4-byte
int32 length markers. Layout:

    Header record (84 bytes payload):
        4 bytes "CORD" magic
        20 × int32 (icntrl):
            icntrl[0]  = NSET    — number of frames
            icntrl[1]  = ISTART  — start timestep
            icntrl[2]  = NSAVC   — steps between frames
            icntrl[3]  = NSTEP   — total simulation steps
            icntrl[7]  = NDEGF   — degrees of freedom
            icntrl[8]  = NFROZEN — frozen atoms
            icntrl[9]  = DELTA   — time step (float32 stored in int slot)
            icntrl[10] = USE_BOX — 1 if unit-cell info is written
            icntrl[19] = VERSION — CHARMM version (24 = unit cell present)
    Title record (4-byte ntitle + ntitle × 80-byte titles)
    N_atoms record (1 × int32)
    Per frame:
        if USE_BOX: 6 × float64 record (a, gamma, b, beta, alpha, c)
        x[N] as N × float32
        y[N] as N × float32
        z[N] as N × float32

DCD does NOT carry element symbols. Our reader optionally pairs the DCD
with a topology file (PDB/XYZ/PSF) of the same atom count to assign
elements. Without a topology, every atom is loaded as carbon and a flag
is set on the returned molecules so the UI can warn the user.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC


def _detect_endian(fh: BinaryIO) -> str:
    """First 4 bytes are the record-length prefix of the 84-byte header.
    Read it in both byte orders; whichever yields 84 is the file's endian."""
    raw = fh.read(4)
    fh.seek(0)
    if len(raw) < 4:
        raise ValueError("DCD file too short")
    if struct.unpack("<i", raw)[0] == 84:
        return "<"
    if struct.unpack(">i", raw)[0] == 84:
        return ">"
    raise ValueError(
        f"DCD header record length is not 84 in either endianness "
        f"(read {raw!r}). File may be corrupt or not a DCD."
    )


def _read_record(fh: BinaryIO, endian: str) -> bytes:
    """Read one Fortran record: <int32 length> <payload> <int32 length>.
    Verifies the suffix length matches the prefix."""
    head = fh.read(4)
    if len(head) < 4:
        raise EOFError("Unexpected EOF reading record header")
    n = struct.unpack(endian + "i", head)[0]
    payload = fh.read(n)
    if len(payload) != n:
        raise EOFError(f"Short read: expected {n} bytes, got {len(payload)}")
    tail = fh.read(4)
    if len(tail) < 4:
        raise EOFError("Unexpected EOF reading record trailer")
    nt = struct.unpack(endian + "i", tail)[0]
    if nt != n:
        raise ValueError(
            f"DCD record bracket mismatch: head={n}, tail={nt} "
            f"(file likely corrupt or written in a different endianness)"
        )
    return payload


def _parse_header(payload: bytes, endian: str) -> dict:
    """Header payload: 4-char magic + 20 int32."""
    if len(payload) != 84:
        raise ValueError(f"Header payload should be 84 bytes, got {len(payload)}")
    magic = payload[:4]
    if magic != b"CORD":
        raise ValueError(f"Expected CORD magic, got {magic!r}")
    icntrl = struct.unpack(endian + "20i", payload[4:84])
    return {
        "n_frames": icntrl[0],
        "i_start": icntrl[1],
        "n_savc": icntrl[2],
        "n_step": icntrl[3],
        "n_degf": icntrl[7],
        "n_frozen": icntrl[8],
        "delta_raw": icntrl[9],  # float32 packed into int32 slot in CHARMM ≥24
        "has_box": bool(icntrl[10]),
        "version": icntrl[19],
    }


def _read_topology_elements(topo_path: str, n_expected: int) -> list[str] | None:
    """Read element symbols from a sibling PDB/XYZ topology file.
    Returns None if it doesn't fit (different atom count) or can't be parsed."""
    suffix = Path(topo_path).suffix.lower()
    try:
        if suffix in (".pdb", ".ent", ".pqr"):
            from pdb_wizard.io import read_pdb
            with open(topo_path) as f:
                atoms, _ = read_pdb(f)
        elif suffix == ".xyz":
            from pdb_wizard.io import read_xyz
            with open(topo_path) as f:
                atoms, _ = read_xyz(f)
        else:
            return None
        if len(atoms) != n_expected:
            return None
        return [a.element.symbol for a in atoms]
    except Exception:
        return None


def _sibling_topologies(dcd_path: str) -> list:
    """Every candidate topology next to the DCD, in the order of _find_sibling_topology:
    same-stem files first, then the other .pdb / .pqr / .xyz files of the folder."""
    p = Path(dcd_path)
    found = [str(p.with_suffix(ext)) for ext in (".pdb", ".pqr", ".ent", ".xyz")
             if p.with_suffix(ext).exists()]
    for ext in (".pdb", ".pqr", ".xyz"):
        for sibling in sorted(p.parent.glob(f"*{ext}")):
            if str(sibling) not in found:
                found.append(str(sibling))
    return found


def _find_sibling_topology(dcd_path: str) -> str | None:
    """Locate a topology file next to the DCD.

    Lookup order:
      1. Same stem with PDB/PQR/ENT/XYZ extension (e.g. traj.pdb next to traj.dcd)
      2. Any .pdb / .pqr / .xyz in the same directory — common when the
         topology was written by openmm (e.g. system.pdb or final.pdb
         next to trajectory.dcd).
    """
    p = Path(dcd_path)
    # 1. Same-stem match
    for ext in (".pdb", ".pqr", ".ent", ".xyz"):
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    # 2. Any topology in the same directory
    for ext in (".pdb", ".pqr", ".xyz"):
        siblings = sorted(p.parent.glob(f"*{ext}"))
        if siblings:
            return str(siblings[0])
    return None


def read_dcd_trajectory(
    filepath: str,
    topology_file: str | None = None,
    progress_callback=None,
) -> tuple[list[Molecule], list[Optional[PBC]]]:
    """Read a DCD file. Returns (frames, pbcs) — same shape as
    read_pdb_trajectory and read_xyz_trajectory.

    If `topology_file` is None, searches for a sibling PDB/XYZ next to the
    DCD. Without any topology, every atom is loaded as carbon and the
    returned Molecules carry `mol._dcd_topology_missing = True` so the UI
    can warn the user.
    """
    # Resolve element list from topology
    # (the first sibling with the right atom count, not just the first sibling)
    topology_paths = [topology_file] if topology_file else _sibling_topologies(filepath)

    with open(filepath, "rb") as fh:
        endian = _detect_endian(fh)
        header_payload = _read_record(fh, endian)
        hdr = _parse_header(header_payload, endian)

        # Title record (skip — purely informational)
        _read_record(fh, endian)

        # N_atoms record (one int32)
        n_atoms_payload = _read_record(fh, endian)
        if len(n_atoms_payload) != 4:
            raise ValueError(
                f"N_atoms record should be 4 bytes, got {len(n_atoms_payload)}"
            )
        n_atoms = struct.unpack(endian + "i", n_atoms_payload)[0]
        if n_atoms <= 0:
            raise ValueError(f"DCD reports {n_atoms} atoms — invalid")

        # Try to fetch element symbols from topology
        elements: list[str] = ["C"] * n_atoms
        topology_missing = True
        for topology_path in topology_paths:
            els = _read_topology_elements(topology_path, n_atoms)
            if els is not None:
                elements = els
                topology_missing = False
                break

        # Frames
        frames: list[Molecule] = []
        pbcs: list[Optional[PBC]] = []
        coords_size = n_atoms * 4
        # A header frame count of 0 (never updated) means: read what is there.
        n_expected = hdr["n_frames"] if hdr["n_frames"] > 0 else None
        fi = -1
        while n_expected is None or fi + 1 < n_expected:
            fi += 1
            frame_start = fh.tell()
            try:
                # A job still running or killed leaves a short last frame, or fewer
                # frames than the header says: keep the complete ones.
                for _ in range(4 if hdr["has_box"] else 3):
                    _read_record(fh, endian)
            except EOFError:
                break
            fh.seek(frame_start)
            pbc: Optional[PBC] = None
            if hdr["has_box"]:
                box_payload = _read_record(fh, endian)
                if len(box_payload) != 48:
                    raise ValueError(
                        f"Frame {fi}: box record should be 48 bytes, "
                        f"got {len(box_payload)}"
                    )
                # CHARMM/NAMD order: a, gamma, b, beta, alpha, c
                a, gamma, b, beta, alpha, c = struct.unpack(endian + "6d", box_payload)
                # Some versions store cosines instead of angles (-1..1)
                if -1.0 <= alpha <= 1.0 and -1.0 <= beta <= 1.0 and -1.0 <= gamma <= 1.0:
                    alpha = float(np.degrees(np.arccos(alpha)))
                    beta = float(np.degrees(np.arccos(beta)))
                    gamma = float(np.degrees(np.arccos(gamma)))
                if a > 0 and b > 0 and c > 0:
                    pbc = PBC(a, b, c, alpha, beta, gamma)

            x_payload = _read_record(fh, endian)
            if len(x_payload) != coords_size:
                raise ValueError(
                    f"Frame {fi}: x-record should be {coords_size} bytes, "
                    f"got {len(x_payload)}"
                )
            xs = np.frombuffer(x_payload, dtype=endian + "f4")
            ys = np.frombuffer(_read_record(fh, endian), dtype=endian + "f4")
            zs = np.frombuffer(_read_record(fh, endian), dtype=endian + "f4")

            atoms = [
                Atom(float(xs[i]), float(ys[i]), float(zs[i]), elements[i])
                for i in range(n_atoms)
            ]
            set_atom_ids(atoms)
            mol = Molecule(atoms=atoms, pbc=pbc)
            if topology_missing:
                # Tag the molecule so UI/CLI can surface a warning to users
                mol._dcd_topology_missing = True
            frames.append(mol)
            pbcs.append(pbc)

            if progress_callback and hdr["n_frames"] > 0:
                progress_callback((fi + 1) / hdr["n_frames"])

    if not frames and hdr["n_frames"] > 0:
        raise ValueError("The DCD file holds no complete frame")
    return frames, pbcs


def check_dcd_trajectory(filepath: str) -> bool:
    """Return True if the file is a parseable DCD with at least one frame.
    Does not load full data — just probes the header."""
    try:
        with open(filepath, "rb") as fh:
            endian = _detect_endian(fh)
            header_payload = _read_record(fh, endian)
            hdr = _parse_header(header_payload, endian)
            if hdr["n_frames"] >= 1:
                return True
            # frame count never written to the header: is there data after it?
            _read_record(fh, endian)  # titles
            _read_record(fh, endian)  # atom count
            return len(fh.read(4)) == 4
    except Exception:
        return False
