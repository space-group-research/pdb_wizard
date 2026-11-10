"""Multi-frame files (PDB, XYZ, DCD) for the viewers."""

from __future__ import annotations

from typing import Callable, List, Optional

from pdb_wizard.molecule import Molecule


def is_trajectory_file(filepath: str) -> bool:
    """True for multi-frame PDB/XYZ files and readable DCD files."""
    from pdb_wizard.io import check_pdb_trajectory, check_xyz_trajectory, detect_filetype
    ft = detect_filetype(filepath)
    if ft == "pdb":
        return check_pdb_trajectory(filepath)
    if ft == "xyz":
        return check_xyz_trajectory(filepath)
    if ft == "dcd":
        from pdb_wizard.dcd import check_dcd_trajectory
        return check_dcd_trajectory(filepath)
    return False


def read_trajectory(filepath: str,
                    progress: Optional[Callable[[float], None]] = None) -> List[Molecule]:
    """All frames of a PDB, XYZ or DCD trajectory."""
    from pdb_wizard.io import detect_filetype, read_pdb_trajectory, read_xyz_trajectory
    ft = detect_filetype(filepath)
    if ft == "dcd":
        from pdb_wizard.dcd import read_dcd_trajectory
        mols, _ = read_dcd_trajectory(filepath, progress_callback=progress)
        return mols
    with open(filepath) as f:
        if ft == "pdb":
            mols, _ = read_pdb_trajectory(f, progress_callback=progress)
        else:
            mols, _ = read_xyz_trajectory(f, progress_callback=progress)
    return mols


def trajectory_caveats(frames: Optional[list]) -> List[str]:
    """Warnings about anything the reader had to assume or fake."""
    if not frames:
        return []
    from pdb_wizard.io import NO_CELL_FALLBACK_LENGTH, is_fallback_cell
    first = frames[0]
    notes = []
    # DCD without a topology file: every atom is a carbon placeholder.
    if getattr(first, "_dcd_topology_missing", False):
        notes.append("DCD has no element info. All atoms loaded as carbon. "
                     "Place a sibling .pdb or .xyz with the same atom count "
                     "next to the .dcd file.")
    # No cell in the file: the reader substituted a huge non-periodic box.
    if is_fallback_cell(first.pbc):
        notes.append(f"Trajectory has no cell information; assuming a "
                     f"{NO_CELL_FALLBACK_LENGTH:.0f} Å box (non-periodic). "
                     "Use Update Cell if the system is periodic.")
    return notes


def frame_bonds(frames: List[Molecule], idx: int) -> None:
    """Give frame ``idx`` bonds: frame 0's when the topology matches, else detect them."""
    mol = frames[idx]
    if mol.bonds or not mol.atoms:
        return
    if frames[0].bonds and len(frames[0].atoms) == len(mol.atoms):
        mol.bonds = frames[0].bonds
    else:
        mol.detect_bonds()
