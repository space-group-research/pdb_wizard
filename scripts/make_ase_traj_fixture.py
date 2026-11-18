"""Write the ASE trajectory fixtures used by tests/test_ase_traj.py.

Needs ASE (any environment that has it); pdb_wizard itself reads these files without ASE.
Run from the repo root:  python scripts/make_ase_traj_fixture.py
The numbers this prints are the expected values hard-coded in the tests.
"""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def water_md():
    """Three frames of one water molecule, no cell, with calculator charges."""
    base = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]])
    path = OUT / "ase_water_md.traj"
    with Trajectory(str(path), "w") as traj:
        for k in range(3):
            atoms = Atoms("OH2", positions=base + 0.1 * k)
            charges = np.array([-0.8, 0.4, 0.4]) + 0.01 * k * np.array([-2, 1, 1])
            atoms.calc = SinglePointCalculator(atoms, energy=-1.0 * k, charges=charges)
            traj.write(atoms)
    return path


def periodic_rotated():
    """Two frames of NaCl in a cell that is NOT in the a-along-x orientation and that
    changes between frames, so the reader has to carry atoms over by fractional
    coordinates."""
    path = OUT / "ase_periodic_rotated.traj"
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.1, 0.9]])
    with Trajectory(str(path), "w") as traj:
        for scale in (1.0, 1.05):
            cell = scale * np.array([[0.0, 4.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 0.0]])
            atoms = Atoms("NaClCu", scaled_positions=frac, cell=cell, pbc=True)
            traj.write(atoms)
    return path


if __name__ == "__main__":
    for make in (water_md, periodic_rotated):
        path = make()
        frames = Trajectory(str(path))
        print(path.name, len(frames), "frames,", path.stat().st_size, "bytes")
        for atoms in frames:
            print("  ", atoms.get_chemical_formula(), np.round(atoms.cell.cellpar(), 4).tolist(),
                  np.round(atoms.get_scaled_positions(wrap=False), 4).tolist()
                  if atoms.pbc.any() else np.round(atoms.positions, 4).tolist())
