"""Generate a small OpenMM DCD fixture for the DCD reader tests.

Builds a tiny water box (3 H2O), runs 100 steps of NPT, saves a DCD with
20 frames + a sibling .pdb topology. Output goes into tests/fixtures/.

Run with any OpenMM-equipped env, e.g.:

    /home/user/.conda-envs/smee/bin/python scripts/make_openmm_dcd_fixture.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    System,
    Vec3,
)
from openmm.app import (
    PME,
    DCDReporter,
    Element,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    Simulation,
    Topology,
)
from openmm.unit import (
    angstrom,
    atmospheres,
    bar,
    femtoseconds,
    kelvin,
    nanometer,
    picosecond,
    picoseconds,
)


OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDB = OUT_DIR / "openmm_water.pdb"
OUT_DCD = OUT_DIR / "openmm_water.dcd"


def main() -> None:
    # Build 3 water molecules in a 1.5 nm cubic box manually so we don't
    # need any external file.
    topology = Topology()
    chain = topology.addChain()
    O_e = Element.getBySymbol("O")
    H_e = Element.getBySymbol("H")

    coords_nm = []
    for w in range(3):
        residue = topology.addResidue(f"HOH", chain)
        ox = topology.addAtom("O1", O_e, residue)
        h1 = topology.addAtom("H1", H_e, residue)
        h2 = topology.addAtom("H2", H_e, residue)
        topology.addBond(ox, h1)
        topology.addBond(ox, h2)
        # Place each water roughly along x, y at the same z
        x = 0.3 + w * 0.4
        coords_nm.append(Vec3(x, 0.5, 0.5))                 # O
        coords_nm.append(Vec3(x + 0.0957, 0.5, 0.5))         # H1
        coords_nm.append(Vec3(x - 0.0240, 0.5928, 0.5))      # H2
    topology.setUnitCellDimensions(Vec3(1.5, 1.5, 1.5))

    # Force field — TIP3P
    ff = ForceField("amber14/tip3p.xml")
    modeller = Modeller(topology, [c * nanometer for c in coords_nm])
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.6 * nanometer,
        constraints=HBonds,
    )
    # Barostat for NPT
    system.addForce(MonteCarloBarostat(1.0 * bar, 300 * kelvin, 25))

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 1.0 * femtoseconds,
    )
    platform = Platform.getPlatformByName("Reference")
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Save a sibling PDB for topology lookup by the DCD reader
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=False)
    with open(OUT_PDB, "w") as f:
        PDBFile.writeFile(simulation.topology, positions, f)

    # 20 frames @ 5-step interval = 100 total steps
    simulation.reporters.append(DCDReporter(str(OUT_DCD), 5))
    simulation.step(100)

    print(f"wrote {OUT_DCD} ({OUT_DCD.stat().st_size} bytes)")
    print(f"wrote {OUT_PDB} ({OUT_PDB.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
