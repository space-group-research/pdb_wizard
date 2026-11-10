"""MPMC sorbate molecule models from the MPMC repository.

Two categories:
  - Nonpolar LJ: Standard Lennard-Jones sigma/epsilon. No polarizability.
  - Polarizable LJ: LJ sigma/epsilon with explicit point polarizabilities.
    Used with many-body polarization turned on in MPMC.

Note: True PHAHST (damped exponential repulsion + C6/C8/C10 dispersion) sorbate
models are not included here — they require parameters specifically fitted for
that potential form. The PHAHST MOF force field parameters are in forcefields.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy import cos, pi, sin, sqrt

from pdb_wizard.io import _expand_numeric_tokens, require_cell
from pdb_wizard.molecule import Molecule


@dataclass
class SorbateSite:
    atomtype: str
    x: float
    y: float
    z: float
    mass: float
    charge: float
    polarizability: float
    epsilon: float  # LJ epsilon (K) or exp-repulsion beta (A^-1)
    sigma: float    # LJ sigma (A) or exp-repulsion rho (A)
    c6: float = 0.0
    c8: float = 0.0
    c10: float = 0.0


@dataclass
class SorbateModel:
    name: str
    label: str
    description: str
    pot_type: str  # "lj", "lj_polar", or "phahst"
    sites: list[SorbateSite] = field(default_factory=list)


def _s(atomtype: str, x: float, y: float, z: float,
       mass: float, charge: float, pol: float, eps: float, sig: float) -> SorbateSite:
    return SorbateSite(atomtype, x, y, z, mass, charge, pol, eps, sig)


def _ph(atomtype: str, x: float, y: float, z: float,
        mass: float, charge: float, pol: float,
        beta: float, rho: float,
        c6: float, c8: float, c10: float) -> SorbateSite:
    return SorbateSite(atomtype, x, y, z, mass, charge, pol, beta, rho, c6, c8, c10)


SORBATE_MODELS: dict[str, SorbateModel] = {
    # =========================================================================
    # Nonpolar LJ models
    # =========================================================================

    # --- Hydrogen ---
    "h2_bss": SorbateModel("h2_bss", "H2", "H2 BSS 5-site", "lj", [
        _s("H2G", 0, 0, 0, 0, -0.7464, 0, 8.8516, 3.2293),
        _s("H2E", 0.371, 0, 0, 1.008, 0.3732, 0, 0, 0),
        _s("H2E", -0.371, 0, 0, 1.008, 0.3732, 0, 0, 0),
        _s("H2N", 0.329, 0, 0, 0, 0, 0, 4.0659, 2.3406),
        _s("H2N", -0.329, 0, 0, 0, 0, 0, 4.0659, 2.3406),
    ]),
    "h2_buch": SorbateModel("h2_buch", "H2", "H2 Buch 1-site", "lj", [
        _s("H2G", 0, 0, 0, 2.016, 0, 0, 34.2, 2.96),
    ]),
    "h2_dl": SorbateModel("h2_dl", "H2", "H2 Darkrim-Levesque 3-site", "lj", [
        _s("H2G", 0, 0, 0, 0, -0.936, 0, 36.7, 2.958),
        _s("H2E", -0.37, 0, 0, 1.008, 0.468, 0, 0, 0),
        _s("H2E", 0.37, 0, 0, 1.008, 0.468, 0, 0, 0),
    ]),
    # --- CO2 ---
    "co2_trappe": SorbateModel("co2_trappe", "CO2", "CO2 TraPPE 3-site", "lj", [
        _s("COG", 0, 0, 0, 12.01, 0.7, 0, 27.0, 2.80),
        _s("COE", 1.16, 0, 0, 16.0, -0.35, 0, 79.0, 3.05),
        _s("COE", -1.16, 0, 0, 16.0, -0.35, 0, 79.0, 3.05),
    ]),
    "co2_epm2": SorbateModel("co2_epm2", "CO2", "CO2 EPM2 Harris-Yung 3-site", "lj", [
        _s("COG", 0, 0, 0, 12.01, 0.6512, 0, 28.129, 2.757),
        _s("COE", 1.149, 0, 0, 16.0, -0.3256, 0, 80.507, 3.033),
        _s("COE", -1.149, 0, 0, 16.0, -0.3256, 0, 80.507, 3.033),
    ]),
    "co2_phast": SorbateModel("co2_phast", "CO2", "CO2 PHAST 5-site", "lj", [
        _s("COG", 0, 0, 0, 12.0107, 0.77106, 0, 8.52238, 3.05549),
        _s("COE", 1.162, 0, 0, 15.9994, -0.38553, 0, 0, 0),
        _s("COE", -1.162, 0, 0, 15.9994, -0.38553, 0, 0, 0),
        _s("CON", 1.091, 0, 0, 0, 0, 0, 76.76607, 2.94473),
        _s("CON", -1.091, 0, 0, 0, 0, 0, 76.76607, 2.94473),
    ]),
    # --- Methane ---
    "ch4_trappe": SorbateModel("ch4_trappe", "CH4", "CH4 TraPPE 1-site", "lj", [
        _s("CHG", 0, 0, 0, 16.0426, 0, 0, 148.0, 3.73),
    ]),
    "ch4_9site": SorbateModel("ch4_9site", "CH4", "CH4 9-site", "lj", [
        _s("CHG", 0, 0, 0, 12.011, -0.5868, 0, 58.53869, 2.22416),
        _s("CHE", 0, 0, 1.099, 1.0079, 0.1467, 0, 0, 0),
        _s("CHE", 1.036, 0, -0.366, 1.0079, 0.1467, 0, 0, 0),
        _s("CHE", -0.518, -0.897, -0.366, 1.0079, 0.1467, 0, 0, 0),
        _s("CHE", -0.518, 0.897, -0.366, 1.0079, 0.1467, 0, 0, 0),
        _s("MOV", 0, 0, 0.816, 0, 0, 0, 16.85422, 2.96286),
        _s("MOV", 0.769, 0, -0.271, 0, 0, 0, 16.85422, 2.96286),
        _s("MOV", -0.385, -0.668, -0.271, 0, 0, 0, 16.85422, 2.96286),
        _s("MOV", -0.385, 0.668, -0.271, 0, 0, 0, 16.85422, 2.96286),
    ]),
    # --- Nitrogen ---
    "n2_trappe": SorbateModel("n2_trappe", "N2", "N2 TraPPE 3-site", "lj", [
        _s("N2G", 0, 0, 0, 0, 0.964, 0, 0, 0),
        _s("N2E", 0.55, 0, 0, 14.0067, -0.482, 0, 36.0, 3.31),
        _s("N2E", -0.55, 0, 0, 14.0067, -0.482, 0, 36.0, 3.31),
    ]),
    "n2_mcquarrie": SorbateModel("n2_mcquarrie", "N2", "N2 McQuarrie 1-site", "lj", [
        _s("N2G", 0, 0, 0, 28.01344, 0, 0, 95.1, 3.7),
    ]),
    # --- Water ---
    "h2o_tip3p": SorbateModel("h2o_tip3p", "H2O", "H2O TIP3P 3-site", "lj", [
        _s("OXY", 0, 0, 0, 16.0, -0.834, 0, 76.42, 3.151),
        _s("HYD", -0.757, -0.586, 0, 1.008, 0.417, 0, 0, 0),
        _s("HYD", 0.757, -0.586, 0, 1.008, 0.417, 0, 0, 0),
    ]),
    "h2o_tip4p": SorbateModel("h2o_tip4p", "H2O", "H2O TIP4P 4-site", "lj", [
        _s("OXY", 0, 0, 0, 16.0, 0, 0, 78.0, 3.154),
        _s("HYD", 0.58588, 0.75695, 0, 1.008, 0.52, 0, 0, 0),
        _s("HYD", 0.58588, -0.75695, 0, 1.008, 0.52, 0, 0, 0),
        _s("M", 0.15, 0, 0, 0, -1.04, 0, 0, 0),
    ]),
    # Berendsen et al., J. Phys. Chem. 91, 6269 (1987): r_OH 1.0 A, HOH 109.47 deg
    "h2o_spce": SorbateModel("h2o_spce", "H2O", "H2O SPC/E 3-site", "lj", [
        _s("OXY", 0, 0, 0, 15.9994, -0.8476, 0, 78.2, 3.166),
        _s("HYD", 0.81650, -0.57735, 0, 1.008, 0.4238, 0, 0, 0),
        _s("HYD", -0.81650, -0.57735, 0, 1.008, 0.4238, 0, 0, 0),
    ]),
    # Mahoney & Jorgensen, J. Chem. Phys. 112, 8910 (2000): r_OL 0.70 A, LOL 109.47 deg
    "h2o_tip5p": SorbateModel("h2o_tip5p", "H2O", "H2O TIP5P 5-site", "lj", [
        _s("OXY", 0, 0, 0, 15.9994, 0, 0, 80.51, 3.12),
        _s("HYD", 0.75695, -0.58588, 0, 1.008, 0.241, 0, 0, 0),
        _s("HYD", -0.75695, -0.58588, 0, 1.008, 0.241, 0, 0, 0),
        _s("LP", 0, 0.40415, 0.57155, 0, -0.241, 0, 0, 0),
        _s("LP", 0, 0.40415, -0.57155, 0, -0.241, 0, 0, 0),
    ]),
    # --- Noble gases ---
    "he": SorbateModel("he", "He", "Helium pore volume probe", "lj", [
        _s("He", 0, 0, 0, 4.002602, 0, 0, 10.22, 2.28),
    ]),
    # --- O2 ---
    "o2_trappe": SorbateModel("o2_trappe", "O2", "O2 TraPPE 3-site", "lj", [
        _s("O", -0.605, 0, 0, 7.9997, -0.113, 0, 49.0, 3.02),
        _s("CoM", 0, 0, 0, 0, 0.226, 0, 0, 0),
        _s("O", 0.605, 0, 0, 7.9997, -0.113, 0, 49.0, 3.02),
    ]),

    # =========================================================================
    # Polarizable LJ models (LJ + explicit point polarizabilities)
    # =========================================================================

    # --- Hydrogen ---
    "h2_bssp": SorbateModel("h2_bssp", "H2", "H2 BSSP 5-site polarizable", "lj_polar", [
        _s("H2G", 0, 0, 0, 0, -0.7464, 0.6938, 12.76532, 3.15528),
        _s("H2E", 0.371, 0, 0, 1.008, 0.3732, 0.00044, 0, 0),
        _s("H2E", -0.371, 0, 0, 1.008, 0.3732, 0.00044, 0, 0),
        _s("H2N", 0.363, 0, 0, 0, 0, 0, 2.16726, 2.37031),
        _s("H2N", -0.363, 0, 0, 0, 0, 0, 2.16726, 2.37031),
    ]),
    # --- CO2 ---
    "co2_phastp": SorbateModel("co2_phastp", "CO2", "CO2 PHASTP 5-site polarizable", "lj_polar", [
        _s("COG", 0, 0, 0, 12.0107, 0.77134, 1.2281, 19.61757, 3.03366),
        _s("COE", 1.162, 0, 0, 15.9994, -0.38567, 0.7395, 0, 0),
        _s("COE", -1.162, 0, 0, 15.9994, -0.38567, 0.7395, 0, 0),
        _s("CON", 1.208, 0, 0, 0, 0, 0, 46.47457, 2.99429),
        _s("CON", -1.208, 0, 0, 0, 0, 0, 46.47457, 2.99429),
    ]),
    "co2_becker": SorbateModel("co2_becker", "CO2", "CO2 Becker 3-site polarizable", "lj_polar", [
        _s("COG", 0, 0, 0, 12.01, 0.7, 0.916, 23.4, 2.8),
        _s("COE", 1.16, 0, 0, 16.0, -0.35, 0.575, 73.08, 3.05),
        _s("COE", -1.16, 0, 0, 16.0, -0.35, 0.575, 73.08, 3.05),
    ]),
    # --- Methane ---
    "ch4_9sitep": SorbateModel("ch4_9sitep", "CH4", "CH4 9-site polarizable", "lj_polar", [
        _s("CHG", 0, 0, 0, 12.011, -0.5868, 1.0987, 45.0973, 2.16247),
        _s("CHE", 0, 0, 1.099, 1.0079, 0.1467, 0.4246, 0, 0),
        _s("CHE", 1.036, 0, -0.366, 1.0079, 0.1467, 0.4246, 0, 0),
        _s("CHE", -0.518, -0.897, -0.366, 1.0079, 0.1467, 0.4246, 0, 0),
        _s("CHE", -0.518, 0.897, -0.366, 1.0079, 0.1467, 0.4246, 0, 0),
        _s("MOV", 0, 0, 0.814, 0, 0, 0, 18.57167, 2.94787),
        _s("MOV", 0.768, 0, -0.270, 0, 0, 0, 18.57167, 2.94787),
        _s("MOV", -0.383, -0.666, -0.270, 0, 0, 0, 18.57167, 2.94787),
        _s("MOV", -0.383, 0.666, -0.270, 0, 0, 0, 18.57167, 2.94787),
    ]),
    # --- Nitrogen ---
    "n2_polar": SorbateModel("n2_polar", "N2", "N2 5-site polarizable", "lj_polar", [
        _s("N2G", 0, 0, 0, 0, 1.04742, 1.4559, 20.6365, 3.42344),
        _s("N2E", 0.549, 0, 0, 14.0067, -0.52371, 0.5138, 0, 0),
        _s("N2E", -0.549, 0, 0, 14.0067, -0.52371, 0.5138, 0, 0),
        _s("N2N", 0.738, 0, 0, 0, 0, 0, 18.12772, 3.15125),
        _s("N2N", -0.738, 0, 0, 0, 0, 0, 18.12772, 3.15125),
    ]),
    # --- Water ---
    "h2o_pol3": SorbateModel("h2o_pol3", "H2O", "H2O POL3 3-site polarizable", "lj_polar", [
        _s("OXY", 0, 0, 0, 15.999, -0.73, 0.528, 78.50225, 3.596),
        _s("HYD", -0.816, -0.577, 0, 1.0079, 0.365, 0.17, 0, 0),
        _s("HYD", 0.816, -0.577, 0, 1.0079, 0.365, 0.17, 0, 0),
    ]),
    # Dang & Chang, J. Chem. Phys. 106, 8149 (1997): polarizable M site 0.215 A from O
    "h2o_dc97": SorbateModel("h2o_dc97", "H2O", "H2O DC97 4-site polarizable", "lj_polar", [
        _s("OXY", 0, 0, 0, 15.9994, 0, 0, 91.84, 3.234),
        _s("HYD", 0.75695, -0.58588, 0, 1.008, 0.519, 0, 0, 0),
        _s("HYD", -0.75695, -0.58588, 0, 1.008, 0.519, 0, 0, 0),
        _s("M", 0, -0.215, 0, 0, -1.038, 1.444, 0, 0),
    ]),
    # --- Noble gases (polarizable LJ) ---
    "ar_pol": SorbateModel("ar_pol", "Ar", "Argon polarizable (LJ)", "lj_polar", [
        _s("Ar", 0, 0, 0, 39.948, 0, 1.6392212, 128.326802, 3.371914),
    ]),
    "ne_pol": SorbateModel("ne_pol", "Ne", "Neon polarizable (LJ)", "lj_polar", [
        _s("Ne", 0, 0, 0, 20.1797, 0, 0.3913212, 36.824138, 2.785823),
    ]),
    "kr_pol": SorbateModel("kr_pol", "Kr", "Krypton polarizable (LJ)", "lj_polar", [
        _s("Kr", 0, 0, 0, 83.798, 0, 2.5004096, 183.795833, 3.601271),
    ]),
    "xe_pol": SorbateModel("xe_pol", "Xe", "Xenon polarizable (LJ)", "lj_polar", [
        _s("Xe", 0, 0, 0, 131.293, 0, 4.0232578, 237.985247, 3.956802),
    ]),

    # =========================================================================
    # PHAHST models (damped exp repulsion + C6/C8/C10 dispersion)
    # Hogan & Space, JCTC 2020, 16, 7632-7644
    # beta = exp repulsion steepness (A^-1), rho = exp repulsion distance (A)
    # C6/C8/C10 in atomic units (Hartree * Bohr^n)
    # =========================================================================

    # --- Hydrogen ---
    "h2_phahst": SorbateModel("h2_phahst", "H2", "H2 PHAHST 3-site", "phahst", [
        _ph("H2DA", 0, 0, 0, 0, -0.846166, 0, 3.627796, 2.664506, 0, 0, 0),
        _ph("H2H", 0.371, 0, 0, 1.008, 0.423083, 0.34325, 3.100603, 1.859425, 2.884735, 38.97178, 644.95683),
        _ph("H2H", -0.371, 0, 0, 1.008, 0.423083, 0.34325, 3.100603, 1.859425, 2.884735, 38.97178, 644.95683),
    ]),
    # --- Nitrogen ---
    "n2_phahst": SorbateModel("n2_phahst", "N2", "N2 PHAHST 3-site", "phahst", [
        _ph("N2DA", 0, 0, 0, 0, 0.94194, 0, 0, 0, 0, 0, 0),
        _ph("N2N", 0.5507, 0, 0, 14.0067, -0.47103, 0.85092, 3.85368, 3.31513, 17.80503, 416.3235, 11924.913),
        _ph("N2N", -0.5507, 0, 0, 14.0067, -0.47103, 0.85092, 3.85368, 3.31513, 17.80503, 416.3235, 11924.913),
    ]),
    # --- Noble gases ---
    "he_phahst": SorbateModel("he_phahst", "He", "Helium PHAHST", "phahst", [
        _ph("He", 0, 0, 0, 4.0026, 0, 0.2002, 4.68451, 2.38376, 1.40716, 11.13635, 107.964),
    ]),
    "ne_phahst": SorbateModel("ne_phahst", "Ne", "Neon PHAHST", "phahst", [
        _ph("Ne", 0, 0, 0, 20.18, 0, 0.3823, 4.99432, 2.8024, 6.21275, 67.98647, 911.376),
    ]),
    "ar_phahst": SorbateModel("ar_phahst", "Ar", "Argon PHAHST", "phahst", [
        _ph("Ar", 0, 0, 0, 39.948, 0, 1.655, 3.88525, 3.68623, 65.46, 1438.9, 38745.0),
    ]),
    "kr_phahst": SorbateModel("kr_phahst", "Kr", "Krypton PHAHST", "phahst", [
        _ph("Kr", 0, 0, 0, 83.798, 0, 2.497, 3.52894, 4.03477, 130.1, 3981.0, 149225.0),
    ]),
    "xe_phahst": SorbateModel("xe_phahst", "Xe", "Xenon PHAHST", "phahst", [
        _ph("Xe", 0, 0, 0, 131.293, 0, 4.026, 3.24691, 4.47518, 288.4, 11390.0, 551047.0),
    ]),
}


def get_sorbate_names() -> list[tuple[str, str, str]]:
    """Return list of (model_name, description, pot_type) for all sorbates."""
    return [(name, m.description, m.pot_type) for name, m in SORBATE_MODELS.items()]


def format_sorbate_pqr(model_name: str, center_x: float, center_y: float,
                       center_z: float, mol_id: int, start_atom_id: int) -> list[str]:
    """Generate MPMC PQR lines for a sorbate placed at the given center coordinates."""
    model = SORBATE_MODELS[model_name]
    return [
        _sorbate_atom_line(model, site, start_atom_id + i, mol_id,
                           (center_x + site.x, center_y + site.y, center_z + site.z))
        for i, site in enumerate(model.sites)
    ]


def _sorbate_atom_line(model: SorbateModel, site: SorbateSite, atom_id: int,
                       mol_id: int, xyz) -> str:
    """One movable (M) MPMC atom record for a sorbate site at cartesian ``xyz``."""
    x, y, z = xyz
    return (
        f"ATOM {atom_id:>6} {site.atomtype:<4} {model.label:<3} M {mol_id:>4}    "
        f"{x:>10.6f} {y:>10.6f} {z:>10.6f}"
        f" {site.mass:>9.5f} {site.charge:>9.5f}"
        f" {site.polarizability:>9.5f} {site.epsilon:>9.5f} {site.sigma:>9.5f}"
        f" 0.0 0.0 {site.c6:>10.5f} {site.c8:>12.5f} {site.c10:>14.5f}"
    )


# ======================================================================
# Module: insert
# ======================================================================
"""Pack rigid sorbate molecules into a periodic cell (a small packmol for MPMC).

Molecules get a random position and a uniformly random orientation. A trial is
kept only if no site lands too close to a framework atom or to an already
placed molecule, measured with the minimum image in the actual (triclinic)
cell, so skewed and monoclinic cells work. "Too close" for a site pair is

    d < max(min_dist, scale * (r_i + r_j))

where r is sigma/2 for sites with LJ parameters, the element vdW radius for
framework atoms that have no sigma yet, and 0 for charge-only sites (H, M, LP),
which are then only kept min_dist away.
"""


def read_sorbate_template(path: str | Path) -> SorbateModel:
    """First movable (M) molecule of an MPMC .pqr, as a SorbateModel.

    Lets users insert any model MPMC can read, not just the built-in ones.
    Raises ValueError when the file has no M molecule.
    """
    sites: list[SorbateSite] = []
    label, mol_id = "MOL", None
    for line in Path(path).read_text().splitlines():
        tok = _expand_numeric_tokens(line.split())
        if len(tok) < 14 or tok[0] not in ("ATOM", "HETATM") or tok[4] != "M":
            continue
        if mol_id is None:
            mol_id, label = tok[5], tok[3]
        elif tok[5] != mol_id:
            break
        c = [float(v) for v in tok[16:19]] if len(tok) > 18 else [0.0, 0.0, 0.0]
        sites.append(SorbateSite(tok[2], float(tok[6]), float(tok[7]), float(tok[8]),
                                 float(tok[9]), float(tok[10]), float(tok[11]),
                                 float(tok[12]), float(tok[13]), *c))
    if not sites:
        raise ValueError(f"{path}: no movable (M) molecule found")
    pot = "phahst" if any(s.c6 for s in sites) else (
        "lj_polar" if any(s.polarizability for s in sites) else "lj")
    return SorbateModel(Path(path).stem, label, f"from {Path(path).name}", pot, sites)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix (Shoemake 1992)."""
    u1, u2, u3 = rng.random(3)
    a, b = sqrt(1.0 - u1), sqrt(u1)
    x, y = a * sin(2 * pi * u2), a * cos(2 * pi * u2)
    z, w = b * sin(2 * pi * u3), b * cos(2 * pi * u3)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def insert_sorbates(
    mol: Molecule,
    model: SorbateModel | str,
    n: int | None = None,
    *,
    scale: float = 0.85,
    min_dist: float = 1.0,
    max_failures: int = 5000,
    seed: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[np.ndarray]:
    """Place up to ``n`` copies of ``model`` in ``mol``'s cell without overlaps.

    ``n=None`` fills the cell: insertion stops after ``max_failures`` rejected
    trials in a row. With a number it also stops there, so fewer than ``n``
    may fit; check ``len()`` of the result. Returns one (n_sites, 3) array of
    cartesian site positions per placed molecule. ``mol`` is not modified.
    Raises CellRequiredError when ``mol`` has no cell.
    """
    require_cell(mol, "Inserting sorbates")
    if isinstance(model, str):
        model = SORBATE_MODELS[model]
    pbc = mol.pbc
    basis, recip = pbc.basis_matrix, pbc.reciprocal_basis_matrix
    rng = np.random.default_rng(seed)

    sites = np.array([[s.x, s.y, s.z] for s in model.sites], dtype=float)
    masses = np.array([s.mass for s in model.sites], dtype=float)
    center = (masses @ sites / masses.sum()) if masses.sum() > 0 else sites.mean(axis=0)
    sites -= center
    site_r = np.array([s.sigma / 2 if s.sigma > 0 and model.pot_type != "phahst" else 0.0
                       for s in model.sites])

    fw_xyz = np.array([a.x for a in mol.atoms], dtype=float).reshape(-1, 3)
    fw_r = np.array([a.sigma / 2 if a.sigma > 0 else a.vdw for a in mol.atoms], dtype=float)
    # Sample centers in the same periodic image as the framework (e.g. a cell
    # stored from -1/2 to 1/2), so molecules show up inside it in the viewer.
    # Molecules are kept whole: never wrap their atoms one by one.
    window = (np.mean(fw_xyz @ recip, axis=0) - 0.5) if len(fw_xyz) else np.zeros(3)

    placed: list[np.ndarray] = []
    others_xyz, others_r = fw_xyz, fw_r
    failures = 0
    while (n is None or len(placed) < n) and failures < max_failures:
        trial = sites @ _random_rotation(rng).T + (window + rng.random(3)) @ basis
        if len(others_xyz):
            d = trial[:, None, :] - others_xyz[None, :, :]
            d -= np.round(d @ recip) @ basis
            dist = np.sqrt((d * d).sum(axis=2))
            limit = np.maximum(min_dist, scale * (site_r[:, None] + others_r[None, :]))
            if (dist < limit).any():
                failures += 1
                continue
        placed.append(trial)
        others_xyz = np.vstack([others_xyz, trial])
        others_r = np.concatenate([others_r, site_r])
        failures = 0
        if progress_callback is not None:
            progress_callback(len(placed))
    return placed


def format_inserted_sorbates(model: SorbateModel | str, placements: list[np.ndarray],
                             start_atom_id: int, start_mol_id: int = 2) -> list[str]:
    """MPMC atom records for ``insert_sorbates`` output, one molecule ID per copy.

    Pass the result to ``write_mpmc_pdb(..., sorbate_lines=...)``.
    """
    if isinstance(model, str):
        model = SORBATE_MODELS[model]
    lines, atom_id = [], start_atom_id
    for k, xyz in enumerate(placements):
        if len(xyz) != len(model.sites):
            raise ValueError(f"placement {k} has {len(xyz)} sites, {model.name} has {len(model.sites)}")
        for site, pos in zip(model.sites, xyz):  # no strict=: must run on Python 3.7
            lines.append(_sorbate_atom_line(model, site, atom_id, start_mol_id + k, pos))
            atom_id += 1
    return lines
