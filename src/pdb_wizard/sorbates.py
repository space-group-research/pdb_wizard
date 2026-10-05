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
    lines = []
    for i, site in enumerate(model.sites):
        atom_id = start_atom_id + i
        x = center_x + site.x
        y = center_y + site.y
        z = center_z + site.z
        lines.append(
            f"ATOM {atom_id:>6} {site.atomtype:<4} {model.label:<3} M {mol_id:>4}    "
            f"{x:>10.6f} {y:>10.6f} {z:>10.6f}"
            f" {site.mass:>9.5f} {site.charge:>9.5f}"
            f" {site.polarizability:>9.5f} {site.epsilon:>9.5f} {site.sigma:>9.5f}"
            f" 0.0 0.0 {site.c6:>10.5f} {site.c8:>12.5f} {site.c10:>14.5f}"
        )
    return lines
