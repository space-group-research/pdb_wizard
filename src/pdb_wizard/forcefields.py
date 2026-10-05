"""Force field parameter data (OPLS-AA/UFF and PHAHST).

UFF vdW parameters from: Rappe et al., JACS 114, 10024-10035 (1992), Table I.
sigma = x_i / 2^(1/6), epsilon = D_i / k_B  (epsilon in K, sigma in Angstroms).
H, C, N use OPLS-AA values; all others are pure UFF.

Polarizabilities (alpha, Angstrom^3):
  - Non-metals: van Duijnen & Swart, JPCA 102, 2399 (1998).
  - Metal ions: Shannon, J. Appl. Phys. 73, 348-366 (1993), Table I.
    Dielectric polarizabilities alpha_D for common MOF oxidation states.
"""

from __future__ import annotations


def _uff(x_i: float, d_i: float, alpha: float = 0.0) -> dict[str, float]:
    """Convert UFF r_min (x_i, Ang) and well depth (d_i, kcal/mol) to sigma/epsilon."""
    sigma = x_i / 1.12246204830937  # 2^(1/6)
    epsilon = d_i / 0.0019872041     # kB in kcal/(mol*K)
    return {"alpha": alpha, "sigma": sigma, "epsilon": epsilon, "c6": 0.0, "c8": 0.0, "c10": 0.0}


def get_forcefield(name: int) -> dict[str, dict[str, float]]:
    ffs: list[dict[str, dict[str, float]]] = []

    # --- OPLS-AA/UFF hybrid ---
    # H, C, N from OPLS-AA; everything else from UFF (Rappe 1992)
    opls_aa_uff: dict[str, dict[str, float]] = {
        # OPLS-AA overrides
        "H":  {"alpha": 0.41380, "sigma": 2.42,    "epsilon": 15.11,    "c6": 0.0, "c8": 0.0, "c10": 0.0},
        "C":  {"alpha": 1.28660, "sigma": 3.55,    "epsilon": 35.25,    "c6": 0.0, "c8": 0.0, "c10": 0.0},
        "N":  {"alpha": 0.97157, "sigma": 3.25,    "epsilon": 85.60,    "c6": 0.0, "c8": 0.0, "c10": 0.0},
        # UFF — Period 1-2
        # alpha sources: non-metals = van Duijnen & Swart 1998; metals = Shannon 1993
        "He": _uff(2.362, 0.056),
        "Li": _uff(2.451, 0.025, alpha=0.029),    # Shannon Li+
        "Be": _uff(2.745, 0.085, alpha=0.008),    # Shannon Be2+
        "B":  _uff(4.083, 0.180, alpha=0.003),    # Shannon B3+
        "O":  _uff(3.500, 0.060, alpha=0.852),    # van Duijnen
        "F":  _uff(3.364, 0.050, alpha=0.444747), # van Duijnen
        "Ne": _uff(3.243, 0.042),
        # Period 3
        "Na": _uff(2.983, 0.030, alpha=0.179),    # Shannon Na+
        "Mg": _uff(3.021, 0.111, alpha=1.32),     # Shannon Mg2+
        "Al": _uff(4.499, 0.505, alpha=0.79),     # Shannon Al3+
        "Si": _uff(4.295, 0.402, alpha=0.87),     # Shannon Si4+
        "P":  _uff(4.147, 0.305, alpha=0.021),    # Shannon P5+
        "S":  _uff(4.035, 0.274, alpha=2.474448), # van Duijnen
        "Cl": _uff(3.947, 0.227, alpha=2.40028),  # van Duijnen
        "Ar": _uff(3.868, 0.185),
        # Period 4
        "K":  _uff(3.812, 0.035, alpha=0.83),     # Shannon K+
        "Ca": _uff(3.399, 0.238, alpha=3.16),     # Shannon Ca2+
        "Sc": _uff(3.295, 0.019, alpha=2.81),     # Shannon Sc3+
        "Ti": _uff(3.175, 0.017, alpha=2.93),     # Shannon Ti4+
        "V":  _uff(3.144, 0.016, alpha=2.92),     # Shannon V5+
        "Cr": _uff(3.023, 0.015, alpha=1.45),     # Shannon Cr3+
        "Mn": _uff(2.961, 0.013, alpha=1.64),     # Shannon Mn2+
        "Fe": _uff(2.912, 0.013, alpha=2.29),     # Shannon Fe3+
        "Co": _uff(2.872, 0.014, alpha=1.65),     # Shannon Co2+
        "Ni": _uff(2.834, 0.015, alpha=1.23),     # Shannon Ni2+
        "Cu": _uff(3.495, 0.005, alpha=2.11),     # Shannon Cu2+
        "Zn": _uff(2.763, 0.124, alpha=2.04),     # Shannon Zn2+
        "Ga": _uff(4.383, 0.415, alpha=1.50),     # Shannon Ga3+
        "Ge": _uff(4.280, 0.379, alpha=1.63),     # Shannon Ge4+
        "As": _uff(4.230, 0.309, alpha=0.47),     # Shannon As5+
        "Se": _uff(4.205, 0.291, alpha=3.29),     # Shannon Se4+
        "Br": _uff(4.189, 0.251, alpha=3.493),    # van Duijnen
        "Kr": _uff(4.141, 0.220),
        # Period 5
        "Rb": _uff(4.114, 0.040, alpha=1.40),     # Shannon Rb+
        "Sr": _uff(3.641, 0.235, alpha=4.24),     # Shannon Sr2+
        "Y":  _uff(3.345, 0.072, alpha=3.81),     # Shannon Y3+
        "Zr": _uff(3.124, 0.069, alpha=3.25),     # Shannon Zr4+
        "Nb": _uff(3.165, 0.059, alpha=3.97),     # Shannon Nb5+
        "Mo": _uff(3.052, 0.056, alpha=3.28),     # Shannon Mo6+
        "Tc": _uff(2.998, 0.048),
        "Ru": _uff(2.963, 0.056, alpha=1.82),     # Shannon Ru4+
        "Rh": _uff(2.929, 0.053, alpha=1.72),     # Shannon Rh3+
        "Pd": _uff(2.899, 0.048, alpha=1.68),     # Shannon Pd2+
        "Ag": _uff(3.148, 0.036, alpha=2.25),     # Shannon Ag+
        "Cd": _uff(2.848, 0.228, alpha=3.40),     # Shannon Cd2+
        "In": _uff(4.463, 0.599, alpha=2.62),     # Shannon In3+
        "Sn": _uff(4.392, 0.567, alpha=2.83),     # Shannon Sn4+
        "Sb": _uff(4.420, 0.449, alpha=1.64),     # Shannon Sb5+
        "Te": _uff(4.470, 0.398, alpha=4.23),     # Shannon Te4+
        "I":  _uff(4.500, 0.339),
        "Xe": _uff(4.404, 0.332),
        # Period 6
        "Cs": _uff(4.517, 0.045, alpha=2.42),     # Shannon Cs+
        "Ba": _uff(3.703, 0.364, alpha=6.40),     # Shannon Ba2+
        "La": _uff(3.522, 0.017, alpha=6.07),     # Shannon La3+
        "Ce": _uff(3.556, 0.013, alpha=6.15),     # Shannon Ce3+
        "Pr": _uff(3.606, 0.010, alpha=5.32),     # Shannon Pr3+
        "Nd": _uff(3.575, 0.010, alpha=5.01),     # Shannon Nd3+
        "Pm": _uff(3.547, 0.009),
        "Sm": _uff(3.520, 0.008, alpha=4.74),     # Shannon Sm3+
        "Eu": _uff(3.493, 0.008, alpha=4.53),     # Shannon Eu3+
        "Gd": _uff(3.368, 0.009, alpha=4.37),     # Shannon Gd3+
        "Tb": _uff(3.451, 0.007, alpha=4.25),     # Shannon Tb3+
        "Dy": _uff(3.428, 0.007, alpha=4.07),     # Shannon Dy3+
        "Ho": _uff(3.409, 0.007, alpha=3.97),     # Shannon Ho3+
        "Er": _uff(3.391, 0.007, alpha=3.81),     # Shannon Er3+
        "Tm": _uff(3.374, 0.006, alpha=3.67),     # Shannon Tm3+
        "Yb": _uff(3.355, 0.228, alpha=3.58),     # Shannon Yb3+
        "Lu": _uff(3.640, 0.041, alpha=3.64),     # Shannon Lu3+
        "Hf": _uff(3.141, 0.072, alpha=3.25),     # Shannon Hf4+
        "Ta": _uff(3.170, 0.081, alpha=4.73),     # Shannon Ta5+
        "W":  _uff(3.069, 0.067, alpha=3.78),     # Shannon W6+
        "Re": _uff(2.954, 0.066, alpha=3.97),     # Shannon Re7+
        "Os": _uff(3.120, 0.037, alpha=2.33),     # Shannon Os4+
        "Ir": _uff(2.840, 0.073, alpha=2.11),     # Shannon Ir4+
        "Pt": _uff(2.754, 0.080, alpha=2.00),     # Shannon Pt4+
        "Au": _uff(3.293, 0.039, alpha=2.47),     # Shannon Au+
        "Hg": _uff(2.705, 0.385),
        "Tl": _uff(4.347, 0.680, alpha=7.28),     # Shannon Tl+
        "Pb": _uff(4.297, 0.663, alpha=6.58),     # Shannon Pb2+
        "Bi": _uff(4.370, 0.518, alpha=6.12),     # Shannon Bi3+
        "Po": _uff(4.709, 0.325),
        "At": _uff(4.750, 0.284),
        "Rn": _uff(4.765, 0.248),
        # Period 7
        "Fr": _uff(4.900, 0.050),
        "Ra": _uff(3.677, 0.404),
        "Ac": _uff(3.478, 0.033),
        "Th": _uff(3.396, 0.026, alpha=4.92),     # Shannon Th4+
        "Pa": _uff(3.424, 0.022),
        "U":  _uff(3.395, 0.022, alpha=4.45),     # Shannon U4+
        "Np": _uff(3.424, 0.019),
        "Pu": _uff(3.424, 0.016),
        "Am": _uff(3.381, 0.014),
        "Cm": _uff(3.326, 0.013),
        "Bk": _uff(3.339, 0.013),
        "Cf": _uff(3.313, 0.013),
        "Es": _uff(3.299, 0.012),
        "Fm": _uff(3.286, 0.012),
        "Md": _uff(3.274, 0.011),
        "No": _uff(3.248, 0.011),
        "Lr": _uff(3.236, 0.011),
    }
    ffs.append(opls_aa_uff)

    # --- PHAHST ---
    phahst: dict[str, dict[str, float]] = {
        "Cu": {"alpha": 0.29252, "sigma": 2.73851, "epsilon": 8.82345, "c6": 6.96956,  "c8": 262.82938, "c10": 13951.49740},
        "C":  {"alpha": 0.71317, "sigma": 3.35929, "epsilon": 4.00147, "c6": 11.88969, "c8": 547.51694, "c10": 27317.97855},
        "O":  {"alpha": 1.68064, "sigma": 3.23867, "epsilon": 3.89544, "c6": 27.70093, "c8": 709.36452, "c10": 19820.89339},
        "H":  {"alpha": 0.02117, "sigma": 1.87446, "epsilon": 3.63874, "c6": 0.16278,  "c8": 5.03239,   "c10": 202.99322},
    }
    ffs.append(phahst)

    return ffs[name]


def apply_ff_to_system_report(
    system: list, ff: dict[str, dict[str, float]],
) -> tuple[list, dict[str, int]]:
    """Apply force-field parameters; return (system, {missing element: atom count}).

    Atoms whose element is absent from the force field get all parameters set
    to zero, as the original pdb_wizard did, and are counted in the returned
    dict so the caller can warn the user instead of silently writing zeros.
    """
    missing: dict[str, int] = {}
    for atom in system:
        el = atom.element.symbol
        if el in ff:
            params = ff[el]
            atom.alpha = params["alpha"]
            atom.sigma = params["sigma"]
            atom.epsilon = params["epsilon"]
            atom.c6 = params["c6"]
            atom.c8 = params["c8"]
            atom.c10 = params["c10"]
        else:
            atom.alpha = atom.sigma = atom.epsilon = 0.0
            atom.c6 = atom.c8 = atom.c10 = 0.0
            missing[el] = missing.get(el, 0) + 1
    return system, missing


def apply_ff_to_system(system: list, ff: dict[str, dict[str, float]]) -> list:
    system, _ = apply_ff_to_system_report(system, ff)
    return system


def missing_ff_messages(missing: dict[str, int]) -> list[str]:
    """Human-readable warnings for apply_ff_to_system_report's missing dict."""
    return [
        f"!!! {n} atom{'s' if n != 1 else ''} of element {el} not found in forcefield, "
        f"parameters set to all zeros !!!"
        for el, n in sorted(missing.items())
    ]
