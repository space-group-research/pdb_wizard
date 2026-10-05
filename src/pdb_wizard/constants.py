"""Unified element data merging pdb_wizard and moltui element tables."""

from __future__ import annotations

from dataclasses import dataclass

BOHR_TO_ANGSTROM = 0.529177249


def _bond_r_from_atomic_number(z: int) -> float:
    if z == 1:
        return 0.9
    if z <= 5:
        return 1.2
    if z <= 7:
        return 1.7
    if z <= 9:
        return 1.5
    if z <= 13:
        return 1.2
    if z <= 17:
        return 1.8
    return 1.8


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    atomic_number: int
    mass: float
    cpk_color: tuple[int, int, int]
    covalent_radius: float
    vdw_radius: float
    bond_r: float


# fmt: off
ELEMENTS: dict[str, Element] = {
    # Period 1
    "H":  Element("H",  "Hydrogen",      1,   1.00797,   (255, 255, 255), 0.31, 1.20, 0.9),
    "He": Element("He", "Helium",        2,   4.0026,    (217, 255, 255), 0.28, 1.40, 1.2),
    # Period 2
    "Li": Element("Li", "Lithium",       3,   6.941,     (204, 128, 255), 1.28, 1.82, 1.2),
    "Be": Element("Be", "Beryllium",     4,   9.01218,   (194, 255,   0), 0.96, 1.98, 1.2),
    "B":  Element("B",  "Boron",         5,  10.81,      (255, 181, 181), 0.84, 1.92, 1.2),
    "C":  Element("C",  "Carbon",        6,  12.011,     (144, 144, 144), 0.76, 1.70, 1.7),
    "N":  Element("N",  "Nitrogen",      7,  14.0067,    ( 48,  80, 248), 0.71, 1.55, 1.7),
    "O":  Element("O",  "Oxygen",        8,  15.9994,    (255,  13,  13), 0.66, 1.52, 1.5),
    "F":  Element("F",  "Fluorine",      9,  18.998403,  (144, 224,  80), 0.57, 1.47, 1.5),
    "Ne": Element("Ne", "Neon",         10,  20.179,     (179, 227, 245), 0.58, 1.54, 1.2),
    # Period 3
    "Na": Element("Na", "Sodium",       11,  22.98977,   (171,  92, 242), 1.66, 2.27, 1.2),
    "Mg": Element("Mg", "Magnesium",    12,  24.305,     (138, 255,   0), 1.41, 1.73, 1.2),
    "Al": Element("Al", "Aluminum",     13,  26.98154,   (191, 166, 166), 1.21, 1.84, 1.2),
    "Si": Element("Si", "Silicon",      14,  28.0855,    (240, 200, 160), 1.11, 2.10, 1.8),
    "P":  Element("P",  "Phosphorus",   15,  30.97376,   (255, 128,   0), 1.07, 1.80, 1.8),
    "S":  Element("S",  "Sulfur",       16,  32.06,      (255, 255,  48), 1.05, 1.80, 1.8),
    "Cl": Element("Cl", "Chlorine",     17,  35.453,     ( 31, 240,  31), 1.02, 1.75, 1.8),
    "Ar": Element("Ar", "Argon",        18,  39.948,     (128, 209, 227), 1.06, 1.88, 1.8),
    # Period 4
    "K":  Element("K",  "Potassium",    19,  39.0983,    (143,  64, 212), 2.03, 2.75, 1.8),
    "Ca": Element("Ca", "Calcium",      20,  40.08,      ( 61, 255,   0), 1.76, 2.31, 1.8),
    "Sc": Element("Sc", "Scandium",     21,  44.9559,    (230, 230, 230), 1.70, 2.15, 1.8),
    "Ti": Element("Ti", "Titanium",     22,  47.9,       (191, 194, 199), 1.60, 2.11, 1.8),
    "V":  Element("V",  "Vanadium",     23,  50.9415,    (166, 166, 171), 1.53, 2.07, 1.8),
    "Cr": Element("Cr", "Chromium",     24,  51.996,     (138, 153, 199), 1.39, 2.06, 1.8),
    "Mn": Element("Mn", "Manganese",    25,  54.938,     (156, 122, 199), 1.50, 2.05, 1.8),
    "Fe": Element("Fe", "Iron",         26,  55.847,     (224, 102,  51), 1.42, 2.04, 1.8),
    "Co": Element("Co", "Cobalt",       27,  58.9332,    (240, 144, 160), 1.38, 2.00, 1.8),
    "Ni": Element("Ni", "Nickel",       28,  58.7,       ( 80, 208,  80), 1.24, 1.97, 1.8),
    "Cu": Element("Cu", "Copper",       29,  63.546,     (200, 128,  51), 1.32, 1.96, 1.8),
    "Zn": Element("Zn", "Zinc",         30,  65.38,      (125, 128, 176), 1.22, 2.01, 1.8),
    "Ga": Element("Ga", "Gallium",      31,  69.72,      (194, 143, 143), 1.22, 1.87, 1.8),
    "Ge": Element("Ge", "Germanium",    32,  72.59,      (102, 143, 143), 1.20, 2.11, 1.8),
    "As": Element("As", "Arsenic",      33,  74.9216,    (189, 128, 227), 1.19, 1.85, 1.8),
    "Se": Element("Se", "Selenium",     34,  78.96,      (255, 161,   0), 1.20, 1.90, 1.8),
    "Br": Element("Br", "Bromine",      35,  79.904,     (166,  41,  41), 1.20, 1.85, 1.8),
    "Kr": Element("Kr", "Krypton",      36,  83.8,       ( 92, 184, 209), 1.16, 2.02, 1.8),
    # Period 5
    "Rb": Element("Rb", "Rubidium",     37,  85.4678,    (112,  46, 176), 2.20, 3.03, 1.8),
    "Sr": Element("Sr", "Strontium",    38,  87.62,      (  0, 255,   0), 1.95, 2.49, 1.8),
    "Y":  Element("Y",  "Yttrium",      39,  88.9059,    (148, 255, 255), 1.90, 2.19, 1.8),
    "Zr": Element("Zr", "Zirconium",    40,  91.22,      (148, 224, 224), 1.75, 2.23, 1.8),
    "Nb": Element("Nb", "Niobium",      41,  92.9064,    (115, 194, 201), 1.64, 2.18, 1.8),
    "Mo": Element("Mo", "Molybdenum",   42,  95.94,      ( 84, 181, 181), 1.54, 2.17, 1.8),
    "Tc": Element("Tc", "Technetium",   43,  98.0,       ( 59, 158, 158), 1.47, 2.16, 1.8),
    "Ru": Element("Ru", "Ruthenium",    44, 101.07,      ( 36, 143, 143), 1.46, 2.13, 1.8),
    "Rh": Element("Rh", "Rhodium",      45, 102.9055,    ( 10, 125, 140), 1.42, 2.10, 1.8),
    "Pd": Element("Pd", "Palladium",    46, 106.4,       (  0, 105, 133), 1.39, 2.10, 1.8),
    "Ag": Element("Ag", "Silver",       47, 107.868,     (192, 192, 192), 1.45, 2.11, 1.8),
    "Cd": Element("Cd", "Cadmium",      48, 112.41,      (255, 217, 143), 1.44, 2.18, 1.8),
    "In": Element("In", "Indium",       49, 114.82,      (166, 117, 115), 1.42, 1.93, 1.8),
    "Sn": Element("Sn", "Tin",          50, 118.69,      (102, 128, 128), 1.39, 2.17, 1.8),
    "Sb": Element("Sb", "Antimony",     51, 121.75,      (158,  99, 181), 1.39, 2.06, 1.8),
    "Te": Element("Te", "Tellurium",    52, 127.6,       (212, 122,   0), 1.38, 2.06, 1.8),
    "I":  Element("I",  "Iodine",       53, 126.9045,    (148,   0, 148), 1.39, 1.98, 1.8),
    "Xe": Element("Xe", "Xenon",        54, 131.3,       ( 66, 158, 176), 1.40, 2.16, 1.8),
    # Period 6
    "Cs": Element("Cs", "Cesium",       55, 132.9054,    ( 87,  23, 143), 2.44, 3.43, 1.8),
    "Ba": Element("Ba", "Barium",       56, 137.33,      (  0, 201,   0), 2.15, 2.68, 1.8),
    "La": Element("La", "Lanthanum",    57, 138.9055,    (112, 212, 255), 2.07, 2.43, 1.8),
    "Ce": Element("Ce", "Cerium",       58, 140.12,      (255, 255, 199), 2.04, 2.42, 1.8),
    "Pr": Element("Pr", "Praseodymium", 59, 140.9077,    (217, 255, 199), 2.03, 2.40, 1.8),
    "Nd": Element("Nd", "Neodymium",    60, 144.24,      (199, 255, 199), 2.01, 2.39, 1.8),
    "Pm": Element("Pm", "Promethium",   61, 145.0,       (163, 255, 199), 1.99, 2.38, 1.8),
    "Sm": Element("Sm", "Samarium",     62, 150.4,       (143, 255, 199), 1.98, 2.36, 1.8),
    "Eu": Element("Eu", "Europium",     63, 151.96,      ( 97, 255, 199), 1.98, 2.35, 1.8),
    "Gd": Element("Gd", "Gadolinium",   64, 157.25,      ( 69, 255, 199), 1.96, 2.34, 1.8),
    "Tb": Element("Tb", "Terbium",      65, 158.9254,    ( 48, 255, 199), 1.94, 2.33, 1.8),
    "Dy": Element("Dy", "Dysprosium",   66, 162.5,       ( 31, 255, 199), 1.92, 2.31, 1.8),
    "Ho": Element("Ho", "Holmium",      67, 164.9304,    (  0, 255, 156), 1.92, 2.30, 1.8),
    "Er": Element("Er", "Erbium",       68, 167.26,      (  0, 230, 117), 1.89, 2.29, 1.8),
    "Tm": Element("Tm", "Thulium",      69, 168.9342,    (  0, 212,  82), 1.90, 2.27, 1.8),
    "Yb": Element("Yb", "Ytterbium",    70, 173.04,      (  0, 191,  56), 1.87, 2.26, 1.8),
    "Lu": Element("Lu", "Lutetium",     71, 174.967,     (  0, 171,  36), 1.87, 2.24, 1.8),
    "Hf": Element("Hf", "Hafnium",      72, 178.49,      ( 77, 194, 255), 1.75, 2.23, 1.8),
    "Ta": Element("Ta", "Tantalum",      73, 180.9479,    ( 77, 166, 255), 1.70, 2.22, 1.8),
    "W":  Element("W",  "Tungsten",      74, 183.85,      ( 33, 148, 214), 1.62, 2.18, 1.8),
    "Re": Element("Re", "Rhenium",       75, 186.207,     ( 38, 125, 171), 1.51, 2.16, 1.8),
    "Os": Element("Os", "Osmium",        76, 190.2,       ( 38, 102, 150), 1.44, 2.16, 1.8),
    "Ir": Element("Ir", "Iridium",       77, 192.22,      ( 23,  84, 135), 1.41, 2.20, 1.8),
    "Pt": Element("Pt", "Platinum",      78, 195.09,      (208, 208, 224), 1.36, 2.13, 1.8),
    "Au": Element("Au", "Gold",          79, 196.9665,    (255, 209,  35), 1.36, 2.14, 1.8),
    "Hg": Element("Hg", "Mercury",       80, 200.59,      (184, 184, 208), 1.32, 2.09, 1.8),
    "Tl": Element("Tl", "Thallium",      81, 204.37,      (166,  84,  77), 1.45, 1.96, 1.8),
    "Pb": Element("Pb", "Lead",          82, 207.2,       ( 87,  89,  97), 1.46, 2.02, 1.8),
    "Bi": Element("Bi", "Bismuth",       83, 208.9804,    (158,  79, 181), 1.48, 2.07, 1.8),
    "Po": Element("Po", "Polonium",      84, 209.0,       (171,  92,   0), 1.40, 1.97, 1.8),
    "At": Element("At", "Astatine",      85, 210.0,       (117,  79,  69), 1.50, 2.02, 1.8),
    "Rn": Element("Rn", "Radon",         86, 222.0,       ( 66, 130, 150), 1.50, 2.20, 1.8),
    # Period 7
    "Fr": Element("Fr", "Francium",      87, 223.0,       ( 66,   0, 102), 2.60, 3.48, 1.8),
    "Ra": Element("Ra", "Radium",        88, 226.0254,    (  0, 125,   0), 2.21, 2.83, 1.8),
    "Ac": Element("Ac", "Actinium",      89, 227.0278,    (112, 171, 250), 2.15, 2.47, 1.8),
    "Th": Element("Th", "Thorium",       90, 232.0381,    (  0, 186, 255), 2.06, 2.45, 1.8),
    "Pa": Element("Pa", "Protactinium",  91, 231.0359,    (  0, 161, 255), 2.00, 2.43, 1.8),
    "U":  Element("U",  "Uranium",       92, 238.029,     (  0, 143, 255), 1.96, 2.41, 1.8),
    "Np": Element("Np", "Neptunium",     93, 237.0482,    (  0, 128, 255), 1.90, 2.39, 1.8),
    "Pu": Element("Pu", "Plutonium",     94, 242.0,       (  0, 107, 255), 1.87, 2.43, 1.8),
    # Beyond Pu (no CPK colors in moltui — use default gray)
    "Am": Element("Am", "Americium",     95, 243.0,       (180, 180, 180), 1.80, 2.44, 1.8),
    "Cm": Element("Cm", "Curium",        96, 247.0,       (180, 180, 180), 1.69, 2.45, 1.8),
    "Bk": Element("Bk", "Berkelium",     97, 247.0,       (180, 180, 180), 1.70, 2.44, 1.8),
    "Cf": Element("Cf", "Californium",   98, 251.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "Es": Element("Es", "Einsteinium",   99, 252.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "Fm": Element("Fm", "Fermium",      100, 257.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "Md": Element("Md", "Mendelevium",  101, 258.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "No": Element("No", "Nobelium",     102, 250.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "Lr": Element("Lr", "Lawrencium",   103, 260.0,       (180, 180, 180), 1.70, 2.45, 1.8),
    "Rf": Element("Rf", "Rutherfordium",104, 261.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Db": Element("Db", "Dubnium",      105, 262.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Sg": Element("Sg", "Seaborgium",   106, 263.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Bh": Element("Bh", "Bohrium",      107, 262.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Hs": Element("Hs", "Hassium",      108, 255.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Mt": Element("Mt", "Meitnerium",   109, 256.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Ds": Element("Ds", "Darmstadtium", 110, 281.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Rg": Element("Rg", "Roentgenium",  111, 282.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Cn": Element("Cn", "Copernicium",  112, 285.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Nh": Element("Nh", "Nihonium",     113, 286.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Fl": Element("Fl", "Flerovium",    114, 289.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Mc": Element("Mc", "Moscovium",    115, 290.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Lv": Element("Lv", "Livermorium",  116, 293.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Ts": Element("Ts", "Tennessine",   117, 294.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    "Og": Element("Og", "Oganesson",    118, 294.0,       (180, 180, 180), 1.70, 2.40, 1.8),
    # Placeholders
    "Da": Element("Da", "Dummy",          0,   0.0,       (255,  20, 147), 1.00, 1.70, 1.8),
    "X":  Element("X",  "Unknown",        0,   0.0,       (255,  20, 147), 1.00, 1.70, 1.8),
}
# fmt: on

DEFAULT_ELEMENT = ELEMENTS["X"]

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    e.atomic_number: e.symbol for e in ELEMENTS.values() if e.atomic_number > 0
}


def get_element(symbol: str) -> Element:
    key = symbol.strip().capitalize()
    return ELEMENTS.get(key, DEFAULT_ELEMENT)


def get_element_by_number(atomic_number: int) -> Element:
    sym = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
    if sym:
        return ELEMENTS[sym]
    return DEFAULT_ELEMENT
