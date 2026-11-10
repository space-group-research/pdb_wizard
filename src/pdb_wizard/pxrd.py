"""Simulated Powder X-Ray Diffraction (PXRD) pattern calculation.

Computes the diffraction pattern from crystal structure using the Debye
scattering equation with atomic form factors.
"""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule

# Cromer-Mann coefficients for X-ray atomic scattering factors
# f(s) = sum_i(a_i * exp(-b_i * s^2)) + c, where s = sin(theta)/lambda
# From International Tables for Crystallography, Vol C
# Format: {element: (a1,b1, a2,b2, a3,b3, a4,b4, c)}
SCATTERING_FACTORS: dict[str, tuple[float, ...]] = {
    "H":  (0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 0.049879, 2.20159, 0.001305),
    "He": (0.8734, 9.1037, 0.6309, 3.3568, 0.3112, 22.9276, 0.1780, 0.9821, 0.0064),
    "Li": (1.1282, 3.9546, 0.7508, 1.0524, 0.6175, 85.3905, 0.4653, 168.261, 0.0377),
    "Be": (1.5919, 43.6427, 1.1278, 1.8623, 0.5391, 103.483, 0.7029, 0.5420, 0.0385),
    "B":  (2.0545, 23.2185, 1.3326, 1.0210, 1.0979, 60.3498, 0.7068, 0.1403, -0.1932),
    "C":  (2.3100, 20.8439, 1.0200, 10.2075, 1.5886, 0.5687, 0.8650, 51.6512, 0.2156),
    "N":  (12.2126, 0.0057, 3.1322, 9.8933, 2.0125, 28.9975, 1.1663, 0.5826, -11.529),
    "O":  (3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.8670, 32.9089, 0.2508),
    "F":  (3.5392, 10.2825, 2.6412, 4.2944, 1.5170, 0.2615, 1.0243, 26.1476, 0.2776),
    "Na": (4.7626, 3.2850, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676),
    "Mg": (5.4204, 2.8275, 2.1735, 79.2611, 1.2269, 0.3808, 2.3073, 7.1937, 0.8584),
    "Al": (6.4202, 3.0387, 1.9002, 0.7426, 1.5936, 31.5472, 1.9646, 85.0886, 1.1151),
    "Si": (6.2915, 2.4386, 3.0353, 32.3337, 1.9891, 0.6785, 1.5410, 81.6937, 1.1407),
    "P":  (6.4345, 1.9067, 4.1791, 27.1570, 1.7800, 0.5260, 1.4908, 68.1645, 1.1149),
    "S":  (6.9053, 1.4679, 5.2034, 22.2151, 1.4379, 0.2536, 1.5863, 56.1720, 0.8669),
    "Cl": (11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574),
    "K":  (8.2186, 12.7949, 7.4398, 0.7748, 1.0519, 213.187, 0.8659, 41.6841, 1.4228),
    "Ca": (8.6266, 10.4421, 7.3873, 0.6599, 1.5899, 85.7484, 1.0211, 178.437, 1.3751),
    "Ti": (9.7595, 7.8508, 7.3558, 0.5000, 1.6991, 35.6338, 1.9021, 116.105, 1.2807),
    "V":  (10.2971, 6.8657, 7.3511, 0.4385, 2.0703, 26.8938, 2.0571, 102.478, 1.2199),
    "Cr": (10.6406, 6.1038, 7.3537, 0.3920, 3.3240, 20.2626, 1.4922, 98.7399, 1.1832),
    "Mn": (11.2819, 5.3409, 7.3573, 0.3432, 3.0193, 17.8674, 2.2441, 83.7543, 1.0896),
    "Fe": (11.7695, 4.7611, 7.3573, 0.3072, 3.5222, 15.3535, 2.3045, 76.8805, 1.0369),
    "Co": (12.2841, 4.2791, 7.3409, 0.2784, 4.0034, 13.5359, 2.3488, 71.1692, 1.0118),
    "Ni": (12.8376, 3.8785, 7.2920, 0.2565, 4.4438, 12.1763, 2.3800, 66.3421, 1.0341),
    "Cu": (13.3380, 3.5828, 7.1676, 0.2470, 5.6158, 11.3966, 1.6735, 64.8126, 1.1910),
    "Zn": (14.0743, 3.2655, 7.0318, 0.2333, 5.1652, 10.3163, 2.4100, 58.7097, 1.3041),
    "Br": (17.1789, 2.1723, 5.2358, 16.5796, 5.6377, 0.2609, 3.9851, 41.4328, 2.9557),
    "Ag": (19.2808, 0.6446, 16.6885, 7.4726, 4.8045, 24.6605, 1.0463, 99.8156, 5.179),
    "Cd": (19.2214, 0.5946, 17.6444, 6.9089, 4.4610, 24.7008, 1.6029, 87.4825, 5.0694),
    "I":  (20.1472, 4.3470, 18.9949, 0.3814, 7.5138, 27.7660, 2.2735, 66.8776, 4.0712),
    "Ba": (20.3361, 3.2160, 19.2970, 0.2756, 10.888, 20.2073, 2.6959, 167.202, 2.7731),
    "La": (20.578, 2.94817, 19.599, 0.244475, 11.3727, 18.7726, 3.28719, 133.124, 2.14678),
    "Pt": (27.0059, 1.5129, 17.7639, 8.8117, 15.7131, 0.1243, 5.7837, 38.6103, 11.6883),
    "Au": (16.8819, 0.4611, 18.5913, 8.6216, 25.5582, 1.4826, 5.860, 36.3956, 12.0658),
    "Pb": (31.0617, 0.6902, 13.0637, 2.3576, 18.442, 8.618, 5.9696, 47.2579, 13.4118),
}

# Default for elements not in the table
_DEFAULT_SF = (6.0, 3.0, 3.0, 10.0, 2.0, 0.5, 1.0, 50.0, 0.5)


def _atomic_form_factor(element: str, s: np.ndarray) -> np.ndarray:
    """Compute atomic form factor f(s) for an element.
    s = sin(theta)/lambda in A^-1.
    """
    params = SCATTERING_FACTORS.get(element, _DEFAULT_SF)
    a1, b1, a2, b2, a3, b3, a4, b4, c = params
    s2 = s * s
    return (a1 * np.exp(-b1 * s2) + a2 * np.exp(-b2 * s2) +
            a3 * np.exp(-b3 * s2) + a4 * np.exp(-b4 * s2) + c)


def compute_pxrd(
    mol: Molecule,
    wavelength: float = 1.5406,  # Cu K-alpha in Angstroms
    two_theta_min: float = 5.0,
    two_theta_max: float = 50.0,
    n_points: int = 2000,
    peak_width: float = 0.1,  # Gaussian broadening in degrees
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute simulated PXRD pattern using Bragg's law and structure factors.

    Uses the crystal structure with PBC to compute hkl reflections and
    their intensities from structure factors.

    Args:
        mol: Molecule with PBC (crystal structure).
        wavelength: X-ray wavelength in Angstroms (default Cu K-alpha).
        two_theta_min: Minimum 2-theta in degrees.
        two_theta_max: Maximum 2-theta in degrees.
        n_points: Number of points in the output pattern.
        peak_width: Gaussian peak broadening (FWHM in degrees).
        progress_callback: Optional callable(frac).

    Returns:
        (two_theta, intensity) arrays.
    """
    if mol.pbc is None or len(mol.atoms) == 0:
        return np.array([]), np.array([])

    two_theta_all, intensities = pxrd_reflections(mol, wavelength, two_theta_min,
                                                  two_theta_max)
    reflections = list(zip(two_theta_all.tolist(), intensities.tolist()))

    if progress_callback:
        progress_callback(1.0)

    if not reflections:
        return np.array([]), np.array([])

    # Build pattern by convolving with Gaussian peaks
    two_theta_arr = np.linspace(two_theta_min, two_theta_max, n_points)
    pattern = np.zeros(n_points)

    sigma = peak_width / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma

    for tt, intensity in reflections:
        peak = intensity * np.exp(-0.5 * ((two_theta_arr - tt) / sigma) ** 2)
        pattern += peak

    # Normalize to max = 100
    if pattern.max() > 0:
        pattern = pattern / pattern.max() * 100

    return two_theta_arr, pattern


def pxrd_reflections(mol: Molecule, wavelength: float = 1.5406, two_theta_min: float = 5.0,
                     two_theta_max: float = 50.0, with_hkl: bool = False):
    """(2-theta, intensity) of every reflection in range; with ``with_hkl`` also the indices.

    Intensities are |F|^2 times the Lorentz-polarization factor.
    """
    pbc = mol.pbc
    # pbc.reciprocal_basis_matrix is inv(basis): its COLUMNS are the reciprocal vectors
    # a*, b*, c* (a_i . a*_j = delta_ij), so G = h a* + k b* + l c* = hkl @ inv(basis).T.
    # Using its rows put peaks in the wrong place for any non-orthogonal cell.
    inv_basis = pbc.reciprocal_basis_matrix
    recip_vectors = inv_basis.T

    # Determine hkl range from max 2-theta
    # d_min = lambda / (2 * sin(theta_max))
    theta_max = np.radians(two_theta_max / 2)
    d_min = wavelength / (2 * np.sin(theta_max))
    1.0 / (2 * d_min)  # sin(theta)/lambda max

    # Generate hkl indices
    # |G_hkl| = |h*a* + k*b* + l*c*|, d = 1/|G|
    h_max = int(np.ceil(pbc.a / d_min)) + 1
    k_max = int(np.ceil(pbc.b / d_min)) + 1
    l_max = int(np.ceil(pbc.c / d_min)) + 1
    # Cap to reasonable range
    h_max = min(h_max, 20)
    k_max = min(k_max, 20)
    l_max = min(l_max, 20)

    coords = np.array([a.x for a in mol.atoms])
    elements = np.array([a.element.symbol for a in mol.atoms])
    unique_elements = list(set(elements.tolist()))

    # Pre-compute fractional coordinates and element-grouped slices
    frac_coords = coords @ inv_basis
    el_groups = {el: frac_coords[elements == el] for el in unique_elements}

    # Generate the full (h,k,l) grid, drop (0,0,0), filter to two_theta range.
    h_arr = np.arange(-h_max, h_max + 1)
    k_arr = np.arange(-k_max, k_max + 1)
    l_arr = np.arange(-l_max, l_max + 1)
    H, K, L = np.meshgrid(h_arr, k_arr, l_arr, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1)        # (M, 3)
    hkl = hkl[(hkl != 0).any(axis=1)]
    G_all = hkl.astype(float) @ recip_vectors                        # (M, 3)
    G_norm = np.linalg.norm(G_all, axis=1)
    valid = G_norm > 1e-10
    hkl = hkl[valid]
    G_norm = G_norm[valid]
    d_spacing = 1.0 / G_norm
    sin_theta = wavelength / (2 * d_spacing)
    valid = np.abs(sin_theta) <= 1.0
    hkl = hkl[valid]
    sin_theta = sin_theta[valid]
    two_theta_all = 2 * np.degrees(np.arcsin(sin_theta))
    in_range = (two_theta_all >= two_theta_min) & (two_theta_all <= two_theta_max)
    hkl = hkl[in_range]
    two_theta_all = two_theta_all[in_range]
    sin_theta = sin_theta[in_range]
    n_refl = len(hkl)

    # Form factors per element vs s = sin(theta)/lambda — vectorized across reflections
    s_all = sin_theta / wavelength
    f_per_el = {el: _atomic_form_factor(el, s_all) for el in unique_elements}

    # Structure factor: for each element group, sum cos/sin of 2π·(hkl·frac)
    # over its atoms. Then total F = Σ_el f_el(s) * Σ_atoms_in_el e^{2πi(hkl·frac)}
    F_real = np.zeros(n_refl)
    F_imag = np.zeros(n_refl)
    if n_refl > 0:
        for el in unique_elements:
            frac_g = el_groups[el]              # (n_el, 3)
            if len(frac_g) == 0:
                continue
            # phase shape: (n_refl, n_el)
            phase = 2 * np.pi * (hkl.astype(float) @ frac_g.T)
            cos_sum = np.cos(phase).sum(axis=1)
            sin_sum = np.sin(phase).sum(axis=1)
            f_el = f_per_el[el]                # (n_refl,)
            F_real += f_el * cos_sum
            F_imag += f_el * sin_sum

    intensities = F_real ** 2 + F_imag ** 2

    # Lorentz-polarization factor
    theta_rad = np.radians(two_theta_all / 2)
    cos2t = np.cos(np.radians(two_theta_all))
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    denom = sin_t * sin_t * cos_t
    lp = np.where(np.abs(denom) > 1e-10, (1 + cos2t ** 2) / denom, 0.0)
    intensities *= np.abs(lp)

    if with_hkl:
        return two_theta_all, intensities, hkl
    return two_theta_all, intensities
