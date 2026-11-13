"""Numbers checked against independent references: analytic results, brute force over
explicit periodic images, Bragg's law, and the masses in MPMC's own model files.

Each test reproduces a wrong number found in the bug hunt of the numerical routines.
"""

import copy
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.analyses import run_coordination, run_pxrd, run_rdf
from pdb_wizard.atom import Atom
from pdb_wizard.checks import check_structure
from pdb_wizard.coordination import compute_coordination, compute_coordination_trajectory
from pdb_wizard.gyration import compute_gyration_radius, compute_gyration_trajectory
from pdb_wizard.hbonds import detect_hbonds
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import (
    extend_axis,
    formula_unit,
    pore_size_distribution,
    surface_area,
    system_info,
    void_volume,
)
from pdb_wizard.pbc import PBC
from pdb_wizard.pxrd import SCATTERING_FACTORS, _atomic_form_factor, pxrd_reflections
from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory
from pdb_wizard.reduce_cell import identify_framework_and_sorbates, reduce_to_primitive
from pdb_wizard.rmsd import compute_rmsd
from pdb_wizard.sim_inputs import generate_cp2k_input
from pdb_wizard.sorbates import SORBATE_MODELS, insert_sorbates, sorbate_atoms

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "fixtures" / "hunt_science_baseline.npz"
R_AR = 1.88  # argon van der Waals radius in constants.py


def _load(name):
    result = read_file(str(ROOT / name))
    return result[0] if isinstance(result, tuple) else result


def _basis(a, b, c, al, be, ga):
    """Cell matrix (rows a, b, c) from the standard crystallographic formula."""
    al, be, ga = np.radians([al, be, ga])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0, 0], [b * np.cos(ga), b * np.sin(ga), 0],
                     [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


def _mk(names, xyz, cell=None):
    atoms = [Atom(x, y, z, n) for n, (x, y, z) in zip(names, xyz)]
    return Molecule(atoms=atoms, pbc=PBC(*cell) if cell else None)


def _images(basis, n=1):
    return np.array(list(itertools.product(range(-n, n + 1), repeat=3))) @ basis


def _nearest(v, basis):
    """Shortest periodic image of vector v, by explicit search."""
    cands = v + _images(basis)
    return cands[np.argmin(np.linalg.norm(cands, axis=1))]


def _with_sorbates(mol, model, n, seed=1):
    mol = copy.deepcopy(mol)
    mod = SORBATE_MODELS[model]
    placed = insert_sorbates(mol, mod, n, seed=seed)
    assert len(placed) == n
    mol.atoms += sorbate_atoms(mod, placed, 2)
    return mol


def _one_sorbate(model, cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), at=(10.0, 10.0, 10.0)):
    mod = SORBATE_MODELS[model]
    xyz = np.array([[s.x, s.y, s.z] for s in mod.sites]) + np.array(at)
    return Molecule(atoms=sorbate_atoms(mod, [xyz], 2), pbc=PBC(*cell))


def _sc_argon(n, a=5.0):
    return np.array([np.array(s) * a for s in itertools.product(range(n), repeat=3)])


# ---------------------------------------------------------------------------
# C1: void volume in non-orthogonal cells
# ---------------------------------------------------------------------------

def _void_reference(mol, probe, n=3000, seed=7):
    """Monte Carlo void fraction using the distance to the nearest of 27 explicit images."""
    basis = mol.pbc.basis_matrix
    xyz = np.array([a.x for a in mol.atoms])
    frac = xyz @ np.linalg.inv(basis)
    xyz = (frac - np.floor(frac)) @ basis
    radii2 = (np.array([a.vdw for a in mol.atoms]) + probe) ** 2
    pts = np.random.default_rng(seed).random((n, 3)) @ basis
    occupied = np.zeros(n, dtype=bool)
    for shift in _images(basis):
        d2 = ((pts[:, None, :] - (xyz + shift)[None]) ** 2).sum(axis=2)
        occupied |= (d2 <= radii2[None]).any(axis=1)
    return 1.0 - occupied.mean()


def test_void_volume_hexagonal_cell_matches_explicit_images():
    mol = _load("Ni-MOF-74.pqr")
    got = void_volume(mol, 20000, 1.14)["void_fraction"]
    assert got == pytest.approx(_void_reference(mol, 1.14), abs=0.025)  # was 0.30 vs 0.375


def test_void_volume_does_not_depend_on_the_choice_of_lattice_vectors():
    # simple cubic argon, a = 5, probe 0.9 (spheres overlap); grid-integrated reference
    a = 5.0
    grid = np.stack(np.meshgrid(*[np.linspace(0, a, 61)[:-1] + a / 120] * 3, indexing="ij"),
                    -1).reshape(-1, 3)
    corners = np.array(list(itertools.product((0, a), repeat=3)))
    nearest = np.linalg.norm(grid[:, None, :] - corners[None], axis=2).min(axis=1)
    reference = 1.0 - (nearest <= R_AR + 0.9).mean()
    xyz = _sc_argon(3)
    ortho = _mk(["Ar"] * 27, xyz, (15, 15, 15, 90, 90, 90))
    sheared = _mk(["Ar"] * 27, xyz, (15, 15 * np.sqrt(2), 15, 90, 90, 45))  # b = a1 + a2
    assert void_volume(ortho, 40000, 0.9)["void_fraction"] == pytest.approx(reference, abs=0.01)
    assert void_volume(sheared, 40000, 0.9)["void_fraction"] == pytest.approx(reference, abs=0.01)


def test_void_volume_of_a_cell_without_atoms_is_the_whole_cell():
    empty = Molecule(atoms=[], pbc=PBC(10, 10, 10, 90, 90, 90))
    result = void_volume(empty)
    assert result["total_volume"] == pytest.approx(1000.0)
    assert result["void_volume"] == pytest.approx(1000.0)
    assert result["void_fraction"] == 1.0


# ---------------------------------------------------------------------------
# C2: masses, formula, radii and form factors of MPMC sorbate models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, extra", [
    ("co2_phast", {"C": 20, "O": 40}),
    ("ch4_9site", {"C": 20, "H": 80}),
    ("h2_bss", {"H": 40}),
    ("h2o_tip4p", {"O": 20, "H": 40}),
])
def test_density_and_formula_use_the_model_masses(model, extra):
    hk = _load("HKUST-1.pdb")
    base = formula_unit(hk)
    framework_mass = sum(a.mass for a in hk.atoms)
    mol = _with_sorbates(hk, model, 20)
    mass = framework_mass + 20 * sum(s.mass for s in SORBATE_MODELS[model].sites)
    info = system_info(mol, "x")
    assert info["density"] == pytest.approx(mass * 1.66054 / hk.pbc.volume, rel=1e-9)
    expected = dict(base)
    for el, n in extra.items():
        expected[el] = expected.get(el, 0) + n
    assert info["formula"] == expected  # no Co, Mo or X, and no massless sites


def test_density_of_mpmc_output_matches_its_mass_column():
    mol = _load("HKUST-1_out_mpmc.pdb")
    mass = sum(a.file_mass if a.mpmc_flag == "M" else a.mass for a in mol.atoms)
    info = system_info(mol, "x")
    assert info["density"] == pytest.approx(mass * 1.66054 / mol.pbc.volume, rel=1e-9)
    assert "Co" not in info["formula"]


def test_surface_area_per_gram_uses_the_model_masses():
    mol = _one_sorbate("co2_trappe")
    result = surface_area(mol, 3000, 1.4)
    mass_amu = result["surface_area"] * 1e-20 / result["area_per_volume"] / 1.66054e-24
    assert mass_amu == pytest.approx(12.01 + 2 * 16.0, rel=1e-6)  # not 3 x cobalt


def test_void_volume_uses_real_radii_for_model_sites():
    # TraPPE CO2: C (r = 1.70) with an O (r = 1.52) 1.16 A either side; the O-O overlap
    # lies inside the C sphere, so union = V_C + 2 (V_O - lens(C, O)). Read as three
    # cobalt atoms (r = 2.00) the molecule would be half as large again.
    def lens(r1, r2, d):
        return np.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (
            12 * d)

    def ball(r):
        return 4 / 3 * np.pi * r ** 3

    union = ball(1.70) + 2 * (ball(1.52) - lens(1.70, 1.52, 1.16))
    mol = _one_sorbate("co2_trappe", cell=(7.0, 7.0, 7.0, 90.0, 90.0, 90.0), at=(3.5, 3.5, 3.5))
    got = void_volume(mol, 200000, 0.0)
    assert got["total_volume"] - got["void_volume"] == pytest.approx(union, rel=0.02)
    # massless sites take no room: BSS H2 is its two H (r = 1.20, 0.742 A apart)
    union = 2 * ball(1.20) - lens(1.20, 1.20, 0.742)
    mol = _one_sorbate("h2_bss", cell=(6.0, 6.0, 6.0, 90.0, 90.0, 90.0), at=(3.0, 3.0, 3.0))
    got = void_volume(mol, 200000, 0.0)
    assert got["total_volume"] - got["void_volume"] == pytest.approx(union, rel=0.02)


def test_pore_sizes_ignore_massless_sites():
    ar = _mk(["Ar"], [[1.0, 1.0, 1.0]], (8, 8, 8, 90, 90, 90))
    with_site = copy.deepcopy(ar)
    site = Atom(5.0, 5.0, 5.0, "M")  # TIP4P's charge site: no mass, no size
    site.mpmc_flag, site.mpmc_mol, site.file_mass = "M", 2, 0.0
    with_site.atoms.append(site)
    np.testing.assert_array_equal(pore_size_distribution(with_site, 2000)[1],
                                  pore_size_distribution(ar, 2000)[1])


def test_pxrd_uses_carbon_and_oxygen_for_a_co2_model():
    # one TraPPE CO2 along x in a 10 A cube: F(h00) = f_C + 2 f_O cos(2 pi h 1.16 / 10)
    lam, a = 1.5406, 10.0
    mol = _one_sorbate("co2_trappe", cell=(a, a, a, 90.0, 90.0, 90.0), at=(0.0, 0.0, 0.0))
    tt, inten, hkl = pxrd_reflections(mol, lam, 5.0, 40.0, with_hkl=True)

    def intensity(h):
        return float(inten[np.all(hkl == (h, 0, 0), axis=1)][0])

    def cromer_mann(el, s):
        p = SCATTERING_FACTORS[el]
        return sum(p[2 * i] * np.exp(-p[2 * i + 1] * s * s) for i in range(4)) + p[8]

    def expected(h):
        theta = np.arcsin(lam * h / (2 * a))
        s = np.sin(theta) / lam
        f = cromer_mann("C", s) + 2 * cromer_mann("O", s) * np.cos(2 * np.pi * h * 1.16 / a)
        lp = (1 + np.cos(2 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))
        return f * f * lp

    assert intensity(2) / intensity(1) == pytest.approx(expected(2) / expected(1), rel=1e-6)
    assert intensity(3) / intensity(1) == pytest.approx(expected(3) / expected(1), rel=1e-6)


def test_cp2k_input_has_real_elements_and_no_massless_sites(tmp_path):
    hk = _load("HKUST-1.pdb")
    for model, kinds, n_real in (("co2_phast", {"C", "Cu", "H", "O"}, 3), ("h2_bss", None, 2)):
        mol = _with_sorbates(hk, model, 2)
        out = generate_cp2k_input(mol, tmp_path / model, "cell_opt")
        lines = Path(out["xyz"]).read_text().splitlines()
        assert int(lines[0]) == len(lines) - 2 == len(hk.atoms) + 2 * n_real
        symbols = {line.split()[0] for line in lines[2:]}
        assert symbols == {"C", "Cu", "H", "O"}
        text = Path(out["input"]).read_text()
        assert set(re.findall(r"^ +&KIND (\w+)$", text, re.M)) == {"C", "Cu", "H", "O"}
        if kinds:
            assert set(out["elements"]) == kinds
    tail = Path(tmp_path / "co2_phast" / "system.xyz").read_text().splitlines()[-3:]
    assert sorted(line.split()[0] for line in tail) == ["C", "O", "O"]


# ---------------------------------------------------------------------------
# C3 / M1 / m1: RDF and coordination number beyond half the cell width
# ---------------------------------------------------------------------------

def _gas(cell, n_ar, n_he, seed):
    xyz = np.random.default_rng(seed).random((n_ar + n_he, 3)) @ _basis(*cell)
    return _mk(["Ar"] * n_ar + ["He"] * n_he, xyz, cell)


def _brute_g(mol, s1, s2, r_max, n_bins):
    """g(r) from an explicit sum over lattice images (no minimum image)."""
    basis = mol.pbc.basis_matrix
    x1 = np.array([a.x for a in mol.atoms if a.element.symbol == s1])
    x2 = np.array([a.x for a in mol.atoms if a.element.symbol == s2])
    dr = r_max / n_bins
    hist = np.zeros(n_bins)
    for shift in _images(basis, 3):
        r = np.linalg.norm(x2[None] + shift - x1[:, None], axis=2).ravel()
        r = r[(r > 1e-9) & (r < r_max)]
        hist += np.bincount((r / dr).astype(int), minlength=n_bins)[:n_bins]
    centers = (np.arange(n_bins) + 0.5) * dr
    return hist / (len(x1) * len(x2) / abs(np.linalg.det(basis)) * 4 * np.pi * centers ** 2 * dr)


@pytest.mark.parametrize("cell", [(20, 20, 20, 90, 90, 90), (20, 20, 20, 60, 75, 110)])
@pytest.mark.parametrize("pair", [("Ar", "Ar"), ("Ar", "He")])
def test_rdf_is_right_beyond_half_the_cell_width(cell, pair):
    mol = _gas(cell, 150, 60, 3)
    assert mol.pbc.perpendicular_widths().min() / 2 < 13.0
    r, g = compute_rdf(mol, pair[0], pair[1], n_bins=26, r_max=13.0)
    np.testing.assert_allclose(g, _brute_g(mol, pair[0], pair[1], 13.0, 26), rtol=1e-9, atol=1e-12)


def test_rdf_of_an_ideal_gas_stays_at_one_out_to_r_max():
    frames = [_gas((20, 20, 20, 60, 75, 110), 300, 100, s) for s in range(4)]
    for el1, el2 in (("Ar", "Ar"), ("He", "Ar")):
        r, g = compute_rdf_trajectory(frames, el1, el2, n_bins=13, r_max=13.0)
        assert np.abs(g[4:] - 1.0).max() < 0.06  # the last bin was 0.19
    res = run_rdf(frames[0], "Ar", "Ar")  # the viewers' default r max of 13
    assert res.y[-20:].mean() == pytest.approx(1.0, abs=0.05)


def test_rdf_name_filter_beyond_half_the_cell_width():
    mol = _gas((16, 16, 16, 90, 90, 90), 120, 0, 5)
    for atom in mol.atoms[:50]:
        atom.name = "ArA"
    r, g = compute_rdf(mol, "Ar", "Ar", n_bins=12, r_max=12.0, name1="ArA")
    # the same 50 atoms as a different element, for the brute force
    sub = _mk(["He"] * 50 + ["Ar"] * 70, np.array([a.x for a in mol.atoms]), (16,) * 3 + (90,) * 3)
    brute = _brute_g(sub, "He", "Ar", 12.0, 12) * 70 / 120 + _brute_g(sub, "He", "He", 12.0, 12) \
        * 50 / 120
    np.testing.assert_allclose(g, brute, rtol=1e-9, atol=1e-12)


def test_rdf_default_r_max_is_half_the_smallest_perpendicular_width():
    mol = _gas((20, 20, 20, 60, 75, 110), 50, 0, 1)
    r, _ = compute_rdf(mol, "Ar", "Ar", n_bins=10)
    half = mol.pbc.perpendicular_widths().min() / 2  # 7.15, not min(a, b, c) / 2 = 10
    assert r[-1] + (r[1] - r[0]) / 2 == pytest.approx(half, rel=1e-12)


def _fcc(n=5, a=4.0):
    base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    xyz = np.array([(b + np.array(s)) * a for s in itertools.product(range(n), repeat=3)
                    for b in base])
    return _mk(["Ar"] * len(xyz), xyz, (a * n,) * 3 + (90, 90, 90)), xyz


def _fcc_neighbours_within(xyz, basis, radius):
    d = np.concatenate([xyz + s - xyz[0] for s in _images(basis, 2)])
    r = np.linalg.norm(d, axis=1)
    return int(((r > 1e-6) & (r < radius)).sum())


def test_coordination_number_beyond_half_the_cell_width():
    mol, xyz = _fcc(4)  # 16 A box: half width 8
    basis = mol.pbc.basis_matrix
    for r_max in (9.9, 13.0):  # beyond half the width the minimum image alone falls short
        _, _, avg = compute_coordination(mol, "Ar", "Ar", r_max=r_max)
        assert avg == pytest.approx(_fcc_neighbours_within(xyz, basis, r_max))
    n13 = _fcc_neighbours_within(xyz, basis, 13.0)
    assert "{}.00".format(n13) in run_coordination(mol, "Ar", "Ar", r_max=13.0).summary
    _, _, avg = compute_coordination_trajectory([mol, mol], "Ar", "Ar", 13.0)
    assert avg == pytest.approx(n13)


def test_coordination_curve_counts_neighbours_closer_than_r():
    mol, xyz = _fcc()
    r, cn, avg = compute_coordination(mol, "Ar", "Ar", r_max=5.0, n_bins=200)
    for i in (0, 50, 112, 113, 158, 159, 160, 194, 195, 199):
        assert cn[i] == pytest.approx(_fcc_neighbours_within(xyz, mol.pbc.basis_matrix, r[i]))
    assert cn[112] == 0 and r[112] < 4.0 / np.sqrt(2) < r[113]  # first shell at 2.828
    assert avg == cn[-1] == 42


def test_results_at_or_below_half_the_cell_width_are_unchanged():
    """Bit-for-bit what the code gave before the image sum was added (fixture written by
    the unfixed code); the coordination curve only moved one bin to the right (see above)."""
    ref = np.load(BASELINE)
    hk = _load("HKUST-1.pdb")
    tri_cell = tuple(float(v) for v in ref["tri_cell"])
    names = ["Ar"] * 350 + ["He"] * 150
    tri = _mk(names, ref["tri_xyz"], tri_cell)
    assert tri.pbc.perpendicular_widths().min() / 2 > 6.0

    def same(got, key):
        np.testing.assert_array_equal(got[0], ref[key + "_r"])
        np.testing.assert_array_equal(got[1], ref[key + "_g"])

    same(compute_rdf(hk, "Cu", "O", 100, 13.0), "hk_cuo")
    same(compute_rdf(hk, "C", "C", 200, None), "hk_cc")
    same(compute_rdf(tri, "Ar", "Ar", 60, 6.0), "tri_aa")
    same(compute_rdf(tri, "Ar", "He", 60, 6.0), "tri_ah")
    same(compute_rdf_trajectory([tri, tri], "He", "Ar", 60, 6.0), "tri_traj")
    for mol, args, key in ((hk, ("Cu", "O", 5.0), "hk_cn"), (tri, ("Ar", "He", 6.0), "tri_cn")):
        r, cn, avg = compute_coordination(mol, *args)
        np.testing.assert_array_equal(r, ref[key + "_r"])
        np.testing.assert_array_equal(cn[1:], ref[key][:-1])
        assert cn[0] == 0 and avg == float(ref[key + "_avg"])
    # r max beyond the limit: the bins that end below half the width keep their old values
    r, g = compute_rdf(tri, "Ar", "Ar", 130, 13.0)
    below = ref["tri_aa13_r"] + 0.05 < tri.pbc.perpendicular_widths().min() / 2
    assert below.sum() > 80
    np.testing.assert_array_equal(g[below], ref["tri_aa13_g"][below])


# ---------------------------------------------------------------------------
# M3 / M4 / m6: PXRD
# ---------------------------------------------------------------------------

def test_pxrd_keeps_reflections_with_indices_above_20():
    # one Cu per 3 A cube, written as a 12 x 1 x 1 supercell: (200) of the small cell is
    # (24 0 0) of the supercell, at 2 theta = 2 asin(lambda / 3) = 61.8 degrees
    lam = 1.5406
    xyz = np.array([[3.0 * i, 0, 0] for i in range(12)])
    mol = _mk(["Cu"] * 12, xyz, (36, 3, 3, 90, 90, 90))
    tt, inten, hkl = pxrd_reflections(mol, lam, 5.0, 65.0, with_hkl=True)
    strong = inten > 1e-6 * inten.max()
    expected = 2 * np.degrees(np.arcsin(lam / 3.0))
    assert np.any(np.abs(tt[strong] - expected) < 1e-6)
    assert {tuple(h) for h in hkl[strong]} >= {(24, 0, 0), (-24, 0, 0)}
    # every reflection in range is used: count lattice points directly
    big = _mk(["Cu"], [[0, 0, 0]], (30, 30, 30, 90, 90, 90))
    n = int(30 / (lam / (2 * np.sin(np.radians(45))))) + 1
    rng = np.arange(-n, n + 1)
    g = np.linalg.norm(np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T, axis=1)
    g = g[g > 0] / 30.0
    two_theta = 2 * np.degrees(np.arcsin(np.clip(lam * g / 2, 0, 1)))
    count = int(((two_theta >= 5) & (two_theta <= 90) & (lam * g / 2 <= 1)).sum())
    assert len(pxrd_reflections(big, lam, 5.0, 90.0)[0]) == count


def test_form_factor_of_an_untabulated_element_has_z_electrons():
    assert "Zr" not in SCATTERING_FACTORS
    for el, z in (("Zr", 40), ("Hf", 72), ("In", 49), ("Y", 39)):
        assert float(_atomic_form_factor(el, np.array([0.0]))[0]) == pytest.approx(z)
    assert float(_atomic_form_factor("Cu", np.array([0.0]))[0]) == pytest.approx(29, abs=0.02)
    mol = _mk(["Zr", "O"], [[0, 0, 0], [2.3, 2.3, 2.3]], (4.6, 4.6, 4.6, 90, 90, 90))
    summary = run_pxrd(mol).summary
    assert "Zr" in summary and "O" not in summary.replace("Zr", "").split("peaks")[-1]


def test_pxrd_summary_counts_peaks_not_grid_points():
    a = 3.615
    xyz = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
    cu = _mk(["Cu"] * 4, xyz, (a, a, a, 90, 90, 90))
    for width in (0.05, 0.1, 0.3):  # (111) (200) (220) (311) below 90 degrees
        assert run_pxrd(cu, 1.5406, 90.0, width).summary.startswith("4 peaks above 5%")


# ---------------------------------------------------------------------------
# M5 / M6: hydrogen bonds, RMSD and radius of gyration across cell faces
# ---------------------------------------------------------------------------

def _water_box(cell, n_water, seed):
    rng = np.random.default_rng(seed)
    basis = _basis(*cell)
    shape = np.array([[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    centers = []
    while len(centers) < n_water:
        c = rng.random(3) @ basis
        if all(np.linalg.norm(_nearest(c - o, basis)) > 2.6 for o in centers):
            centers.append(c)
    xyz = []
    for c in centers:
        q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        xyz += list(shape @ q.T + c)
    frac = np.array(xyz) @ np.linalg.inv(basis)
    return _mk(["O", "H", "H"] * n_water, (frac - np.floor(frac)) @ basis, cell)  # wrapped


def _brute_hbonds(mol, d_cut=3.5, angle_cut=120.0):
    basis = mol.pbc.basis_matrix
    xyz = np.array([a.x for a in mol.atoms])
    out = set()
    for h in range(1, len(xyz), 3):
        for h_idx in (h, h + 1):
            d = h - 1  # the O of this water
            dh = _nearest(xyz[d] - xyz[h_idx], basis)
            for a in range(0, len(xyz), 3):
                if a == d:
                    continue
                da = _nearest(xyz[a] - xyz[d], basis)
                if np.linalg.norm(da) > d_cut:
                    continue
                ha = da + dh
                cos = np.dot(dh, ha) / np.linalg.norm(dh) / np.linalg.norm(ha)
                if np.degrees(np.arccos(cos)) >= angle_cut:
                    out.add((d, h_idx, a))
    return out


@pytest.mark.parametrize("cell", [(13, 13, 13, 90, 90, 90), (13, 14, 15, 70, 80, 115)])
def test_hbonds_of_waters_wrapped_across_cell_faces(cell):
    mol = _water_box(cell, 40, 3)
    mol.detect_bonds()
    assert len(mol.bonds) == 80
    got = {(d, h, a) for d, h, a, _, _ in detect_hbonds(mol)}
    assert got == _brute_hbonds(mol)
    for d, h, a, dist, angle in detect_hbonds(mol):
        assert dist == pytest.approx(np.linalg.norm(_nearest(mol.atoms[a].x - mol.atoms[d].x,
                                                             mol.pbc.basis_matrix)))


def _drifting_frames():
    start = np.array([[4.0, 5, 5], [5.0, 5, 5], [6.0, 5, 5]])
    return [_mk(["C"] * 3, (start + np.array([0.5 * k, 0, 0])) % 10.0, (10, 10, 10, 90, 90, 90))
            for k in range(12)]


def test_rmsd_follows_atoms_through_the_cell_face():
    idx, rmsd = compute_rmsd(_drifting_frames())
    np.testing.assert_allclose(rmsd, 0.5 * np.arange(12), atol=1e-12)
    idx, rmsd = compute_rmsd(_drifting_frames(), reference_frame=11)
    np.testing.assert_allclose(rmsd, 0.5 * np.arange(12)[::-1], atol=1e-12)


def test_rmsd_without_a_crossing_is_unchanged():
    rng = np.random.default_rng(0)
    base = rng.random((20, 3)) * 4 + 3
    frames = [_mk(["C"] * 20, base + 0.05 * rng.standard_normal((20, 3)), (10, 10, 10, 90, 90, 90))
              for _ in range(5)]
    expected = [np.sqrt(np.mean(np.sum((np.array([a.x for a in f.atoms])
                                        - np.array([a.x for a in frames[0].atoms])) ** 2, axis=1)))
                for f in frames]
    np.testing.assert_array_equal(compute_rmsd(frames)[1], np.array(expected))


def test_radius_of_gyration_of_a_molecule_split_by_the_cell_face():
    frames = _drifting_frames()
    _, rg = compute_gyration_trajectory(frames)
    np.testing.assert_allclose(rg, np.sqrt(2.0 / 3.0), atol=1e-12)
    # triclinic cell, butane-like chain wrapped atom by atom
    cell = (11, 12, 13, 70, 80, 115)
    basis = _basis(*cell)
    chain = np.array([[0, 0, 0], [1.53, 0, 0], [2.1, 1.42, 0], [3.63, 1.42, 0.3]]) - 0.6
    expected = np.sqrt(np.mean(np.sum((chain - chain.mean(axis=0)) ** 2, axis=1)))
    frac = chain @ np.linalg.inv(basis)
    mol = _mk(["C"] * 4, (frac - np.floor(frac)) @ basis, cell)
    assert compute_gyration_radius(mol) == pytest.approx(expected, abs=1e-9)


def test_radius_of_gyration_without_a_split_molecule_is_unchanged():
    def plain(xyz):
        return float(np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1))))

    # a chain bonded to its own periodic image (like a framework) cannot be made whole
    chain = np.array([[0.75 + 1.5 * i, 3.0, 3.0] for i in range(4)])
    polymer = _mk(["C"] * 4, chain, (6, 6, 6, 90, 90, 90))
    polymer.detect_bonds()
    assert len(polymer.bonds) == 4
    assert compute_gyration_radius(polymer) == plain(chain)
    # molecules that no cell face cuts
    water = np.array([[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    xyz = np.concatenate([water + c for c in ([3.0, 3, 3], [9.0, 14, 5], [16.0, 6, 12])])
    mol = _mk(["O", "H", "H"] * 3, xyz, (20, 20, 20, 90, 90, 90))
    assert compute_gyration_radius(mol) == plain(np.array([a.x for a in mol.atoms]))
    oxy = np.array([a.x for a in mol.atoms if a.element.symbol == "O"])
    assert compute_gyration_radius(mol, "O") == plain(oxy)


# ---------------------------------------------------------------------------
# M7: reduce a supercell to one cell
# ---------------------------------------------------------------------------

def _small_framework():
    """A small made-up framework with a cavity (fast to reduce): 12 A cubic cell, atoms on
    a 3 A grid except within 5 A of the cell centre; every parameter set."""
    names, xyz = [], []
    for n, s in enumerate(itertools.product(range(4), repeat=3)):
        p = np.array(s) * 3.0 + 0.4
        if np.linalg.norm(p - 6.0) > 5.0:
            names.append(("C", "O", "Cu")[n % 3])
            xyz.append(p)
    hk = _mk(names, xyz, (12.0, 12.0, 12.0, 90.0, 90.0, 90.0))
    for a in hk.atoms:
        a.charge, a.alpha, a.sigma, a.epsilon = 0.1, 0.5, 3.0, 50.0
        a.c6, a.c8, a.c10 = 1.0, 2.0, 3.0
    return hk


@pytest.mark.parametrize("model, reps", [("co2_phast", (2, 1, 1)), ("h2_phahst", (3, 1, 1)),
                                         ("h2_bss", (2, 1, 1))])
def test_reduce_keeps_every_site_of_distinct_sorbates(model, reps):
    hk = _small_framework()
    sup = copy.deepcopy(hk)
    extend_axis(sup, 0, reps[0] - 1)
    mod = SORBATE_MODELS[model]
    placed = insert_sorbates(sup, mod, 4, seed=4, min_dist=1.6)
    sup.atoms += sorbate_atoms(mod, placed, 2)
    sup.detect_bonds()
    framework, sorbates = identify_framework_and_sorbates(sup, *reps)
    assert len(framework) == len(hk.atoms)
    assert sorted(len(s) for s in sorbates) == [len(mod.sites)] * 4
    red = reduce_to_primitive(sup, *reps)
    movable = [a for a in red.atoms if a.mpmc_flag == "M"]
    assert len(red.atoms) == len(hk.atoms) + 4 * len(mod.sites)
    assert sorted(a.name for a in movable) == sorted([s.atomtype for s in mod.sites] * 4)
    assert len({a.mpmc_mol for a in movable}) == 4
    assert all(a.file_mass is not None and a.mpmc_res == mod.label for a in movable)
    framework_atoms = [a for a in red.atoms if a.mpmc_flag != "M"]
    assert all((a.c6, a.c8, a.c10, a.charge) == (1.0, 2.0, 3.0, 0.1) for a in framework_atoms)
    # molecules stay whole and rigid: same inter-site distances as the model
    sites = np.array([[s.x, s.y, s.z] for s in mod.sites])
    want = sorted(np.linalg.norm(p - q) for p, q in itertools.combinations(sites, 2))
    by_mol = {}
    for a in movable:
        by_mol.setdefault(a.mpmc_mol, []).append(a.x)
    for xyz in by_mol.values():
        got = sorted(np.linalg.norm(p - q) for p, q in itertools.combinations(xyz, 2))
        np.testing.assert_allclose(got, want, atol=1e-9)
    kept = reduce_to_primitive(sup, *reps, keep_sorbate_indices=[0, 3])
    assert len(kept.atoms) == len(hk.atoms) + 2 * len(mod.sites)


@pytest.mark.parametrize("model", ["co2_phast", "h2_bss", "h2_phahst"])
def test_extend_then_reduce_gives_the_original_cell_back(model):
    hk = _small_framework()
    mol = _with_sorbates(hk, model, 3, seed=2)
    n_sites = len(SORBATE_MODELS[model].sites)
    sup = copy.deepcopy(mol)
    extend_axis(sup, 0, 1)
    extend_axis(sup, 1, 1)
    sup.detect_bonds()
    red = reduce_to_primitive(sup, 2, 2, 1)
    assert len(red.atoms) == len(mol.atoms) == len(hk.atoms) + 3 * n_sites
    movable = [a for a in red.atoms if a.mpmc_flag == "M"]
    assert len(movable) == 3 * n_sites and len({a.mpmc_mol for a in movable}) == 3
    assert red.pbc.a == pytest.approx(hk.pbc.a) and red.pbc.b == pytest.approx(hk.pbc.b)
    # each original site is found again (modulo a lattice vector of the small cell)
    basis = red.pbc.basis_matrix
    for orig in (a for a in mol.atoms if a.mpmc_flag == "M"):
        d = min(np.linalg.norm(_nearest(orig.x - a.x, basis)) for a in movable
                if a.name == orig.name)
        assert d < 1e-6


@pytest.mark.parametrize("sizes", [(2, 2, 2), (3, 1, 1), (1, 2, 1)])
def test_reduce_refuses_a_structure_that_is_not_that_supercell(sizes):
    hk = _load("HKUST-1.pdb")  # one conventional cell
    # independent check: no atom has a same-element partner one sub-cell along
    xyz = np.array([a.x for a in hk.atoms])
    axis = next(i for i, n in enumerate(sizes) if n > 1)
    shift = hk.pbc.basis_matrix[axis] / sizes[axis]
    hits = 0
    for i, atom in enumerate(hk.atoms):
        d = xyz - (xyz[i] + shift)
        d -= np.round(d @ hk.pbc.reciprocal_basis_matrix) @ hk.pbc.basis_matrix
        hits += any(np.linalg.norm(d[j]) < 0.2 and hk.atoms[j].element is atom.element
                    for j in np.nonzero(np.abs(d).max(axis=1) < 0.2)[0])
    assert hits == 0
    n_before = len(hk.atoms)
    label = "{} x {} x {} supercell".format(*sizes)
    with pytest.raises(ValueError, match="not a " + label):
        identify_framework_and_sorbates(hk, *sizes)
    with pytest.raises(ValueError, match="not a " + label):
        reduce_to_primitive(hk, *sizes)
    assert len(hk.atoms) == n_before and hk.pbc.a == pytest.approx(26.343)
    with_sorbates = _with_sorbates(hk, "h2_bss", 3)
    with pytest.raises(ValueError, match="not a " + label):
        reduce_to_primitive(with_sorbates, *sizes)


def test_reduce_of_a_real_supercell_is_what_it_was():
    hk = _small_framework()
    sup = copy.deepcopy(hk)
    extend_axis(sup, 2, 1)
    framework, sorbates = identify_framework_and_sorbates(sup, 1, 1, 2)
    assert sorted(framework) == list(range(len(hk.atoms))) and sorbates == []  # as before the check
    red = reduce_to_primitive(sup, 1, 1, 2)
    assert [a.name for a in red.atoms] == [a.name for a in hk.atoms]
    assert (red.pbc.a, red.pbc.b, red.pbc.c) == pytest.approx((12.0, 12.0, 12.0))
    frac = np.array([a.x for a in hk.atoms]) @ hk.pbc.reciprocal_basis_matrix
    expected = (frac - np.floor(frac)) @ hk.pbc.basis_matrix
    np.testing.assert_allclose(np.array([a.x for a in red.atoms]), expected, atol=1e-9)
    assert reduce_to_primitive(hk, 1, 1, 1).pbc.a == pytest.approx(12.0)  # 1 x 1 x 1 is fine
    # sorbates added after the supercell was built, without MPMC flags (as from an XYZ)
    plain = copy.deepcopy(sup)
    for k in range(3):
        for dx in (0.0, 0.74):
            plain.atoms.append(Atom(5.0 + dx, 5.0 + k, 4.0 + 7.0 * k, "H"))
    plain.detect_bonds()
    framework, sorbates = identify_framework_and_sorbates(plain, 1, 1, 2)
    assert len(framework) == len(hk.atoms) and sorted(len(s) for s in sorbates) == [2, 2, 2]
    assert len(reduce_to_primitive(plain, 1, 1, 2).atoms) == len(hk.atoms) + 6


# ---------------------------------------------------------------------------
# M8 / m5: neighbour grids in skewed cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", [(60, 60, 60, 90, 90, 90), (60, 60, 60, 60, 60, 60),
                                  (60, 60, 60, 110, 115, 120), (60, 60, 60, 50, 55, 60)])
def test_bonds_in_large_skewed_cells(cell):
    # 800 separate C-C pairs 1.54 A apart (1600 atoms selects the cell-list path)
    rng = np.random.default_rng(3)
    basis = _basis(*cell)
    grid = np.array([(i, j, k) for i in range(10) for j in range(10) for k in range(10)])[:800]
    xyz = []
    for c in grid / 10.0 @ basis:
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        xyz += [c - 0.77 * u, c + 0.77 * u]
    mol = _mk(["C"] * 1600, np.array(xyz), cell)
    mol.detect_bonds()
    assert sorted(mol.bonds) == [(2 * i, 2 * i + 1) for i in range(800)]


def test_surface_area_does_not_depend_on_the_choice_of_lattice_vectors():
    # simple cubic argon (a = 5, 6 x 6 x 6), probe 1.4. Reference: one atom with its 26
    # explicit neighbours, sampled directly.
    a, radius = 5.0, R_AR + 1.4
    pts = np.random.default_rng(5).standard_normal((40000, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius
    nb = np.array([np.array(s) * a for s in itertools.product((-1, 0, 1), repeat=3)
                   if s != (0, 0, 0)])
    free = (np.linalg.norm(pts[:, None, :] - nb[None], axis=2) >= radius).all(axis=1).mean()
    reference = 216 * free * 4 * np.pi * radius ** 2
    xyz = _sc_argon(6)
    cells = [(30, 30, 30, 90, 90, 90), (30, 30 * np.sqrt(2), 30, 90, 90, 45),
             (30, 30 * np.sqrt(5), 30, 90, 90, float(np.degrees(np.arctan(0.5))))]  # b = (60,30,0)
    for cell in cells:
        mol = _mk(["Ar"] * 216, xyz, cell)
        assert mol.pbc.volume == pytest.approx(27000)
        got = surface_area(mol, 60000, 1.4)["surface_area"]
        assert got == pytest.approx(reference, rel=0.02)  # the last cell gave +27%


# ---------------------------------------------------------------------------
# m9: structure check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["h2o_tip4p", "h2o_tip5p"])
def test_massless_model_sites_are_not_reported_as_unknown_elements(model):
    mol = _with_sorbates(_load("HKUST-1.pdb"), model, 3)
    assert not [f for f in check_structure(mol) if "unknown element" in f.title]
    stray = copy.deepcopy(mol)
    stray.atoms.append(Atom(1.0, 2.0, 3.0, "Xx"))  # a real unknown is still reported
    assert [f.title for f in check_structure(stray) if "unknown element" in f.title] == [
        "1 atoms have an unknown element"]
