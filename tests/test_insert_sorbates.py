"""Tests for sorbate insertion (insert_sorbates) and the MPMC writer it feeds.

Run from the repo root:  python -m pytest tests/
"""

import importlib.util
import io
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "dist" / "pdb_wizard.py"  # built by conftest.py
spec = importlib.util.spec_from_file_location("pdb_wizard_dist", SRC)
pw = importlib.util.module_from_spec(spec)
sys.modules["pdb_wizard_dist"] = (
    pw  # dataclasses look the module up while it loads; not "pdb_wizard", which is the package
)
spec.loader.exec_module(pw)


def min_image_dists(pbc, a, b):
    d = a[:, None, :] - b[None, :, :]
    d -= np.round(d @ pbc.reciprocal_basis_matrix) @ pbc.basis_matrix
    return np.sqrt((d * d).sum(axis=2))


def framework(pbc):
    """A sparse 'framework': one LJ carbon per 6 A lattice point of the cell."""
    atoms = []
    for f in np.array(np.meshgrid(*[np.arange(0, 1, 0.3)] * 3)).reshape(3, -1).T:
        x, y, z = f @ pbc.basis_matrix
        atom = pw.Atom(x, y, z, "C")
        atom.sigma, atom.epsilon = 3.4, 50.0
        atoms.append(atom)
    return pw.Molecule(atoms=atoms, pbc=pbc)


@pytest.mark.parametrize(
    "cell", [(20, 20, 20, 90, 90, 90), (20.88, 13.3, 17.1, 90, 99.9, 90), (15, 17, 19, 70, 80, 100)]
)
def test_no_overlaps_in_skewed_cells(cell):
    """Every site respects the distance rule against the framework and each other,
    measured with the minimum image, also across the periodic boundary."""
    mol = framework(pw.PBC(*cell))
    model = pw.SORBATE_MODELS["h2o_dc97"]
    placed = pw.insert_sorbates(mol, model, 15, seed=1)
    assert len(placed) == 15
    fw = np.array([a.x for a in mol.atoms])
    o_idx = 0  # OXY is the only LJ site
    for k, mk in enumerate(placed):
        # oxygen vs framework carbons: >= 0.85 * (3.234 + 3.4) / 2
        assert min_image_dists(mol.pbc, mk[[o_idx]], fw).min() >= 0.85 * (3.234 + 3.4) / 2 - 1e-9
        # every site >= 1 A from every framework atom
        assert min_image_dists(mol.pbc, mk, fw).min() >= 1.0 - 1e-9
        for j in range(k):
            mj = placed[j]
            assert min_image_dists(mol.pbc, mk[[0]], mj[[0]]).min() >= 0.85 * 3.234 - 1e-9
            assert min_image_dists(mol.pbc, mk, mj).min() >= 1.0 - 1e-9


def test_rigid_geometry_kept():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    model = pw.SORBATE_MODELS["h2o_tip5p"]
    ref = np.array([[s.x, s.y, s.z] for s in model.sites])
    ref_d = np.linalg.norm(ref[:, None] - ref[None], axis=2)
    for xyz in pw.insert_sorbates(mol, model, 5, seed=2):
        assert np.allclose(np.linalg.norm(xyz[:, None] - xyz[None], axis=2), ref_d, atol=1e-9)


def test_fill_stops_when_full_and_n_is_capped():
    mol = framework(pw.PBC(12, 12, 12, 90, 90, 90))
    full = pw.insert_sorbates(mol, "h2o_spce", None, seed=3, max_failures=500)
    assert len(full) > 0
    capped = pw.insert_sorbates(mol, "h2o_spce", 10_000, seed=3, max_failures=500)
    assert 0 < len(capped) < 10_000  # stops at the failure limit instead of looping forever


def test_orientations_are_random():
    mol = pw.Molecule(atoms=[], pbc=pw.PBC(30, 30, 30, 90, 90, 90))
    placed = pw.insert_sorbates(mol, "h2o_tip3p", 200, seed=4)
    # O->H1 unit vectors should average to ~0 for uniform orientations
    v = np.array([(m[1] - m[0]) / np.linalg.norm(m[1] - m[0]) for m in placed])
    assert np.linalg.norm(v.mean(axis=0)) < 0.2


def test_molecules_sit_in_the_framework_image():
    """A framework stored around the origin gets its molecules in the unit window
    centered on the framework, each kept whole."""
    pbc = pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90)
    mol = framework(pbc)
    for a in mol.atoms:  # shift the framework to the centered image
        a.x = pbc.wrap(a.x)
    fw = np.array([a.x for a in mol.atoms]) @ pbc.reciprocal_basis_matrix
    assert fw.min() < -0.2  # really centered
    lo = fw.mean(axis=0) - 0.5
    model = pw.SORBATE_MODELS["h2o_dc97"]
    for xyz in pw.insert_sorbates(mol, model, 10, seed=9):
        com = xyz[0] @ pbc.reciprocal_basis_matrix  # O is ~ the center of mass
        assert np.all(com >= lo - 0.02) and np.all(com < lo + 1.02)
        assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(0.9572, abs=1e-4)


def test_progress_callback_counts():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    seen = []
    pw.insert_sorbates(mol, "h2o_tip3p", 4, seed=8, progress_callback=seen.append)
    assert seen == [1, 2, 3, 4]


def test_seed_is_reproducible():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    a = pw.insert_sorbates(mol, "h2o_dc97", 5, seed=7)
    b = pw.insert_sorbates(mol, "h2o_dc97", 5, seed=7)
    assert all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_requires_cell():
    with pytest.raises(pw.CellRequiredError):
        pw.insert_sorbates(pw.Molecule(atoms=[pw.Atom(0, 0, 0, "C")]), "h2o_tip3p", 1)


def test_format_and_mpmc_file(tmp_path):
    """Molecule IDs 2, 3, ...; box records after them; REMARK lines MPMC can parse."""
    mol = framework(pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90))
    placed = pw.insert_sorbates(mol, "h2o_dc97", 3, seed=5)
    lines = pw.format_inserted_sorbates("h2o_dc97", placed, start_atom_id=len(mol.atoms) + 1)
    assert len(lines) == 12
    assert [ln.split()[5] for ln in lines[::4]] == ["2", "3", "4"]
    assert all(ln.split()[4] == "M" for ln in lines)
    out = tmp_path / "packed.pdb"
    pw.write_mpmc_pdb(mol, str(out), sorbate_lines=lines)
    text = out.read_text().splitlines()
    box = [ln.split() for ln in text if ln.startswith("ATOM") and " BOX " in ln]
    assert box and all(t[5] == "5" for t in box)
    remarks = [ln.split() for ln in text if ln.startswith("REMARK BOX BASIS")]
    assert len(remarks) == 3 and all(t[3] == "=" for t in remarks)  # MPMC checks token 4


def test_read_sorbate_template(tmp_path):
    f = tmp_path / "tip5p.pqr"
    f.write_text(
        "ATOM 1 OXY  H2O M 2  0.0 -0.06556 0.0  15.9994 0.0 0.0 80.52 3.12 0 0 0 0 0\n"
        "ATOM 2 HYD  H2O M 2  0.75695 0.52032 0.0 1.008 0.241 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 3 HYD  H2O M 2 -0.75695 0.52032 0.0 1.008 0.241 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 4 LP   H2O M 2  0.0 -0.46971 0.57154 1.0e-6 -0.241 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 5 LP   H2O M 2  0.0 -0.46971 -0.57154 1.0e-6 -0.241 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 6 X    H2O M 3  9.0 9.0 9.0 1.0 0.0 0.0 0.0 0.0 0 0 0 0 0\n"
    )
    model = pw.read_sorbate_template(f)
    assert model.label == "H2O" and len(model.sites) == 5 and model.pot_type == "lj"
    assert model.sites[3].mass == pytest.approx(1e-6) and model.sites[0].epsilon == 80.52
    (tmp_path / "empty.pqr").write_text("ATOM 1 C MOF F 1 0 0 0 12 0 0 0 0\n")
    with pytest.raises(ValueError):
        pw.read_sorbate_template(tmp_path / "empty.pqr")


@pytest.mark.parametrize("name", ["h2o_spce", "h2o_tip5p", "h2o_dc97"])
def test_new_water_models_geometry(name):
    m = pw.SORBATE_MODELS[name]
    o = np.array([m.sites[0].x, m.sites[0].y, m.sites[0].z])
    h1, h2 = (np.array([s.x, s.y, s.z]) for s in m.sites[1:3])
    r_oh = {"h2o_spce": 1.0, "h2o_tip5p": 0.9572, "h2o_dc97": 0.9572}[name]
    ang = {"h2o_spce": 109.47, "h2o_tip5p": 104.52, "h2o_dc97": 104.52}[name]
    assert np.linalg.norm(h1 - o) == pytest.approx(r_oh, abs=1e-4)
    cosang = np.dot(h1 - o, h2 - o) / np.linalg.norm(h1 - o) / np.linalg.norm(h2 - o)
    assert np.degrees(np.arccos(cosang)) == pytest.approx(ang, abs=0.02)
    assert sum(s.charge for s in m.sites) == pytest.approx(0.0, abs=1e-9)


# ======================================================================
# Thorough checks
# ======================================================================

SKEWED = [
    (20, 20, 20, 90, 90, 90),
    (20.88, 13.3, 17.1, 90, 99.9, 90),
    (15, 17, 19, 70, 80, 100),
    (18, 18, 18, 60, 60, 60),
    (16, 18, 20, 50, 60, 70),
]
ALL_MODELS = sorted(pw.SORBATE_MODELS)


def brute_force_dists(pbc, a, b):
    """Shortest distance over all 27 neighbouring images, independent of the
    rounding minimum image used by the code under test."""
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ pbc.basis_matrix
    d = a[:, None, None, :] - b[None, :, None, :] - shifts[None, None, :, :]
    return np.sqrt((d * d).sum(axis=3)).min(axis=2)


def site_radius(model, site):
    return site.sigma / 2 if site.sigma > 0 and model.pot_type != "phahst" else 0.0


def check_rule(mol, model, placed, scale=0.85, min_dist=1.0):
    """Every site pair obeys d >= max(min_dist, scale * (r_i + r_j))."""
    fw = np.array([a.x for a in mol.atoms]).reshape(-1, 3)
    fw_r = np.array([a.sigma / 2 if a.sigma > 0 else a.vdw for a in mol.atoms])
    r = np.array([site_radius(model, s) for s in model.sites])
    for k, mk in enumerate(placed):
        if len(fw):
            lim = np.maximum(min_dist, scale * (r[:, None] + fw_r[None, :]))
            assert (brute_force_dists(mol.pbc, mk, fw) >= lim - 1e-9).all()
        for mj in placed[:k]:
            lim = np.maximum(min_dist, scale * (r[:, None] + r[None, :]))
            assert (brute_force_dists(mol.pbc, mk, mj) >= lim - 1e-9).all()


@pytest.mark.parametrize("cell", SKEWED)
@pytest.mark.parametrize("name", ["h2o_dc97", "h2o_tip5p", "co2_trappe", "h2_bss", "ch4_9site"])
def test_rule_holds_against_all_periodic_images(cell, name):
    """Checked by brute force over all 27 neighbouring images, independently of
    the rounding minimum image the code uses."""
    mol = framework(pw.PBC(*cell))
    model = pw.SORBATE_MODELS[name]
    placed = pw.insert_sorbates(mol, model, 12, seed=21)
    assert placed
    check_rule(mol, model, placed)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_built_in_model_packs_and_formats(name):
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    model = pw.SORBATE_MODELS[name]
    placed = pw.insert_sorbates(mol, name, 3, seed=22)
    assert len(placed) == 3
    check_rule(mol, model, placed)
    lines = pw.format_inserted_sorbates(name, placed, start_atom_id=1)
    assert len(lines) == 3 * len(model.sites)
    for ln in lines:
        assert pw._is_mpmc_record(ln.split())


def test_framework_without_sigma_uses_vdw_radius():
    pbc = pw.PBC(16, 16, 16, 90, 90, 90)
    mol = framework(pbc)
    for a in mol.atoms:
        a.sigma = 0.0
    placed = pw.insert_sorbates(mol, "h2o_spce", 6, seed=23)
    fw = np.array([a.x for a in mol.atoms])
    for mk in placed:  # O vs carbon: 0.85 * (3.166/2 + C vdW radius)
        assert (
            brute_force_dists(pbc, mk[[0]], fw).min()
            >= 0.85 * (3.166 / 2 + mol.atoms[0].vdw) - 1e-9
        )


def test_phahst_sites_only_use_the_floor():
    pbc = pw.PBC(12, 12, 12, 90, 90, 90)
    mol = framework(pbc)
    model = pw.SORBATE_MODELS["h2_phahst"]
    placed = pw.insert_sorbates(mol, model, None, seed=24, max_failures=2000)
    check_rule(mol, model, placed)
    fw = np.array([a.x for a in mol.atoms])
    closest = min(brute_force_dists(pbc, mk, fw).min() for mk in placed)
    # epsilon/sigma hold beta/rho for PHAHST, not LJ, so they must not act as radii
    assert closest < 0.85 * (model.sites[0].sigma / 2 + 1.7)


@pytest.mark.parametrize("scale,min_dist", [(0.7, 1.0), (1.0, 1.0), (0.85, 2.0)])
def test_scale_and_min_dist_are_respected(scale, min_dist):
    mol = framework(pw.PBC(18, 18, 18, 90, 90, 90))
    model = pw.SORBATE_MODELS["h2o_tip3p"]
    placed = pw.insert_sorbates(mol, model, 8, scale=scale, min_dist=min_dist, seed=25)
    assert placed
    check_rule(mol, model, placed, scale=scale, min_dist=min_dist)


def test_looser_rule_fits_more():
    mol = framework(pw.PBC(14, 14, 14, 90, 90, 90))
    tight = pw.insert_sorbates(mol, "h2o_spce", None, scale=1.0, seed=26, max_failures=1500)
    loose = pw.insert_sorbates(mol, "h2o_spce", None, scale=0.7, seed=26, max_failures=1500)
    assert len(loose) > len(tight)


def dense_framework(a=6.0, step=0.25):
    pbc = pw.PBC(a, a, a, 90, 90, 90)
    atoms = []
    for f in itertools.product(np.arange(0, 1, step), repeat=3):
        x, y, z = np.array(f) @ pbc.basis_matrix
        atom = pw.Atom(x, y, z, "C")
        atom.sigma = 3.4
        atoms.append(atom)
    return pw.Molecule(atoms=atoms, pbc=pbc)


def test_edge_cases_return_empty():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    assert pw.insert_sorbates(mol, "h2o_spce", 0, seed=1) == []
    assert pw.insert_sorbates(mol, "h2o_spce", 5, seed=1, max_failures=0) == []
    # framework atoms every 1.5 A leave no room at all
    assert pw.insert_sorbates(dense_framework(), "h2o_spce", 3, seed=1, max_failures=300) == []


def test_empty_cell_fills_without_framework():
    mol = pw.Molecule(atoms=[], pbc=pw.PBC(10, 10, 10, 90, 90, 90))
    placed = pw.insert_sorbates(mol, "h2o_spce", None, seed=27, max_failures=800)
    assert 10 < len(placed) < 60  # 1000 A^3; liquid water would hold ~33
    check_rule(mol, pw.SORBATE_MODELS["h2o_spce"], placed)


def test_model_name_and_object_are_equivalent():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    a = pw.insert_sorbates(mol, "h2o_dc97", 4, seed=28)
    b = pw.insert_sorbates(mol, pw.SORBATE_MODELS["h2o_dc97"], 4, seed=28)
    assert all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_input_molecule_is_not_modified():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    before = [a.x.copy() for a in mol.atoms]
    pw.insert_sorbates(mol, "h2o_dc97", 5, seed=29)
    assert len(mol.atoms) == len(before)
    assert all(np.array_equal(a.x, b) for a, b in zip(mol.atoms, before, strict=True))


def test_centers_are_uniform_in_an_empty_cell():
    pbc = pw.PBC(40, 40, 40, 90, 90, 90)
    mol = pw.Molecule(atoms=[], pbc=pbc)
    placed = pw.insert_sorbates(mol, "he", 800, seed=30)  # one site = the center
    frac = np.array([m[0] for m in placed]) @ pbc.reciprocal_basis_matrix
    assert np.all(frac >= -1e-9) and np.all(frac < 1 + 1e-9)
    counts = np.bincount((frac > 0.5).astype(int) @ [1, 2, 4], minlength=8)
    assert counts.min() > 70 and counts.max() < 130  # 100 expected per octant


def test_rotations_are_uniform():
    rng = np.random.default_rng(31)
    mats = np.array([pw._random_rotation(rng) for _ in range(4000)])
    assert np.allclose(mats.mean(axis=0), 0.0, atol=0.05)  # uniform (Haar) rotations average to 0
    assert np.allclose(np.einsum("nij,nkj->nik", mats, mats), np.eye(3), atol=1e-12)


def test_center_of_mass_is_the_placement_center():
    pbc = pw.PBC(30, 30, 30, 90, 90, 90)
    mol = pw.Molecule(atoms=[], pbc=pbc)
    model = pw.SORBATE_MODELS["h2o_tip4p"]
    masses = np.array([s.mass for s in model.sites])
    for xyz in pw.insert_sorbates(mol, model, 50, seed=32):
        frac = (masses @ xyz / masses.sum()) @ pbc.reciprocal_basis_matrix
        assert np.all(frac >= -1e-9) and np.all(frac < 1 + 1e-9)


# ---- templates ------------------------------------------------------------------------


def test_template_reader_variants(tmp_path):
    f = tmp_path / "co2.pqr"
    f.write_text(
        "REMARK a comment\n"
        "HETATM 1 C   CO2 M 7 0.0 0.0 0.0 12.011 0.7 1.2 27.0 2.8\n"
        "ATOM 2 O   CO2 M 7 1.16 0.0 0.0 15.999 -0.35 0.8 79.0 3.05\n"
        "ATOM 3 O   CO2 M 7 -1.16 0.0 0.0 15.999 -0.35 0.8 79.0 3.05\n"
        "ATOM 4 C   MOF F 1 5.0 5.0 5.0 12.011 0.0 0.0 50.0 3.4\n"
    )
    m = pw.read_sorbate_template(f)
    assert (m.name, m.label, len(m.sites), m.pot_type) == ("co2", "CO2", 3, "lj_polar")
    assert m.sites[1].x == 1.16 and m.sites[2].charge == -0.35 and m.sites[0].c6 == 0.0
    ph = tmp_path / "h2.pqr"
    ph.write_text("ATOM 1 H2G H2 M 2 0 0 0 0.0 -0.7 0.69 3.6 2.4 0 0 12.0 200.0 4000.0\n")
    assert pw.read_sorbate_template(ph).pot_type == "phahst"
    with pytest.raises(OSError):
        pw.read_sorbate_template(tmp_path / "missing.pqr")


def test_template_packs_like_the_built_in_model(tmp_path):
    model = pw.SORBATE_MODELS["h2o_dc97"]
    lines = pw.format_sorbate_pqr("h2o_dc97", 0, 0, 0, mol_id=2, start_atom_id=1)
    (tmp_path / "w.pqr").write_text("\n".join(lines) + "\nEND\n")
    t = pw.read_sorbate_template(tmp_path / "w.pqr")
    params = [(s.atomtype, s.charge, s.polarizability, s.epsilon, s.sigma) for s in t.sites]
    assert params == [
        (s.atomtype, s.charge, s.polarizability, s.epsilon, s.sigma) for s in model.sites
    ]
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    a = pw.insert_sorbates(mol, t, 4, seed=33)
    b = pw.insert_sorbates(mol, model, 4, seed=33)
    assert all(np.allclose(x, y, atol=1e-5) for x, y in zip(a, b, strict=True))


# ---- formatting and the MPMC file -----------------------------------------------------


def test_formatted_records_round_trip_through_the_reader(tmp_path):
    pbc = pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90)
    mol = framework(pbc)
    placed = pw.insert_sorbates(mol, "h2o_tip5p", 4, seed=34)
    lines = pw.format_inserted_sorbates("h2o_tip5p", placed, start_atom_id=len(mol.atoms) + 1)
    ids = [int(ln.split()[1]) for ln in lines]
    assert ids == list(range(len(mol.atoms) + 1, len(mol.atoms) + 1 + len(lines)))
    out = tmp_path / "p.pqr"
    pw.write_mpmc_pdb(mol, str(out), sorbate_lines=lines)
    with open(out) as fh:
        atoms, pbc2 = pw.read_pdb(fh, str(out))
    assert pbc2 is not None and np.allclose(pbc2.basis_matrix, pbc.basis_matrix, atol=1e-3)
    back = np.array([a.x for a in atoms[-len(lines) :]])
    assert np.allclose(back, np.vstack(placed), atol=1e-5)
    assert [a.charge for a in atoms[-5:]] == [
        s.charge for s in pw.SORBATE_MODELS["h2o_tip5p"].sites
    ]


def test_remark_lines_carry_the_basis():
    pbc = pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90)
    buf = io.StringIO()
    pw.write_mpmc_pdb(framework(pbc), buf)
    rows = [ln.split() for ln in buf.getvalue().splitlines() if ln.startswith("REMARK BOX")]
    assert [r[2] for r in rows] == ["BASIS[0]", "BASIS[1]", "BASIS[2]"]
    assert all(r[3] == "=" for r in rows)
    assert np.allclose([[float(v) for v in r[4:7]] for r in rows], pbc.basis_matrix)


def box_ids(text):
    return {ln.split()[5] for ln in text.splitlines() if ln.startswith("ATOM") and " BOX " in ln}


def test_box_id_without_sorbates_and_with_odd_lines():
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    buf = io.StringIO()
    pw.write_mpmc_pdb(mol, buf)
    assert box_ids(buf.getvalue()) == {"2"}
    buf = io.StringIO()
    pw.write_mpmc_pdb(mol, buf, sorbate_lines=["ATOM 1 X H2O M x 0 0 0 0 0 0 0 0"])
    assert box_ids(buf.getvalue()) == {"2"}


# ---- classic menu ---------------------------------------------------------------------


def run_menu(monkeypatch, mol, answers):
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_: next(feed))
    pw._menu_insert_sorbates(mol)


def count_m_molecules(path):
    return len(
        {
            ln.split()[5]
            for ln in Path(path).read_text().splitlines()
            if ln.startswith("ATOM") and ln.split()[4] == "M"
        }
    )


def test_menu_by_number_and_count(monkeypatch, tmp_path, capsys):
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    idx = str(list(pw.SORBATE_MODELS).index("h2o_dc97"))
    out = tmp_path / "a.pqr"
    run_menu(monkeypatch, mol, [idx, "4", str(out)])
    assert count_m_molecules(out) == 4
    assert "Placed 4 H2O" in capsys.readouterr().out


def test_menu_by_name_fill_and_retries(monkeypatch, tmp_path, capsys):
    mol = framework(pw.PBC(12, 12, 12, 90, 90, 90))
    out = tmp_path / "b.pqr"
    run_menu(monkeypatch, mol, ["not_a_model", "h2o_spce", "zero", "-3", "fill", str(out)])
    text = capsys.readouterr().out
    assert "not_a_model" in text  # template lookup failed, asked again
    assert text.count("Enter a positive number or 'fill'") == 2
    assert count_m_molecules(out) > 0


def test_menu_template_path_and_cell_full(monkeypatch, tmp_path, capsys):
    lines = pw.format_sorbate_pqr("h2o_tip3p", 0, 0, 0, mol_id=2, start_atom_id=1)
    tpl = tmp_path / "t.pqr"
    tpl.write_text("\n".join(lines) + "\n")
    out = tmp_path / "c.pqr"
    run_menu(monkeypatch, framework(pw.PBC(12, 12, 12, 90, 90, 90)), [str(tpl), "500", str(out)])
    assert "asked for 500; the cell is full" in capsys.readouterr().out
    assert 0 < count_m_molecules(out) < 500


def test_menu_cancel_and_no_cell(monkeypatch, tmp_path, capsys):
    run_menu(monkeypatch, framework(pw.PBC(20, 20, 20, 90, 90, 90)), ["q"])
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not prompt"))
    pw._menu_insert_sorbates(pw.Molecule(atoms=[pw.Atom(0, 0, 0, "C")]))
    assert "needs a unit cell" in capsys.readouterr().out


def test_menu_nothing_fits_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pw, "insert_sorbates", lambda *a, **k: [])
    run_menu(monkeypatch, framework(pw.PBC(20, 20, 20, 90, 90, 90)), ["h2o_spce", "3"])
    assert "Placed 0" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_main_loop_option_10_reaches_the_menu(monkeypatch):
    called = []
    monkeypatch.setattr(pw, "_menu_insert_sorbates", called.append)
    monkeypatch.setattr(pw, "overlap_detector", lambda m: m)
    monkeypatch.setattr(pw, "print_info", lambda *a: None)
    feed = iter(["10", "9"])
    monkeypatch.setattr("builtins.input", lambda *_: next(feed))
    mol = framework(pw.PBC(20, 20, 20, 90, 90, 90))
    pw._main_loop_single(mol, "x.pqr")
    assert called == [mol]


# ---- TUI steps, called on a stand-in for the app -------------------------------------


class FakeApp:
    def __init__(self, mol=None):
        self.molecule = mol
        self.filepath = "/tmp/mof.pqr"
        self._mpmc_state = {"write_charges": False, "write_ff": False}
        self.pushed, self.notes, self.files, self.steps = [], [], {}, []

    def push_screen(self, screen, callback=None):
        self.pushed.append((screen, callback))

    def notify(self, msg, **kw):
        self.notes.append(msg)

    def _mpmc_step_filename(self):
        self.steps.append("filename")

    def _mpmc_step_sorbate_result(self, name):
        pw.PdbWizardApp.__dict__["_mpmc_step_sorbate_result"](self, name)

    def _mpmc_step_sorbate_count(self, value):
        pw.PdbWizardApp.__dict__["_mpmc_step_sorbate_count"](self, value)

    def _write_text_file(self, filepath, writer, what):
        buf = io.StringIO()
        writer(buf)
        self.files[filepath] = buf.getvalue()
        return True


def app_method(name):
    return pw.PdbWizardApp.__dict__[name]


def test_tui_sorbate_choice_asks_how_many(monkeypatch):
    seen = []
    monkeypatch.setattr(pw, "InputModal", lambda *a, **k: seen.append((a, k)) or "modal")
    app = FakeApp()
    app_method("_mpmc_step_sorbate_result")(app, "h2o_dc97")
    assert app._mpmc_state["sorbate"] == "h2o_dc97" and app._mpmc_state["sorbate_n"] == 1
    assert seen and "How many" in seen[0][0][0] and app.pushed[0][0] == "modal"
    assert app.steps == []  # waits for the count first


def test_tui_skip_and_cancel():
    app = FakeApp()
    app_method("_mpmc_step_sorbate_result")(app, "")
    assert app._mpmc_state["sorbate"] is None and app.steps == ["filename"]
    app = FakeApp()
    app_method("_mpmc_step_sorbate_result")(app, None)
    assert "sorbate" not in app._mpmc_state and not app.pushed and not app.steps


@pytest.mark.parametrize("raw,expect", [("fill", None), (" FILL ", None), ("7", 7), ("1", 1)])
def test_tui_count_accepts(raw, expect):
    app = FakeApp()
    app._mpmc_state["sorbate"] = "h2o_spce"
    app_method("_mpmc_step_sorbate_count")(app, raw)
    assert app._mpmc_state["sorbate_n"] == expect and app.steps == ["filename"]


@pytest.mark.parametrize("raw", ["0", "-2", "abc", "3.5"])
def test_tui_count_rejects_and_asks_again(raw, monkeypatch):
    monkeypatch.setattr(pw, "InputModal", lambda *a, **k: "modal")
    app = FakeApp()
    app._mpmc_state["sorbate"] = "h2o_spce"
    app_method("_mpmc_step_sorbate_count")(app, raw)
    assert app.notes and app.steps == [] and app.pushed  # warned, then asked again


@pytest.mark.parametrize("value", [None, "", "   "])
def test_tui_count_cancel(value):
    """InputModal returns "" for Cancel/Escape, so that has to abort, not re-ask."""
    app = FakeApp()
    app._mpmc_state["sorbate"] = "h2o_spce"
    app_method("_mpmc_step_sorbate_count")(app, value)
    assert not app.steps and not app.notes and not app.pushed


def m_molecules(text):
    return len(
        {
            ln.split()[5]
            for ln in text.splitlines()
            if ln.startswith("ATOM") and ln.split()[4] == "M"
        }
    )


def test_tui_write_inserts_n_molecules():
    app = FakeApp(framework(pw.PBC(20, 20, 20, 90, 90, 90)))
    app._mpmc_state.update(sorbate="h2o_dc97", sorbate_n=5)
    app_method("_write_mpmc_file")(app, "out.pqr")
    assert m_molecules(app.files["out.pqr"]) == 5 and not app.notes


def test_tui_write_reports_full_cell_and_no_room(monkeypatch):
    app = FakeApp(framework(pw.PBC(12, 12, 12, 90, 90, 90)))
    app._mpmc_state.update(sorbate="h2o_spce", sorbate_n=None)
    app_method("_write_mpmc_file")(app, "full.pqr")
    assert "full.pqr" in app.files and any("cell full" in n for n in app.notes)
    monkeypatch.setattr(pw, "insert_sorbates", lambda *a, **k: [])
    app2 = FakeApp(framework(pw.PBC(9, 9, 9, 90, 90, 90)))
    app2._mpmc_state.update(sorbate="h2o_spce", sorbate_n=3)
    app_method("_write_mpmc_file")(app2, "none.pqr")
    assert "none.pqr" not in app2.files and any("No room" in n for n in app2.notes)


def test_tui_write_without_sorbate():
    app = FakeApp(framework(pw.PBC(20, 20, 20, 90, 90, 90)))
    app._mpmc_state.update(sorbate=None)
    app_method("_write_mpmc_file")(app, "bare.pqr")
    assert m_molecules(app.files["bare.pqr"]) == 0


# ---- model data -----------------------------------------------------------------------


def test_new_models_details():
    tip5p = pw.SORBATE_MODELS["h2o_tip5p"]
    lp = [np.array([s.x, s.y, s.z]) for s in tip5p.sites if s.atomtype == "LP"]
    h = [np.array([s.x, s.y, s.z]) for s in tip5p.sites if s.atomtype == "HYD"]
    assert all(np.linalg.norm(v) == pytest.approx(0.70, abs=1e-4) for v in lp)
    assert np.degrees(np.arccos(np.dot(lp[0], lp[1]) / 0.70**2)) == pytest.approx(109.47, abs=0.02)
    assert np.dot(lp[0] + lp[1], h[0] + h[1]) < 0  # lone pairs opposite the hydrogens
    dc97 = pw.SORBATE_MODELS["h2o_dc97"]
    m = next(s for s in dc97.sites if s.atomtype == "M")
    hsum = sum(np.array([s.x, s.y, s.z]) for s in dc97.sites if s.atomtype == "HYD")
    mv = np.array([m.x, m.y, m.z])
    assert np.linalg.norm(mv) == pytest.approx(0.215) and np.dot(mv, hsum) > 0  # M on the H side
    assert (m.polarizability, m.charge, m.mass) == (1.444, -1.038, 0)
    for name in ("h2o_spce", "h2o_tip5p", "h2o_dc97"):
        assert sum(s.mass for s in pw.SORBATE_MODELS[name].sites) == pytest.approx(18.015, abs=0.01)


# ---- optional: real MPMC (pytest --mpmc /path/to/mpmc) --------------------------------


@pytest.fixture
def mpmc_bin(request):
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("pass --mpmc /path/to/mpmc to run MPMC itself")
    return path


def test_mpmc_reads_the_packed_file_with_sane_energy(tmp_path, mpmc_bin):
    mol = framework(pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90))
    placed = pw.insert_sorbates(mol, "h2o_spce", 10, seed=35)
    lines = pw.format_inserted_sorbates("h2o_spce", placed, start_atom_id=len(mol.atoms) + 1)
    pw.write_mpmc_pdb(mol, str(tmp_path / "in.pqr"), sorbate_lines=lines)
    (tmp_path / "sp.inp").write_text(
        "job_name sp\nensemble total_energy\ntemperature 298\npqr_input in.pqr\nread_pqr_box on\n"
    )
    out = subprocess.run(
        [mpmc_bin, "sp.inp"], cwd=tmp_path, capture_output=True, text=True, timeout=300
    ).stdout
    assert "basis[2] successfully read" in out
    energy = float(out.split("potential energy =")[1].split()[0])
    # 0.85*sigma lets a few contacts be mildly repulsive (~1e3 K); a real overlap is 1e10+ K
    assert -1e6 < energy < 1e4


def test_format_rejects_a_placement_with_the_wrong_site_count():
    """Without zip(strict=True) (Python 3.10+), the length check is explicit."""
    good = pw.insert_sorbates(
        pw.Molecule(atoms=[], pbc=pw.PBC(20, 20, 20, 90, 90, 90)), "h2o_spce", 1, seed=1
    )
    with pytest.raises(ValueError, match="has 2 sites"):
        pw.format_inserted_sorbates("h2o_spce", [good[0][:2]], start_atom_id=1)
