"""Tests for the modules both front ends share: analyses, inputgen, trajectory, the database
search helper and the new geometry helpers, plus the RDF normalisation fix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdb_wizard import analyses, database, inputgen, trajectory
from pdb_wizard.atom import Atom
from pdb_wizard.geometry import coordination_polyhedra, water_atom_indices
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory
from pdb_wizard.templates import ENGINES

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"


def ideal_gas(n=1500, L=30.0, seed=0):
    rng = np.random.default_rng(seed)
    atoms = [Atom(*x, "Ar") for x in rng.random((n, 3)) * L]
    atoms += [Atom(*x, "Ne") for x in rng.random((n, 3)) * L]
    return Molecule(atoms=atoms, pbc=PBC(L, L, L, 90, 90, 90))


def waters(n_frames=6, named=True):
    rng = np.random.default_rng(3)
    centres = rng.uniform(2, 13, (6, 3))
    frames = []
    for f in range(n_frames):
        atoms = []
        for c in centres:
            o = Atom(*(c + [0.2 * f, 0, 0]), "O")
            if named:
                o.name = "OW"
            atoms += [o, Atom(*(o.x + [0.96, 0, 0]), "H"), Atom(*(o.x + [-0.24, 0.93, 0]), "H")]
        m = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
        m.detect_bonds()
        frames.append(m)
    return frames


# ---- RDF -------------------------------------------------------------------------------


def test_rdf_tends_to_one_for_an_ideal_gas_same_and_cross_type():
    """Regression: the terminal viewer's RDF counted same-type pairs once but normalised
    as if all ordered pairs were counted, so its O-O g(r) levelled off at 0.5."""
    gas = ideal_gas()
    for a, b in (("Ar", "Ar"), ("Ar", "Ne")):
        res = analyses.run_rdf(gas, a, b, r_max=13.0, n_bins=100)
        assert res.y[res.x > 6].mean() == pytest.approx(1.0, abs=0.02), (a, b)


def test_rdf_name_filters():
    frames = waters()
    mol = frames[0]
    r_el, g_el = compute_rdf(mol, "O", "O", 50, 6.0)
    r_nm, g_nm = compute_rdf(mol, "O", "O", 50, 6.0, name1="OW", name2="OW")
    np.testing.assert_array_equal(g_el, g_nm)  # every O is OW: the same selection
    r, g = compute_rdf(mol, "O", "O", 50, 6.0, name1="OW", name2=None)
    np.testing.assert_allclose(g, g_el)  # identical index sets count as one type
    assert compute_rdf(mol, "O", "O", name1="nope")[0].size == 0
    r_t, g_t = compute_rdf_trajectory(frames, "O", "H", 50, 6.0, name1="OW")
    assert len(r_t) == 50 and g_t.max() > 0


def test_rdf_overlapping_selections_skip_self_pairs():
    """All O against a subset of O: an atom is never paired with itself (no r=0 spike)."""
    mol = waters(1, named=False)[0]
    mol.atoms[0].name = "OX"
    r, g = compute_rdf(mol, "O", "O", 40, 6.0, name1=None, name2="OX")
    assert g[0] == 0


# ---- analyses ---------------------------------------------------------------------------


def test_frame_slice_names_and_elements():
    frames = waters()
    assert analyses.frame_slice(frames, 2, 4) == frames[1:4]
    assert analyses.frame_slice(frames) == frames
    assert analyses.frame_slice(frames[:1], 3, 9) == frames[:1]
    assert analyses.frame_slice(None) is None
    assert analyses.elements_of(frames[0]) == ["H", "O"]
    assert analyses.names_of(frames[0], "O") == ["OW"]
    assert "OW" in analyses.names_of(frames[0], analyses.ALL_ATOMS)


def test_unavailable_messages_match_the_terminal_viewer():
    mol = waters(1)[0]
    frames = waters(3)
    assert analyses.unavailable("msd", mol, None) == "Need a trajectory with 3+ frames for MSD"
    assert analyses.unavailable("rmsd", mol, frames[:2]).endswith("for RMSD")
    assert analyses.unavailable("msd", mol, frames) is None
    assert analyses.unavailable("gyration", mol, None) == "Need a trajectory for Rg"
    assert analyses.unavailable("density3d", mol, None).startswith("Need a trajectory")
    nopbc = Molecule(atoms=list(mol.atoms))
    assert analyses.unavailable("density3d", nopbc, frames) == "Need PBC for 3D density map"
    assert analyses.unavailable("pxrd", nopbc, None).startswith("Need a crystal")
    assert analyses.unavailable("void_vol", nopbc, None) == "Need PBC for void volume"
    assert analyses.unavailable("surface_area", nopbc, None) == "Need PBC for surface area"
    assert analyses.unavailable("pore_size", nopbc, None) == "Need PBC for pore size"
    assert analyses.unavailable("rdf", Molecule(atoms=[]), None) == "No atoms loaded yet"
    assert analyses.unavailable("hbonds", Molecule(atoms=[]), None) == "No atoms loaded"
    assert analyses.unavailable("info", Molecule(atoms=[]), None) is None
    assert analyses.unavailable("pxrd", Molecule(atoms=[]), None).startswith("Need a crystal")
    assert analyses.unavailable("energy_plot", None, None) is None


def test_runs_report_missing_data():
    mol = waters(1)[0]
    frames = waters()
    assert analyses.run_rdf(mol, "Xe", "O").summary == "No Xe-O pairs found"
    assert analyses.run_coordination(mol, "Xe", "O").summary == "No Xe or O atoms found"
    assert analyses.run_msd(frames, "Xe").summary == "No Xe atoms found"
    assert analyses.run_rmsd(frames, "Xe").summary == "No matching atoms found"
    assert analyses.run_density_profile(mol, "Xe").summary == "No Xe atoms found"
    nocell = analyses.run_pore_size(Molecule(atoms=list(mol.atoms)))
    assert nocell.empty and nocell.summary == "No PBC for pore size calculation"
    short = analyses.run_msd(frames, "O")  # 6 frames: too few lags for a diffusion fit
    assert short.summary == "Done" and short.extra["D"] is None
    long = analyses.run_msd(waters(24), "O", dt=2.0)  # lags up to half the frames: 12
    assert long.summary.startswith("D = ") and long.extra["D"] > 0
    np.testing.assert_array_equal(long.x, long.csv_arrays()[0] * 2.0)
    hb = analyses.run_hbonds(mol)
    assert hb.x.tolist() == [0.0] and hb.text.startswith("donor")
    rg = analyses.run_gyration(frames, "O", dt=0.5, start=2, end=4)
    assert len(rg.x) == 3 and rg.csv_header == "frame,rg_A"


def test_energy_helpers(tmp_path):
    data = {"step": np.arange(1.0, 11.0), "energy": np.linspace(-5, -1, 10),
            "N": np.arange(10.0)}
    assert analyses.energy_columns(data) == (["energy", "N"], "N")
    assert analyses.energy_columns({"step": [1], "x": [2]}) == (["x"], "x")
    assert analyses.energy_n_steps(data) == 10
    assert analyses.energy_slice(data, 3, 5) == slice(2, 5)
    assert analyses.energy_slice(data, 0, 99) == slice(0, 10)
    assert analyses.energy_slice(data, 9, 2) == slice(8, 9)
    assert analyses.energy_slice({}) == slice(0, 0)
    res = analyses.run_energy(data, "energy", 3, 8)
    assert res.title == "energy (steps 3-8)"
    assert res.summary.startswith("energy [3-8]: last-quarter avg = ")
    assert analyses.run_energy(data, "nope").summary == "No column 'nope'"
    full = analyses.run_energy(data, "N")
    assert full.title == "N" and full.y_max > full.y_min
    out = tmp_path / "e.csv"
    analyses.write_energy_csv(data, str(out), slice(2, 4))
    lines = out.read_text().splitlines()
    assert lines[0] == "step,energy,N" and len(lines) == 3


def test_experimental_pxrd_formats(tmp_path):
    files = {"hdr.csv": "two_theta,intensity\n5,10\n6,40\n7,20\n",
             "plain.csv": "5,10\n6,40\n7,20\n",
             "pattern.xy": "# comment\n5 10\n6 40\n7 20\n"}
    for name, text in files.items():
        p = tmp_path / name
        p.write_text(text)
        tt, inten = analyses.read_experimental_pxrd(str(p))
        assert tt.tolist() == [5, 6, 7] and inten.max() == pytest.approx(100.0), name
    bad = tmp_path / "one.xy"
    bad.write_text("5\n6\n7\n")
    with pytest.raises(ValueError):
        analyses.read_experimental_pxrd(str(bad))


def test_reports_and_csv(tmp_path):
    mol = read_file(str(HKUST))
    text = analyses.system_info_text(mol, str(HKUST))
    first = text.splitlines()[0]
    assert first.startswith("Filename:  ") and first.endswith("HKUST-1.pdb")
    assert "Formula:   C6  Cu1  H2  O4" in text
    report = analyses.void_volume_report({"void_volume": 1, "total_volume": 2,
                                          "void_fraction": 0.5}, {}, 1000, 1.14)
    assert "Void fraction: 50.0%" in report and "He probe (r = 1.14 A):" in report
    assert "(N2-sized)" not in analyses.surface_area_text(mol, 2000, 1.2)
    for a in mol.atoms:
        a.charge = 0.1 if a.element.symbol == "Cu" else -0.05
    q = analyses.charge_report(mol)
    assert "Per-element averages:" in q and "Cu" in q
    out = tmp_path / "x.csv"
    analyses.write_csv(str(out), "a,b", [1, 2], [3, 4])
    assert out.read_text() == "a,b\n1.000000,3.000000\n2.000000,4.000000\n"
    p = tmp_path / "energy.dat"
    p.write_text("1 2\n")
    assert analyses.find_energy_file(str(tmp_path / "s.pdb")) == str(p)
    assert analyses.find_energy_file(str(tmp_path / "sub" / "s.pdb")) is None
    assert analyses.run_isotherm(str(tmp_path)).summary == "No completed pressure points found"


# ---- input generator --------------------------------------------------------------------


def _values(engine):
    return {v.name: inputgen.parse_form_value(v, v.default if not isinstance(v.default, list)
                                              else ",".join(map(str, v.default)))
            for v in ENGINES[engine].user_vars}


def test_parse_form_value():
    var = {v.name: v for v in ENGINES["mpmc_isotherm"].user_vars}
    assert inputgen.parse_form_value(var["pressures"], "0.1, 1,5") == [0.1, 1.0, 5.0]
    assert inputgen.parse_form_value(var["temperature"], " 77 ") == 77.0
    assert inputgen.parse_form_value(var["numsteps"], 5000) == 5000
    assert inputgen.parse_form_value(var["sorbate"], "h2_bss") == "h2_bss"
    for name, raw in (("pressures", " , "), ("temperature", "  ")):
        with pytest.raises(ValueError, match="is empty"):
            inputgen.parse_form_value(var[name], raw)


def test_default_outdir_and_template():
    assert inputgen.default_engine_outdir("/a/b/x.cif", "cp2k_dft_md") == "/a/b/cp2k_md"
    assert inputgen.default_engine_outdir("", "weird") == "sim_input"
    out = inputgen.engine_template_text("cp2k_cell_opt")
    assert out.ok and out.preview.startswith("-- cp2k_cell_opt.inp --")


def test_checks_and_generation(tmp_path):
    mol = read_file(str(HKUST))
    nopbc = Molecule(atoms=list(mol.atoms))
    assert inputgen.check_engine(nopbc, "cp2k_cell_opt").status == "Fix issues before Generate"
    assert "OpenMM NPT" in inputgen.check_engine(nopbc, "openmm_npt").preview
    assert inputgen.check_engine(nopbc, "openmm_nvt").ok
    mpmc = inputgen.check_engine(mol, "mpmc_isotherm")
    assert mpmc.preview and mpmc.status
    for engine, expect in (("cp2k_cell_opt", "input.inp"), ("cp2k_dft_md", "input.inp"),
                           ("openmm_npt", "run.py"), ("openmm_nvt", "run.py")):
        out = inputgen.generate_engine_inputs(mol, engine, _values(engine), str(tmp_path / engine))
        assert out.ok, out.status
        assert (tmp_path / engine / expect).exists()
        assert out.status.startswith("Wrote ")
    out = inputgen.generate_engine_inputs(nopbc, "cp2k_cell_opt", _values("cp2k_cell_opt"),
                                          str(tmp_path / "x"))
    assert out.status == "CP2K requires a periodic box"
    assert inputgen.generate_engine_inputs(mol, "cp2k_cell_opt", {}, "").status == (
        "Output dir is required")
    assert inputgen.generate_engine_inputs(mol, "nope", {}, "d").status == "Unknown engine: nope"
    broken = inputgen.generate_engine_inputs(mol, "cp2k_cell_opt", {}, str(tmp_path / "y"))
    assert broken.status.startswith("Error: ")


def test_mpmc_generation_refuses_on_errors_and_runs_when_clean(tmp_path, monkeypatch):
    mol = read_file(str(HKUST))
    monkeypatch.setattr(inputgen, "validate_mpmc_inputs", lambda m, s=None: (["no charges"], []))
    out = inputgen.generate_engine_inputs(mol, "mpmc_isotherm", _values("mpmc_isotherm"),
                                          str(tmp_path / "a"))
    assert not out.ok and out.status.startswith("Refusing: 1 error ")
    assert "✗ no charges" in out.preview
    monkeypatch.setattr(inputgen, "validate_mpmc_inputs", lambda m, s=None: ([], ["small box"]))
    values = _values("mpmc_isotherm")
    values["pressures"] = [0.1, 1.0]
    out = inputgen.generate_engine_inputs(mol, "mpmc_isotherm", values, str(tmp_path / "b"))
    assert out.ok and out.status.endswith("(1 warnings)")
    assert "Generated 2 pressure points" in out.preview
    assert "⚠ 1 warning" in inputgen.format_validation([], ["small box"])
    assert inputgen.format_validation([], []).startswith("✓ All checks passed")
    assert inputgen.check_engine(mol, "mpmc_isotherm").status.startswith("1 warning")
    monkeypatch.setattr(inputgen, "validate_mpmc_inputs", lambda m, s=None: (["e1", "e2"], []))
    assert inputgen.check_engine(mol, "mpmc_isotherm").status == "2 errors — fix before Generate"
    monkeypatch.setattr(inputgen, "validate_mpmc_inputs", lambda m, s=None: ([], []))
    assert inputgen.check_engine(mol, "mpmc_isotherm").status == "Ready to generate"


# ---- database search --------------------------------------------------------------------


def test_database_helpers(monkeypatch):
    assert database.modes_for("all") == database.ALL_MODES
    assert database.modes_for("mp") == ("formula", "id")
    assert database.modes_for("nope") == ("text",)
    assert "formula across all DBs" in database.placeholder_for("all", "formula")
    assert database.placeholder_for("all", "text") == "free-text search across all DBs"
    assert database.placeholder_for("rcsb", "id") == "1VII"
    assert set(database.backends_for("all", "formula")) == {"cod", "mp"}
    assert database.backends_for("pubchem", "text") == ["pubchem"]

    def fake(code):
        return lambda q, mode="text": [{"source": code, "id": f"{code}-{q}", "formula": "X",
                                        "title": mode}]

    def boom(q, mode="text"):
        raise TimeoutError

    for code in database.BACKENDS:
        monkeypatch.setitem(database.BACKENDS[code], "search", fake(code))
    monkeypatch.setitem(database.BACKENDS["rcsb"], "search", boom)
    results, msg = database.search_backends("all", "text", "q")
    assert [r["source"] for r in results] == ["cod", "pubchem"]
    assert msg.startswith("2 results  (COD=1, RCSB PDB=0, PubChem=1)")
    assert msg.endswith("errors: RCSB PDB (TimeoutError)")
    results, msg = database.search_backends("cod", "formula", "Cu O")
    assert msg == "1 results in COD" and results[0]["title"] == "formula"


# ---- trajectories -----------------------------------------------------------------------


def write_xyz_trajectory(path, frames):
    with open(path, "w") as f:
        for m in frames:
            f.write(f"{len(m.atoms)}\nframe\n")
            for a in m.atoms:
                f.write(f"{a.element.symbol} {a.x[0]:.6f} {a.x[1]:.6f} {a.x[2]:.6f}\n")


def test_trajectory_helpers(tmp_path):
    frames = waters(4)
    path = tmp_path / "traj.xyz"
    write_xyz_trajectory(path, frames)
    single = tmp_path / "one.xyz"
    write_xyz_trajectory(single, frames[:1])
    assert trajectory.is_trajectory_file(str(path))
    assert not trajectory.is_trajectory_file(str(single))
    assert not trajectory.is_trajectory_file(str(HKUST))
    read = trajectory.read_trajectory(str(path))
    assert len(read) == 4
    np.testing.assert_allclose(read[2].atoms[0].x, frames[2].atoms[0].x, atol=1e-5)
    notes = trajectory.trajectory_caveats(read)
    assert any("no cell information" in n for n in notes)
    assert trajectory.trajectory_caveats([]) == []
    read[0].detect_bonds()
    trajectory.frame_bonds(read, 2)
    assert read[2].bonds is read[0].bonds
    odd = [read[0], Molecule(atoms=list(read[1].atoms[:3]))]
    trajectory.frame_bonds(odd, 1)  # different atom count: detect its own bonds
    assert odd[1].bonds and odd[1].bonds is not read[0].bonds


# ---- geometry helpers -------------------------------------------------------------------


def test_water_atom_indices():
    mol = waters(1)[0]
    extra = Atom(*(mol.atoms[0].x + [0, 0, 0.97]), "H")  # an H3O+: not water
    mol.atoms.append(extra)
    mol.detect_bonds()
    water = water_atom_indices(mol)
    assert len(water) == 15 and 0 not in water and len(mol.atoms) - 1 not in water
    assert water_atom_indices(Molecule(atoms=list(mol.atoms))) == set()  # no bonds yet


def test_coordination_polyhedra_use_the_minimum_image():
    mol = read_file(str(HKUST))
    polys = coordination_polyhedra(mol.atoms, mol.pbc)
    cu = [i for i, a in enumerate(mol.atoms) if a.element.symbol == "Cu"]
    assert {i for i, _ in polys} == set(cu)
    for i, nbrs in polys:
        d = np.linalg.norm(nbrs - mol.atoms[i].position, axis=1)
        assert len(nbrs) >= 3 and d.max() < 2.5 * mol.atoms[i].element.covalent_radius + 0.5
    lone = [Atom(0, 0, 0, "Zn"), Atom(2, 0, 0, "O")]
    assert coordination_polyhedra(lone) == []
    assert coordination_polyhedra([]) == []
    edge = [Atom(0.1, 5, 5, "Zn")] + [Atom(9.0, 5 + dy, 5 + dz, "O")
                                      for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    (ci, nbrs), = coordination_polyhedra(edge, PBC(10, 10, 10, 90, 90, 90))
    assert ci == 0 and np.all(nbrs[:, 0] < 0)  # wrapped across the face


def test_trajectory_helpers_read_dcd(tmp_path):
    """DCD without a topology file: carbon placeholders, and the caveat says so."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_dcd import _make_dcd

    coords = np.random.default_rng(2).uniform(0, 10, (5, 4, 3)).astype(np.float32)
    path = tmp_path / "run.dcd"
    path.write_bytes(_make_dcd(coords))
    assert trajectory.is_trajectory_file(str(path))
    seen = []
    frames = trajectory.read_trajectory(str(path), seen.append)
    assert len(frames) == 5 and len(frames[0].atoms) == 4 and seen
    assert any("DCD has no element info" in n for n in trajectory.trajectory_caveats(frames))
