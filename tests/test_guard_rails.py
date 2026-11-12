"""Guard rails: the warnings that stop a wrong structure or a meaningless analysis from
passing silently.

Each test builds the situation a student would hit (a CIF with a space group but no
symmetry operators, disordered sites, missing hydrogens, a Monte Carlo trajectory fed to
MSD, a copper MOF sent to a closed-shell CP2K template, ...) and checks both that the
warning appears and that the numbers themselves are left alone.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.analyses import check_text, run_isotherm, run_msd, run_rmsd, unavailable
from pdb_wizard.atom import Atom
from pdb_wizard.charges import apply_qeq_charges
from pdb_wizard.checks import (
    CHECK_ERROR,
    CHECK_NOTE,
    CHECK_WARNING,
    check_structure,
    format_findings,
    load_notes,
    lone_atom_summary,
    lone_atom_warning,
    mc_trajectory_reason,
    mc_trajectory_warning,
    open_shell_metals,
    openmm_template_warning,
    supercell_for_cutoff,
)
from pdb_wizard.forcefields import apply_ff_to_system_report, get_forcefield
from pdb_wizard.formats import write_cif
from pdb_wizard.inputgen import check_engine, engine_cautions, generate_engine_inputs
from pdb_wizard.io import next_mpmc_mol_id, read_file
from pdb_wizard.isotherm import isotherm_log_table, parse_isotherm_results, read_mpmc_log
from pdb_wizard.molecule import Molecule
from pdb_wizard.msd import compute_msd
from pdb_wizard.pbc import PBC
from pdb_wizard.sorbates import SORBATE_MODELS, insert_sorbates, sorbate_atoms
from pdb_wizard.templates import engine_defaults

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"


def hkust(prepared: bool = True) -> Molecule:
    mol = read_file(str(HKUST))
    if prepared:
        apply_qeq_charges(mol)
        apply_ff_to_system_report(mol.atoms, get_forcefield(0))
    else:
        for a in mol.atoms:
            a.charge = a.epsilon = a.sigma = a.alpha = 0.0
    return mol


def levels(findings, level):
    return [f for f in findings if f.level == level]


def titles(findings):
    return " | ".join(f.title for f in findings)


# ---- CIF reader notes -------------------------------------------------------------------

CIF_HEAD = """data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
{extra}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{columns}
{rows}
"""


def cif(tmp_path, extra="", columns="", rows="C1 C 0.1 0.1 0.1\nO1 O 0.3 0.3 0.3", name="t.cif"):
    path = tmp_path / name
    path.write_text(CIF_HEAD.format(extra=extra, columns=columns, rows=rows))
    return str(path)


@pytest.mark.parametrize("extra,expect", [
    ("_symmetry_space_group_name_H-M 'F m -3 m'", "F m -3 m"),
    ("_space_group_name_H-M_alt 'P 21/c'", "P 21/c"),
    ("_symmetry_Int_Tables_number 225", "number 225"),
    ("_space_group_IT_number 14", "number 14"),
])
def test_cif_space_group_without_operators_is_flagged(tmp_path, extra, expect):
    mol = read_file(cif(tmp_path, extra))
    notes = load_notes(mol)
    assert len(notes) == 1 and expect in notes[0]
    assert "asymmetric unit" in notes[0] and "P1" in notes[0]
    assert len(mol.atoms) == 2  # the reader still does what it did: no silent change
    assert "From the file reader" in titles(check_structure(mol))


@pytest.mark.parametrize("extra", [
    "", "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_space_group_name_H-M P1",
    "_symmetry_Int_Tables_number 1",
])
def test_p1_cif_has_no_note(tmp_path, extra):
    assert load_notes(read_file(cif(tmp_path, extra))) == []


def test_cif_with_operators_is_expanded_and_not_flagged(tmp_path):
    extra = ("_symmetry_space_group_name_H-M 'P -1'\nloop_\n_symmetry_equiv_pos_as_xyz\n"
             "'x, y, z'\n'-x, -y, -z'")
    mol = read_file(cif(tmp_path, extra))
    assert len(mol.atoms) == 4 and load_notes(mol) == []


def test_cif_disorder_is_flagged(tmp_path):
    mol = read_file(cif(tmp_path, columns="_atom_site_occupancy\n_atom_site_disorder_group",
                        rows="C1 C 0.1 0.1 0.1 1.0 .\nO1A O 0.3 0.3 0.3 0.5(2) 1\n"
                             "O1B O 0.35 0.3 0.3 0.5 2"))
    notes = load_notes(mol)
    assert len(notes) == 1
    assert "2 sites with occupancy below 1" in notes[0] and "2 disorder groups" in notes[0]
    assert len(mol.atoms) == 3


def test_full_occupancy_cif_has_no_disorder_note(tmp_path):
    mol = read_file(cif(tmp_path, columns="_atom_site_occupancy",
                        rows="C1 C 0.1 0.1 0.1 1.0\nO1 O 0.3 0.3 0.3 1"))
    assert load_notes(mol) == []


def test_load_notes_follow_the_structure_through_a_copy(tmp_path):
    mol = read_file(cif(tmp_path, "_symmetry_Int_Tables_number 225"))
    assert load_notes(copy.deepcopy(mol)) == load_notes(mol)
    assert load_notes(None) == [] and load_notes(Molecule(atoms=[])) == []


def test_hkust_written_as_p1_cif_round_trips_without_notes(tmp_path):
    path = tmp_path / "h.cif"
    with open(path, "w") as f:
        write_cif(hkust(), f)
    mol = read_file(str(path))
    assert len(mol.atoms) == 624 and load_notes(mol) == []


# ---- check_structure --------------------------------------------------------------------


def test_prepared_hkust_is_clean():
    findings = check_structure(hkust())
    assert not levels(findings, CHECK_ERROR) and not levels(findings, CHECK_WARNING)
    text = format_findings(findings)
    assert text.startswith("Structure check: nothing suspicious found")
    assert "Cell allows a cutoff up to 13.17 Å" in text
    assert "Open-shell metals: Cu" in text and "Charges present, net" in text


def test_unprepared_hkust_lists_what_is_missing():
    findings = check_structure(hkust(prepared=False))
    warned = titles(levels(findings, CHECK_WARNING))
    assert "No charges" in warned and "No force-field parameters" in warned
    assert format_findings(findings).startswith("Structure check: 2 warnings")


def test_check_does_not_change_the_structure():
    mol = hkust()
    before = [(a.element.symbol, tuple(a.x), a.charge, a.epsilon) for a in mol.atoms]
    bonds = list(mol.bonds)
    check_structure(mol)
    assert before == [(a.element.symbol, tuple(a.x), a.charge, a.epsilon) for a in mol.atoms]
    assert mol.bonds == bonds


def test_empty_and_missing_structures():
    assert check_structure(None)[0].title == "No atoms loaded"
    assert check_structure(Molecule(atoms=[]))[0].level == CHECK_ERROR
    assert "1 problem" in check_text(None)
    assert unavailable("check", hkust(), None) is None


def test_no_cell_is_a_warning():
    mol = Molecule(atoms=[Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C")])
    assert "No unit cell" in titles(levels(check_structure(mol), CHECK_WARNING))


def test_small_cell_suggests_a_supercell():
    mol = Molecule(atoms=[Atom(1, 1, 1, "Cu"), Atom(3, 3, 3, "O")], pbc=PBC(8, 12, 25, 90, 90, 90))
    assert supercell_for_cutoff(mol) == (3, 2, 1)
    finding = next(f for f in check_structure(mol) if "small for a 10 Å cutoff" in f.title)
    assert finding.level == CHECK_WARNING
    assert "4.00 Å" in finding.detail and "3 × 2 × 1 supercell (12 atoms)" in finding.detail


def test_skewed_cell_uses_the_perpendicular_width():
    mol = Molecule(atoms=[Atom(1, 1, 1, "C")], pbc=PBC(21, 21, 21, 90, 90, 120))
    # a = 21 A but the width across the 120-degree face is 21 sin(60) = 18.19 A
    na, nb, nc = supercell_for_cutoff(mol)
    assert (na, nb, nc) == (2, 2, 1)
    assert "9.09 Å" in next(f.detail for f in check_structure(mol) if "small" in f.title)


def test_overlapping_atoms_are_an_error():
    atoms = [Atom(1, 1, 1, "O"), Atom(1.3, 1, 1, "O"), Atom(5, 5, 5, "C")]
    mol = Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))
    err = levels(check_structure(mol), CHECK_ERROR)
    assert len(err) == 1 and "1 pairs of atoms are closer than 0.7 Å" in err[0].title
    assert "O1-O2 at 0.30 Å" in err[0].detail


def test_overlap_across_the_cell_boundary_is_found():
    atoms = [Atom(0.1, 5, 5, "O"), Atom(29.9, 5, 5, "O")]
    mol = Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))
    assert "0.20 Å" in levels(check_structure(mol), CHECK_ERROR)[0].detail


@pytest.mark.parametrize("model", ["h2_bss", "h2_bssp", "co2_phast", "h2o_tip4p", "n2_polar"])
def test_packed_sorbate_sites_are_not_called_overlaps(model):
    """Rigid models put charge sites a fraction of an angstrom apart on purpose."""
    mol = hkust()
    placed = insert_sorbates(mol, SORBATE_MODELS[model], 5, seed=3)
    mol.atoms.extend(sorbate_atoms(SORBATE_MODELS[model], placed, next_mpmc_mol_id(mol)))
    assert not levels(check_structure(mol), CHECK_ERROR)


def test_missing_hydrogens():
    mol = hkust()
    mol.atoms = [a for a in mol.atoms if a.element.symbol != "H"]
    finding = next(f for f in check_structure(mol) if f.title == "No hydrogen atoms")
    assert finding.level == CHECK_WARNING and "288 carbons" in finding.detail
    inorganic = Molecule(atoms=[Atom(0, 0, 0, "Cu"), Atom(2, 0, 0, "O")],
                         pbc=PBC(30, 30, 30, 90, 90, 90))
    assert "No hydrogen atoms" not in titles(check_structure(inorganic))


def test_net_charge_warning():
    mol = hkust()
    mol.atoms[0].charge += 0.5
    assert "Net charge +0.5000 e" in titles(levels(check_structure(mol), CHECK_WARNING))


def test_element_without_parameters_is_named():
    mol = hkust()
    for a in mol.atoms:
        if a.element.symbol == "Cu":
            a.epsilon = 0.0
    assert "No force-field parameters for Cu" in titles(check_structure(mol))


def test_unknown_elements_are_flagged():
    mol = Molecule(atoms=[Atom(0, 0, 0, "Qq"), Atom(3, 0, 0, "C")], pbc=PBC(30, 30, 30, 90, 90, 90))
    assert "1 atoms have an unknown element" in titles(check_structure(mol))


def test_findings_are_ordered_by_severity():
    atoms = [Atom(1, 1, 1, "O"), Atom(1.2, 1, 1, "O")]
    found = check_structure(Molecule(atoms=atoms, pbc=PBC(8, 8, 8, 90, 90, 90)))
    order = [f.level for f in found]
    assert order == sorted(order, key=[CHECK_ERROR, CHECK_WARNING, CHECK_NOTE].index)
    assert format_findings(found).startswith("Structure check: 1 problem, ")


# ---- lone atoms -------------------------------------------------------------------------


def test_lone_atom_preview_names_the_framework_copper():
    mol = hkust()
    assert lone_atom_summary(mol) == "48 Cu"
    assert "metal atoms (Cu)" in lone_atom_warning(mol)
    assert len(mol.atoms) == 624  # a preview, nothing deleted


def test_lone_atom_preview_without_metals_or_cell():
    mol = Molecule(atoms=[Atom(5, 5, 5, "Ar"), Atom(15, 15, 15, "O")],
                   pbc=PBC(30, 30, 30, 90, 90, 90))
    assert lone_atom_summary(mol) in ("1 Ar, 1 O", "1 O, 1 Ar")
    assert lone_atom_warning(mol) == ""
    free = Molecule(atoms=[Atom(0, 0, 0, "Cu")])
    assert lone_atom_summary(free) == "" and lone_atom_warning(free) == ""


# ---- Monte Carlo trajectories ---------------------------------------------------------------


def md_frames(n=14):
    rng = np.random.default_rng(1)
    start = rng.uniform(2, 12, (6, 3))
    frames = []
    for f in range(n):
        atoms = [Atom(*(p + 0.2 * f), "O") for p in start]
        frames.append(Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90)))
    return frames


def uvt_frames(n=14):
    """A framework atom plus a sorbate count that changes, like MPMC μVT output."""
    frames = []
    for f in range(n):
        atoms = [Atom(0, 0, 0, "Cu")] + [Atom(2.0 + k, 3.0, 4.0 + 0.1 * f, "H")
                                         for k in range(3 + f % 3)]
        frames.append(Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90)))
    return frames


def nvt_mpmc_frames(n=14):
    frames = []
    for f in range(n):
        fw = Atom(0, 0, 0, "Cu")
        fw.mpmc_flag = "F"
        h = Atom(3, 3, 3 + 0.1 * f, "H")
        h.mpmc_flag = "M"
        frames.append(Molecule(atoms=[fw, h], pbc=PBC(15, 15, 15, 90, 90, 90)))
    return frames


def test_mc_trajectory_detection():
    assert mc_trajectory_reason(md_frames()) == ""
    assert mc_trajectory_reason(None) == "" and mc_trajectory_reason(md_frames(1)) == ""
    assert "number of atoms changes between frames (4 to 6)" in mc_trajectory_reason(uvt_frames())
    assert "MPMC output" in mc_trajectory_reason(nvt_mpmc_frames())
    assert mc_trajectory_warning(md_frames(), "x") == ""


def test_msd_on_a_monte_carlo_trajectory_warns_but_keeps_its_numbers():
    frames = nvt_mpmc_frames()
    res = run_msd(frames, "H")
    assert "Monte Carlo trajectory" in res.text and "diffusion constant" in res.text
    lags, msd = compute_msd(frames, element="H")
    np.testing.assert_array_equal(res.csv_arrays()[1], msd)
    assert "Monte Carlo" in run_rmsd(frames).text
    assert "Monte Carlo trajectory" in check_text(frames[0], frames)


def test_msd_on_molecular_dynamics_has_no_warning():
    frames = md_frames()
    assert run_msd(frames, "O").text == "" and run_rmsd(frames).text == ""
    assert "Monte Carlo" not in check_text(frames[0], frames)


def test_batch_msd_warns_on_stderr_only(tmp_path):
    path = tmp_path / "mc.xyz"
    with open(path, "w") as f:
        for frame in uvt_frames():
            f.write(f"{len(frame.atoms)}\n\n")
            for a in frame.atoms:
                f.write(f"{a.element.symbol} {a.x[0]} {a.x[1]} {a.x[2]}\n")
    proc = subprocess.run([sys.executable, "-m", "pdb_wizard", str(path), "--msd", "-e", "Cu"],
                          capture_output=True, text=True, cwd=str(ROOT),
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert "Monte Carlo trajectory" in proc.stderr and "Monte Carlo" not in proc.stdout


# ---- isotherm results ---------------------------------------------------------------------

# The end of a real mpmc.log (HKUST-1, H2 BSS, 77 K, 1 atm, 3000 steps; MPMC bab2c06).
REAL_LOG_TAIL = """OUTPUT: Root collecting statistics at Thu Nov 12 19:26:13 2026
OUTPUT: Completed step 2900/3000  (96.667 %)
OUTPUT: N = 38.43333 +- 4.72283 molecules
OUTPUT: wt % = 0.79423 +- 0.03650 %
OUTPUT: wt % (ME) = 0.80059 +- 0.03654 %
OUTPUT: qst = 4.64034 kJ/mol

OUTPUT: Root collecting statistics at Thu Nov 12 19:26:21 2026
OUTPUT: Completed step 3000/3000  (100.000 %)
OUTPUT: potential energy = -16970.02153 +- 2296.02549 K
OUTPUT: N = 38.83871 +- 4.63933 molecules
OUTPUT: density = 0.00711 +- 0.00085 g/cm^3
OUTPUT: wt % = 0.80254 +- 0.03525 %
OUTPUT: wt % (ME) = 0.80904 +- 0.03529 %
OUTPUT: qst = 4.65359 kJ/mol
"""
DONE = "\nMAIN: simulation exiting successfully\n"
ENERGY_HEAD = ("#step #energy #coulombic #rd #polar #vdw #kinetic #kin_temp #N #spin_ratio "
               "#volume #core_temp\n")
OPENDX = ("# OpenDX format population histogram\nobject 1 class gridpositions counts 2 2 2\n"
          "origin\t-5.0\t-5.0\t-5.0\ndelta \t5.0\t0.0\t0.0\n\n0.1 0.2 \n0.3 0.4 \n")


def isotherm_folder(tmp_path, points):
    """points: {folder name: (N values for energy.dat, log text or None)}"""
    for name, (ns, log) in points.items():
        d = tmp_path / name
        d.mkdir()
        rows = "".join(f"{100 * i} -1.0 0 0 0 0 0 0 {n:f} 1.0 1000.0 77.0\n"
                       for i, n in enumerate(ns))
        (d / "energy.dat").write_text(ENERGY_HEAD + rows)
        (d / "histogram.dat").write_text(OPENDX)
        if log is not None:
            (d / "mpmc.log").write_text(log)
    return str(tmp_path)


def test_real_mpmc_log_is_read(tmp_path):
    base = isotherm_folder(tmp_path, {"P_1_atm": ([10, 20, 30, 40], REAL_LOG_TAIL + DONE)})
    log = read_mpmc_log(Path(base) / "P_1_atm")
    assert log["file"] == "mpmc.log" and log["finished"]
    assert log["N"] == (38.83871, 4.63933)  # the last block, not the one before
    assert log["wt"] == (0.80254, 0.03525) and log["wt_me"] == (0.80904, 0.03529)
    assert log["qst"] == (4.65359, None)
    assert (log["step"], log["steps"]) == (3000, 3000)


def test_unfinished_run_is_called_out(tmp_path):
    unfinished = REAL_LOG_TAIL.split("\n\n")[0] + "\n"
    base = isotherm_folder(tmp_path, {
        "P_1_atm": ([10, 20, 30, 40], REAL_LOG_TAIL + DONE),
        "P_10_atm": ([50, 60, 70, 80], unfinished),
    })
    table = isotherm_log_table(base)
    assert "38.839 ± 4.639" in table and "0.8025 ± 0.035" in table and "4.654" in table
    assert "⚠ 10 atm has not finished (step 2900/3000)" in table
    assert "1 atm has not finished" not in table


def test_isotherm_plot_numbers_are_unchanged_and_the_log_is_added(tmp_path):
    base = isotherm_folder(tmp_path, {
        "P_1_atm": ([10, 20, 30, 40], REAL_LOG_TAIL + DONE),
        "P_0p1_atm": ([1, 2, 3, 4], None),
    })
    res = run_isotherm(base)
    np.testing.assert_array_equal(res.x, [0.1, 1.0])
    np.testing.assert_array_equal(res.y, [4.0, 40.0])  # last quarter of energy.dat, as before
    assert "38.839 ± 4.639" in res.text and "last quarter of energy.dat" in res.text
    assert parse_isotherm_results(base) == [(0.1, 4.0), (1.0, 40.0)]


def test_no_logs_means_no_table(tmp_path):
    base = isotherm_folder(tmp_path, {"P_1_atm": ([10, 20, 30, 40], None)})
    assert isotherm_log_table(base) == "" and run_isotherm(base).text == ""
    assert read_mpmc_log(Path(base) / "P_1_atm") == {}
    assert isotherm_log_table(str(tmp_path / "missing")) == ""


def test_slurm_output_is_found_when_there_is_no_mpmc_log(tmp_path):
    base = isotherm_folder(tmp_path, {"P_1_atm": ([10, 20, 30, 40], None)})
    (Path(base) / "P_1_atm" / "slurm-123.out").write_text(REAL_LOG_TAIL + DONE)
    (Path(base) / "P_1_atm" / "notes.log").write_text("nothing from MPMC here\n")
    assert read_mpmc_log(Path(base) / "P_1_atm")["file"] == "slurm-123.out"


def test_opendx_histogram_is_not_read_as_an_uptake_histogram(tmp_path):
    """MPMC's histogram.dat is a density grid; its numbers must not be averaged as N."""
    base = isotherm_folder(tmp_path, {"P_1_atm": ([10, 20, 30, 40], None)})
    assert parse_isotherm_results(base) == [(1.0, 40.0)]
    (Path(base) / "P_1_atm" / "energy.dat").unlink()
    assert parse_isotherm_results(base) == []


def test_a_real_n_histogram_is_still_read(tmp_path):
    d = tmp_path / "P_2_atm"
    d.mkdir()
    (d / "histogram.dat").write_text("# N probability\n10 0.25\n20 0.75\n")
    assert parse_isotherm_results(str(tmp_path)) == [(2.0, 17.5)]


# ---- CP2K and OpenMM cautions ---------------------------------------------------------------


def test_cp2k_warns_about_open_shell_metals_but_still_generates(tmp_path):
    mol = hkust()
    assert open_shell_metals(mol) == ["Cu"]
    check = check_engine(mol, "cp2k_cell_opt", engine_defaults("cp2k_cell_opt"))
    assert check.ok and "1 warning" in check.status
    assert "Cu usually has unpaired electrons" in check.preview and "UKS" in check.preview
    out = generate_engine_inputs(mol, "cp2k_cell_opt", engine_defaults("cp2k_cell_opt"),
                                 str(tmp_path))
    assert out.ok and "UKS" in out.preview and list(tmp_path.glob("*.inp"))


def test_cp2k_has_no_caution_for_a_closed_shell_framework():
    mol = hkust()
    for a in mol.atoms:
        if a.element.symbol == "Cu":
            a._element = Atom(0, 0, 0, "Zn").element
    assert engine_cautions(mol, "cp2k_dft_md") == []
    check = check_engine(mol, "cp2k_dft_md", engine_defaults("cp2k_dft_md"))
    assert check.ok and check.status == "Ready to generate"


def test_openmm_warns_that_a_mof_has_no_templates(tmp_path):
    mol = hkust()
    values = engine_defaults("openmm_npt")
    check = check_engine(mol, "openmm_npt", values)
    assert check.ok and "No template found for residue" in check.preview
    assert "contains Cu" in check.preview
    out = generate_engine_inputs(mol, "openmm_npt", values, str(tmp_path))
    assert out.ok and "No template found" in out.preview


def test_openmm_is_quiet_for_water_and_for_a_custom_force_field():
    water = Molecule(atoms=[Atom(0, 0, 0, "O"), Atom(0.96, 0, 0, "H"), Atom(-0.24, 0.93, 0, "H")],
                     pbc=PBC(20, 20, 20, 90, 90, 90))
    assert openmm_template_warning(water, "amber14-all.xml,amber14/tip3pfb.xml") == ""
    assert openmm_template_warning(hkust(), "my_mof.xml") == ""
    assert engine_cautions(water, "openmm_nvt") == []
    ligand = Molecule(atoms=[Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C")],
                      pbc=PBC(20, 20, 20, 90, 90, 90))
    assert "is not a protein or water" in openmm_template_warning(ligand, "amber14-all.xml")


def test_mpmc_engine_checks_are_unchanged():
    mol = hkust()
    check = check_engine(mol, "mpmc_isotherm", {"sorbate": "h2_bss"})
    assert check.ok and engine_cautions(mol, "mpmc_isotherm") == []
