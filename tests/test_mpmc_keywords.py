"""mpmc.inp gets the keywords its PQR parameters need.

Before, the template only had LJ + Coulomb settings: MPMC ignored the polarizabilities and
read PHAHST beta/rho (columns 13-14) as Lennard-Jones epsilon/sigma. Uncharged sorbates
(TraPPE CH4) were also refused on an uncharged framework.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pdb_wizard.charges import apply_qeq_charges
from pdb_wizard.forcefields import apply_ff_to_system_report, get_forcefield
from pdb_wizard.inputgen import check_engine
from pdb_wizard.io import read_file
from pdb_wizard.isotherm import generate_isotherm, mpmc_model_keywords, validate_for_mpmc

ROOT = Path(__file__).resolve().parent.parent
OPLS_UFF, PHAHST = 0, 1


def hkust(ff=None, qeq=False, zero_charges=False):
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    if ff is not None:
        apply_ff_to_system_report(mol.atoms, get_forcefield(ff))
    if qeq:
        apply_qeq_charges(mol)
    if zero_charges:
        for a in mol.atoms:
            a.charge = 0.0
    return mol


def keys(text):
    return [ln.split()[0] for ln in text.splitlines() if ln.split()]


def test_lj_without_polarizabilities_only_sets_the_cutoff():
    mol = hkust()
    for a in mol.atoms:
        a.alpha, a.epsilon, a.sigma = 0.0, 50.0, 3.4
    assert keys(mpmc_model_keywords(mol, "h2_bss")) == ["pbc_cutoff"]


def test_polarizabilities_switch_polarization_on():
    text = mpmc_model_keywords(hkust(OPLS_UFF), "h2_bssp")
    assert "polarization          on" in text and "polar_damp            2.1304" in text
    assert "disp_expansion" not in text


def test_phahst_switches_disp_expansion_on():
    text = mpmc_model_keywords(hkust(PHAHST), "h2_phahst")
    expected = ("disp_expansion        on", "damp_dispersion       on",
                "polarization          on")
    for line in expected:
        assert line in text


def test_cutoff_is_inside_the_minimum_image_limit_for_skewed_cells():
    mol = read_file(str(ROOT / "Ni-MOF-74.pqr"))  # hexagonal, gamma = 120
    cutoff = float(mpmc_model_keywords(mol).split()[1])
    assert cutoff <= mol.pbc.perpendicular_widths().min() / 2
    assert cutoff < mol.pbc.shortest_lattice_vector() / 2  # MPMC's default would be too big
    cubic = hkust()
    assert float(mpmc_model_keywords(cubic).split()[1]) == pytest.approx(cubic.pbc.a / 2, abs=1e-4)


def test_uncharged_sorbate_on_an_uncharged_framework_is_allowed():
    mol = hkust(OPLS_UFF, zero_charges=True)
    errors, _ = validate_for_mpmc(mol, sorbate_model="ch4_trappe")
    assert errors == []
    errors, _ = validate_for_mpmc(mol, sorbate_model="h2_bss")  # charged sorbate: still an error
    assert any("charges are zero" in e for e in errors)
    errors, _ = validate_for_mpmc(mol)  # sorbate not known yet: as before
    assert any("charges are zero" in e for e in errors)
    assert check_engine(mol, "mpmc_isotherm", {"sorbate": "ch4_trappe"}).ok


def test_mixing_phahst_and_lj_is_flagged():
    _, warnings = validate_for_mpmc(hkust(PHAHST, qeq=True), sorbate_model="h2_bss")
    assert any("Only the framework has PHAHST" in w for w in warnings)
    _, warnings = validate_for_mpmc(hkust(PHAHST, qeq=True), sorbate_model="h2_phahst")
    assert not any("PHAHST" in w for w in warnings)


def test_generated_input_contains_the_keywords(tmp_path):
    d = generate_isotherm(hkust(PHAHST, qeq=True), "h2_phahst", 77.0, [1.0], str(tmp_path))[0]
    text = (Path(d) / "mpmc.inp").read_text()
    assert "disp_expansion        on" in text and "pbc_cutoff" in text


@pytest.fixture
def mpmc_bin(request):
    import shutil
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("pass --mpmc /path/to/mpmc to run MPMC itself")
    return path


@pytest.mark.parametrize("ff, sorbate", [(OPLS_UFF, "h2_bssp"), (PHAHST, "h2_phahst")])
def test_mpmc_runs_the_generated_inputs(tmp_path, mpmc_bin, ff, sorbate):
    d = generate_isotherm(hkust(ff, qeq=True), sorbate, 298.0, [1.0], str(tmp_path),
                          numsteps=100, corrtime=50)[0]
    run = subprocess.run([mpmc_bin, "mpmc.inp"], cwd=d, capture_output=True, text=True,
                         timeout=900)
    log = run.stdout + run.stderr
    assert run.returncode == 0 and "invalid command" not in log.lower(), log[-2000:]
    assert "OUTPUT: N =" in log
