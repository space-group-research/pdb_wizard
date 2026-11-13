"""End-to-end tests: run the real pdb_wizard.py as a program and check the files it writes.

The classic menu is driven through stdin, batch mode through its flags, and with
``--mpmc /path/to/mpmc`` the written files are also run through MPMC itself.

Run from the repo root:  python -m pytest tests/
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "dist" / "pdb_wizard.py"  # built by conftest.py

# Monoclinic cell, framework on a 4x3x4 grid: C with charges and UFF-like LJ.
A, B, C, BETA = 20.0, 15.0, 18.0, 99.0


def basis():
    b = np.radians(BETA)
    return np.array([[A, 0, 0], [0, B, 0], [C * np.cos(b), 0, C * np.sin(b)]])


def framework_pqr():
    lines = [f"CRYST1 {A:8.3f} {B:8.3f} {C:8.3f}  90.00 {BETA:6.2f}  90.00 P 1           1\n"]
    i = 1
    for fa in (0.0, 0.25, 0.5, 0.75):
        for fb in (0.0, 1 / 3, 2 / 3):
            for fc in (0.0, 0.25, 0.5, 0.75):
                x, y, z = np.array([fa, fb, fc]) @ basis()
                q = 0.1 if i % 2 else -0.1
                lines.append(
                    f"ATOM {i} C MOF F 1 {x:.4f} {y:.4f} {z:.4f} 12.011 {q} 1.288 52.84 3.431"
                    " 0.0 0.0 0.0 0.0 0.0\n"
                )
                i += 1
    lines += [
        f"REMARK BOX BASIS[{k}] = {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n" for k, r in enumerate(basis())
    ]
    lines.append("END\n")
    return "".join(lines)


@pytest.fixture
def mof(tmp_path):
    path = tmp_path / "mof.pqr"
    path.write_text(framework_pqr())
    return path


def run(args, stdin="", cwd=None):
    """Run pdb_wizard.py; never in TUI mode (that would prompt to pip install textual)."""
    proc = subprocess.run(
        [sys.executable, str(SRC), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=cwd,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def records(path):
    return [
        ln.split()
        for ln in Path(path).read_text().splitlines()
        if ln.startswith("ATOM") and " BOX " not in ln
    ]


def movable_ids(path):
    return [r[5] for r in records(path) if r[4] == "M"]


def n_molecules(path):
    return len(set(movable_ids(path)))


# ---- classic menu: insert -----------------------------------------------------------


def test_classic_insert_by_name(mof, tmp_path):
    out = tmp_path / "packed.pqr"
    proc = run(["--classic", str(mof)], f"10\nh2o_dc97\n6\n{out}\n9\n")
    assert "Placed 6 H2O" in proc.stdout and f"wrote {out}" in proc.stdout
    assert n_molecules(out) == 6
    assert movable_ids(out) == [str(k) for k in range(2, 8) for _ in range(4)]
    text = out.read_text()
    assert text.count("REMARK BOX BASIS[") == 3 and all(
        ln.split()[3] == "=" for ln in text.splitlines() if ln.startswith("REMARK BOX")
    )
    assert sum(r[4] == "F" for r in records(out)) == 48


def test_classic_insert_by_number_with_fill(mof, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pw_names", SRC)
    pw = importlib.util.module_from_spec(spec)
    sys.modules["pw_names"] = pw
    spec.loader.exec_module(pw)
    idx = list(pw.SORBATE_MODELS).index("h2o_spce")
    out = tmp_path / "full.pqr"
    proc = run(["--classic", str(mof)], f"10\n{idx}\nfill\n{out}\n9\n")
    n = n_molecules(out)
    assert n > 6 and f"Placed {n} H2O" in proc.stdout


def test_classic_insert_from_a_template_file(mof, tmp_path):
    tpl = tmp_path / "co2.pqr"
    tpl.write_text(
        "ATOM 1 C CO2 M 2 0.0 0.0 0.0 12.011 0.70 0.0 27.0 2.80\n"
        "ATOM 2 O CO2 M 2 1.16 0.0 0.0 15.999 -0.35 0.0 79.0 3.05\n"
        "ATOM 3 O CO2 M 2 -1.16 0.0 0.0 15.999 -0.35 0.0 79.0 3.05\n"
    )
    out = tmp_path / "co2_packed.pqr"
    run(["--classic", str(mof)], f"10\n{tpl}\n5\n{out}\n9\n")
    mv = [r for r in records(out) if r[4] == "M"]
    assert len(mv) == 15 and {r[3] for r in mv} == {"CO2"}
    for k in range(5):  # rigid geometry survives: C-O 1.16 A
        c, o = (np.array([float(v) for v in mv[3 * k + j][6:9]]) for j in (0, 1))
        assert np.linalg.norm(c - o) == pytest.approx(1.16, abs=1e-4)


def test_classic_bad_answers_then_cancel(mof, tmp_path):
    proc = run(["--classic", str(mof)], "10\nnope_model\nq\n9\n", cwd=tmp_path)
    assert "nope_model" in proc.stdout  # reported, asked again, then cancelled
    assert [f.name for f in tmp_path.glob("*.pqr")] == ["mof.pqr"]  # nothing new written


# ---- re-saving files that contain waters --------------------------------------------


@pytest.fixture
def packed(mof, tmp_path):
    out = tmp_path / "packed.pqr"
    run(["--classic", str(mof)], f"10\nh2o_dc97\n6\n{out}\n9\n")
    return out


def coords(path, flag):
    return np.array([[float(v) for v in r[6:9]] for r in records(path) if r[4] == flag])


def test_batch_convert_keeps_waters_movable(packed, tmp_path):
    out = tmp_path / "resaved.pqr"
    proc = run([str(packed), "--convert", "-o", str(out)])
    assert "Unrecognised element" not in proc.stderr
    assert movable_ids(out) == movable_ids(packed)
    assert np.allclose(coords(out, "M"), coords(packed, "M"), atol=1e-5)
    m_sites = [r for r in records(out) if r[4] == "M" and r[2] == "M"]
    assert len(m_sites) == 6 and all(float(r[9]) == 0.0 for r in m_sites)  # massless, as in DC97


def test_classic_write_mpmc_keeps_waters_movable(packed, tmp_path):
    out = tmp_path / "resaved_classic.pqr"
    run(["--classic", str(packed)], f"7\nn\nn\n{out}\n9\n")
    assert n_molecules(out) == 6
    assert np.allclose(coords(out, "M"), coords(packed, "M"), atol=1e-5)


def test_classic_insert_more_into_a_packed_file(packed, tmp_path):
    out = tmp_path / "more.pqr"
    run(["--classic", str(packed)], f"10\nh2o_dc97\n3\n{out}\n9\n")
    assert movable_ids(out) == [str(k) for k in range(2, 11) for _ in range(4)]


def test_classic_extend_then_insert(packed, tmp_path):
    out = tmp_path / "super.pqr"
    run(["--classic", str(packed)], f"2\n1\n1\n10\nh2o_dc97\n2\n{out}\n9\n")
    assert sum(r[4] == "F" for r in records(out)) == 96
    ids = movable_ids(out)
    assert len(set(ids)) == 6 * 2 + 2 and len(ids) == 4 * 14


def test_batch_info_on_a_packed_file(packed):
    proc = run([str(packed), "--info"])
    assert "Unrecognised element" not in proc.stderr and proc.stdout.strip()


# ---- optional: real MPMC (pytest --mpmc /path/to/mpmc) --------------------------------


@pytest.fixture
def mpmc_bin(request):
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("pass --mpmc /path/to/mpmc to run MPMC itself")
    return path


def mpmc(mpmc_bin, workdir, pqr, extra=""):
    workdir.mkdir()
    shutil.copy(pqr, workdir / "in.pqr")
    (workdir / "run.inp").write_text(
        "job_name e2e\ntemperature 298\npqr_input in.pqr\nread_pqr_box on\n" + extra
    )
    proc = subprocess.run(
        [mpmc_bin, "run.inp"], cwd=workdir, capture_output=True, text=True, timeout=300
    )
    assert "ERROR" not in proc.stdout + proc.stderr, proc.stdout[-2000:]
    return proc.stdout


def energy(out):
    return float(out.split("potential energy =")[1].split()[0])


def test_mpmc_single_point_same_before_and_after_resave(packed, tmp_path, mpmc_bin):
    resaved = tmp_path / "resaved.pqr"
    run([str(packed), "--convert", "-o", str(resaved)])
    e1 = energy(mpmc(mpmc_bin, tmp_path / "a", packed, "ensemble total_energy\n"))
    e2 = energy(mpmc(mpmc_bin, tmp_path / "b", resaved, "ensemble total_energy\n"))
    # 0.85*sigma lets a few contacts be mildly repulsive (~1e3 K); a real overlap is 1e10+ K
    assert -1e6 < e1 < 1e4 and e2 == pytest.approx(e1, abs=1e-6)


def test_mpmc_runs_gcmc_on_the_packed_file(packed, tmp_path, mpmc_bin):
    out = mpmc(
        mpmc_bin,
        tmp_path / "uvt",
        packed,
        "ensemble uvt\npressure 1.0\nnumsteps 200\ncorrtime 100\ninsert_probability 0.5\n"
        "cavity_autoreject_absolute on\ncavity_autoreject_scale 0.76\n",
    )
    assert "basis[2] successfully read" in out
    assert "Completed step 200/200" in out and "OUTPUT: N =" in out


def test_classic_force_field_on_a_packed_file_keeps_the_water_model(packed, tmp_path):
    out = tmp_path / "ff.pqr"
    proc = run(["--classic", str(packed)], f"7\nn\ny\nOPLSAA\n{out}\n9\n")
    assert "element X" not in proc.stdout  # no "not found in forcefield" for M sites
    before = [r[9:14] for r in records(packed) if r[4] == "M"]
    after = [r[9:14] for r in records(out) if r[4] == "M"]
    assert after == before  # mass, charge, alpha, epsilon, sigma of every water site
