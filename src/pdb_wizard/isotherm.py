"""Isotherm planner — generates MPMC input directories at multiple pressures.

The actual MPMC input file is rendered from `templates/mpmc_uvt.inp` via the
template engine; this module supplies the per-pressure derived variables
(box size, sorbate placement, file paths) and writes the supporting files
(input.pqr, insert.pqr, run.sh).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pdb_wizard.io import next_mpmc_mol_id, write_mpmc_pdb
from pdb_wizard.molecule import Molecule
from pdb_wizard.sorbates import format_sorbate_pqr
from pdb_wizard.templates import load_template, render

# LJ cutoff (Å) the box is checked against. Minimum image needs half the
# smallest perpendicular cell width to be at least this.
_DEFAULT_LJ_CUTOFF = 10.0
# A charge is "set" if |q| exceeds this — anything smaller is rounding noise.
_CHARGE_NONZERO = 1e-9
# An LJ parameter is "set" if epsilon > this — typical organic atom epsilon
# is 30-150 K, hydrogen is ~10 K, so 0.1 K is a safe lower bound for "applied".
_EPSILON_SET = 0.1
# Sorbate–framework overlap distance (Å) below which we warn.
_SORBATE_OVERLAP = 1.5


# Polarization and PHAHST settings from the MPMC tutorials, which call them "generally
# acceptable for most situations" (tutorials/02_bssp_h2_mof-5, 03_phahst_h2_hkust-1).
_POLARIZATION_KEYWORDS = """polarization          on
polar_damp_type       exponential
polar_damp            2.1304
polar_wolf            on
polar_wolf_alpha      0.13
polar_iterative       on
polar_palmo           on
polar_gs_ranked       on
polar_gamma           1.03
polar_max_iter        4"""
_PHAHST_KEYWORDS = """disp_expansion        on
extrapolate_disp_coeffs on
damp_dispersion       on
cavity_autoreject_repulsion 10000"""


def _model_flags(mol: Molecule, sorbate_model=None) -> dict:
    """What the framework and sorbate parameters need MPMC to switch on."""
    from pdb_wizard.sorbates import SORBATE_MODELS
    model = SORBATE_MODELS.get(sorbate_model) if isinstance(sorbate_model, str) else sorbate_model
    sites = model.sites if model is not None else []
    framework = [a for a in mol.atoms if a.mpmc_flag != "M"]
    return {
        "polar": any(a.alpha > 0 for a in mol.atoms) or any(s.polarizability > 0 for s in sites),
        "phahst_framework": any(max(a.c6, a.c8, a.c10) > 0 for a in framework),
        "phahst_sorbate": any(max(s.c6, s.c8, s.c10) > 0 for s in sites),
        "lj_sorbate": bool(sites) and not any(max(s.c6, s.c8, s.c10) > 0 for s in sites),
        "sorbate_charged": any(abs(s.charge) > _CHARGE_NONZERO for s in sites),
        "sorbate_known": model is not None,
    }


def mpmc_model_keywords(mol: Molecule, sorbate_model=None) -> str:
    """The mpmc.inp lines the PQR parameters need.

    Without them MPMC ignores the polarizabilities and reads PHAHST beta/rho (columns
    13-14) as Lennard-Jones epsilon/sigma. The cutoff is half the smallest perpendicular
    cell width: the same as MPMC's default (half the shortest lattice vector) for an
    orthogonal cell, and inside the minimum-image limit for a skewed one.
    """
    flags = _model_flags(mol, sorbate_model)
    cutoff = float(mol.pbc.perpendicular_widths().min()) / 2.0
    lines = [f"pbc_cutoff            {np.floor(cutoff * 1e4) / 1e4:.4f}"]
    if flags["phahst_framework"] or flags["phahst_sorbate"]:
        lines += ["", _PHAHST_KEYWORDS]
    if flags["polar"]:
        lines += ["", _POLARIZATION_KEYWORDS]
    return "\n".join(lines)


def validate_for_mpmc(
    mol: Molecule,
    sorbate_pos: tuple[float, float, float] | None = None,
    lj_cutoff: float = _DEFAULT_LJ_CUTOFF,
    sorbate_model=None,
) -> tuple[list[str], list[str]]:
    """Inspect a molecule for common MPMC-input footguns.

    Returns (errors, warnings). 'errors' block generation outright; 'warnings'
    let the user proceed but flag likely issues.

    Errors:
      - No PBC (MPMC requires a periodic box)
      - All charges effectively zero (zero electrostatics — meaningless)
      - All epsilon/sigma effectively zero (zero LJ — sorbate floats freely)

    Warnings:
      - |net charge| > 1e-3 e (non-neutral box)
      - Half the smallest perpendicular width < lj_cutoff (cell too thin)
      - MPMC's automatic cutoff exceeds the minimum-image limit (skewed cell)
      - Sorbate placement overlaps a framework atom (within 1.5 Å)

    Triclinic cells need no warning: 'abcbasis a b c α β γ' builds the same
    basis (a along x, b in the xy plane) as PBC._compute_matrices.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if mol.pbc is None:
        errors.append("Molecule has no PBC. MPMC requires a periodic cell — "
                      "use 'Update Unit Cell' to add one.")
        return errors, warnings  # other checks need PBC

    n = len(mol.atoms)
    if n == 0:
        errors.append("Molecule has no atoms.")
        return errors, warnings

    # Charges (an uncharged framework is fine with an uncharged sorbate, e.g. TraPPE CH4)
    flags = _model_flags(mol, sorbate_model)
    charges = np.array([a.charge for a in mol.atoms])
    n_set = int((np.abs(charges) > _CHARGE_NONZERO).sum())
    if n_set == 0 and flags["sorbate_known"] and not flags["sorbate_charged"]:
        pass
    elif n_set == 0:
        errors.append(
            "All atom charges are zero. Apply a force field or "
            "Generate QEq Charges before planning the isotherm — otherwise "
            "MPMC computes no electrostatics and the uptake is meaningless."
        )
    else:
        net_q = float(charges.sum())
        if abs(net_q) > 1e-3:
            warnings.append(
                f"Net system charge = {net_q:+.4f} e (non-neutral). "
                f"MPMC assumes a neutral simulation cell — add counterions "
                f"or rescale charges before running."
            )

    # LJ parameters
    epsilons = np.array([a.epsilon for a in mol.atoms])
    n_lj_set = int((epsilons > _EPSILON_SET).sum())
    if n_lj_set == 0:
        errors.append(
            "All Lennard-Jones epsilon values are zero. Apply a force field "
            "(OPLS-AA or PHAHST) before planning — without LJ, MPMC sees no "
            "framework potential and the sorbate just floats."
        )
    elif n_lj_set < n // 2:
        warnings.append(
            f"Only {n_lj_set}/{n} atoms have nonzero LJ epsilon. "
            f"Some elements are likely missing from the chosen force field."
        )

    if flags["phahst_framework"] != flags["phahst_sorbate"] and flags["sorbate_known"]:
        which = "framework" if flags["phahst_framework"] else "sorbate"
        warnings.append(
            f"Only the {which} has PHAHST (C6/C8/C10) parameters. MPMC's disp_expansion "
            f"applies to every pair, so the other side's LJ epsilon/sigma would be read as "
            f"exponential-repulsion beta/rho. Use PHAHST (or LJ) for both."
        )

    # Box size vs LJ cutoff. Minimum image is only exact up to half the
    # smallest perpendicular width, which for a skewed cell is less than
    # min(a, b, c) / 2.
    pbc = mol.pbc
    max_cutoff = float(pbc.perpendicular_widths().min()) / 2.0
    if max_cutoff < lj_cutoff:
        warnings.append(
            f"Cell is too thin for a {lj_cutoff:.1f} Å cutoff: half the "
            f"smallest perpendicular width is {max_cutoff:.2f} Å. Either build "
            f"a supercell (Edit > Extend Axis) or set pbc_cutoff ≤ "
            f"{max_cutoff:.2f} in mpmc.inp."
        )

    # Without pbc_cutoff, MPMC uses half the shortest lattice vector, which
    # exceeds the minimum-image limit in any non-orthogonal cell. Only pairs
    # in the shell between the two radii are affected, so stay quiet for
    # mild skew (<5%); a hexagonal cell (γ = 120°) is ~15% over.
    mpmc_auto = pbc.shortest_lattice_vector() / 2.0
    if mpmc_auto > 1.05 * max_cutoff:
        warnings.append(
            f"Cell is skewed: MPMC's automatic cutoff ({mpmc_auto:.2f} Å, half "
            f"the shortest lattice vector) is larger than the minimum-image "
            f"limit ({max_cutoff:.2f} Å), so some pairs would be missed. Set "
            f"pbc_cutoff ≤ {max_cutoff:.2f} in mpmc.inp."
        )

    # Sorbate placement overlap
    if sorbate_pos is None:
        cx, cy, cz = mol.center()
    else:
        cx, cy, cz = sorbate_pos
    coords = np.array([a.x for a in mol.atoms])
    dx = coords - np.array([float(cx), float(cy), float(cz)])
    # Min-image distance to the sorbate placement
    frac = dx @ pbc.reciprocal_basis_matrix
    frac -= np.round(frac)
    dx = frac @ pbc.basis_matrix
    dists = np.linalg.norm(dx, axis=1)
    n_overlap = int((dists < _SORBATE_OVERLAP).sum())
    if n_overlap > 0:
        warnings.append(
            f"Sorbate placement at ({cx:.2f}, {cy:.2f}, {cz:.2f}) overlaps "
            f"{n_overlap} framework atom{'s' if n_overlap != 1 else ''} "
            f"(within {_SORBATE_OVERLAP} Å). Use 'sorbate_pos=' to pick a "
            f"pore center, or let MPMC's μVT insertion handle placement."
        )

    return errors, warnings


def _pressure_dir_name(pressure: float) -> str:
    p_str = f"{pressure:.4g}".replace(".", "p").replace("-", "m")
    return f"P_{p_str}_atm"


def generate_isotherm(
    mol: Molecule,
    sorbate_model: str,
    temperature: float,
    pressures: list[float],
    output_dir: str,
    numsteps: int = 500000,
    corrtime: int = 1000,
    ensemble: str = "uvt",
    sorbate_pos: tuple[float, float, float] | None = None,
    insert_probability: float = 0.667,
    move_factor: float = 0.01,
    rot_factor: float = 0.01,
) -> list[str]:
    """Generate MPMC input directories for an isotherm.

    Creates one subdirectory per pressure point with:
      - input.pqr   : framework + sorbate, charges + LJ params
      - mpmc.inp    : rendered from templates/mpmc_uvt.inp
      - insert.pqr  : sorbate-only insert pool
      - run.sh      : launch script

    Args:
        mol: MOF structure with PBC.
        sorbate_model: Sorbate model name (from sorbates.py).
        temperature: Temperature in Kelvin.
        pressures: List of pressures in atm.
        output_dir: Base directory for all pressure subdirs.
        numsteps: MC steps per pressure point.
        corrtime: Correlation time for sampling.
        ensemble: "uvt" for grand canonical, "nvt" for canonical.
        sorbate_pos: Optional (x, y, z) for sorbate placement. Auto if None.
        insert_probability, move_factor, rot_factor: MPMC move probabilities.

    Returns:
        List of created directory paths.
    """
    if mol.pbc is None:
        raise ValueError("Molecule must have PBC for MPMC simulation")

    pbc = mol.pbc
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    if sorbate_pos is None:
        center = mol.center()
        sx, sy, sz = float(center[0]), float(center[1]), float(center[2])
    else:
        sx, sy, sz = sorbate_pos

    sorbate_lines = format_sorbate_pqr(
        sorbate_model, sx, sy, sz, mol_id=next_mpmc_mol_id(mol),
        start_atom_id=len(mol.atoms) + 1,
    )

    template = load_template("mpmc_uvt.inp")
    created_dirs: list[str] = []

    for pressure in pressures:
        p_name = _pressure_dir_name(pressure)
        p_dir = base / p_name
        p_dir.mkdir(exist_ok=True)

        pqr_path = str(p_dir / "input.pqr")
        write_mpmc_pdb(
            mol, pqr_path,
            write_charges=True, write_params=True, sorbate_lines=sorbate_lines,
        )

        rendered = render(template, {
            "job_name": f"isotherm_{p_name}",
            "ensemble": ensemble,
            "cell_a": f"{pbc.a:.4f}",
            "cell_b": f"{pbc.b:.4f}",
            "cell_c": f"{pbc.c:.4f}",
            "cell_alpha": f"{pbc.alpha:.2f}",
            "cell_beta": f"{pbc.beta:.2f}",
            "cell_gamma": f"{pbc.gamma:.2f}",
            "temperature": f"{temperature:.1f}",
            "pressure": pressure,
            "numsteps": numsteps,
            "corrtime": corrtime,
            "insert_probability": insert_probability,
            "move_factor": move_factor,
            "rot_factor": rot_factor,
            "pqr_input": "input.pqr",
            "model_keywords": mpmc_model_keywords(mol, sorbate_model),
        })
        (p_dir / "mpmc.inp").write_text(rendered)

        # insert.pqr — just the sorbate atoms + END
        (p_dir / "insert.pqr").write_text("\n".join(sorbate_lines) + "\nEND\n")

        # run.sh
        run_path = p_dir / "run.sh"
        run_path.write_text("#!/bin/bash\nmpmc mpmc.inp\n")
        run_path.chmod(0o755)

        created_dirs.append(str(p_dir))

    # Master run-all script
    run_all = base / "run_all.sh"
    lines = ["#!/bin/bash", f"# Isotherm: {sorbate_model} at {temperature} K", ""]
    for p_dir_str in created_dirs:
        p_name = Path(p_dir_str).name
        lines.append(f"echo 'Running {p_name}...'")
        lines.append(f"cd {p_name} && mpmc mpmc.inp > mpmc.log 2>&1 && cd ..")
    lines.append("")
    lines.append("echo 'All pressure points complete.'")
    run_all.write_text("\n".join(lines) + "\n")
    run_all.chmod(0o755)

    return created_dirs


def parse_isotherm_results(base_dir: str) -> list[tuple[float, float]]:
    """Parse MPMC output to extract isotherm data (pressure, uptake).

    Reads histogram.dat or energy.dat from each pressure subdirectory.

    Returns:
        List of (pressure_atm, avg_molecules) tuples, sorted by pressure.
    """
    base = Path(base_dir)
    results = []

    for p_dir in sorted(base.iterdir()):
        if not p_dir.is_dir() or not p_dir.name.startswith("P_"):
            continue
        p_str = p_dir.name.replace("P_", "").replace("_atm", "").replace("p", ".").replace("m", "-")
        try:
            pressure = float(p_str)
        except ValueError:
            continue

        hist_file = p_dir / "histogram.dat"
        if hist_file.exists() and not _is_opendx(hist_file):
            try:
                lines = hist_file.read_text().strip().split("\n")
                total_n = 0.0
                total_p = 0.0
                for line in lines:
                    if line.startswith("#"):
                        continue
                    tokens = line.split()
                    if len(tokens) >= 2:
                        n = float(tokens[0])
                        prob = float(tokens[1])
                        total_n += n * prob
                        total_p += prob
                if total_p > 0:
                    avg_n = total_n / total_p
                    results.append((pressure, avg_n))
                    continue
            except Exception:
                pass

        energy_file = p_dir / "energy.dat"
        if energy_file.exists():
            try:
                from pdb_wizard.energy import read_energy_dat
                data = read_energy_dat(str(energy_file))
                if "N" in data and len(data["N"]) > 0:
                    n_vals = data["N"]
                    start = len(n_vals) * 3 // 4
                    avg_n = float(n_vals[start:].mean())
                    results.append((pressure, avg_n))
            except Exception:
                pass

    results.sort(key=lambda x: x[0])
    return results


def _is_opendx(path: Path) -> bool:
    """MPMC's histogram.dat is an OpenDX density grid, not an N histogram."""
    try:
        with open(path, errors="replace") as f:
            head = f.read(400)
    except OSError:
        return False
    return "OpenDX" in head or "gridpositions" in head


_LOG_NAMES = ("mpmc.log", "runlog", "run.log", "output.log")
_LOG_VALUE = re.compile(r"OUTPUT:\s*(N|wt %|wt % \(ME\)|qst)\s*=\s*([-+0-9.eE]+)"
                        r"(?:\s*\+-\s*([-+0-9.eE]+))?")
_LOG_STEP = re.compile(r"Completed step\s+(\d+)/(\d+)")


def read_mpmc_log(p_dir: Path) -> dict:
    """MPMC's own final averages from the run log in a pressure folder.

    Looks for mpmc.log (what run_all.sh writes), a few common names, then any *.log or
    slurm-*.out. Returns {} when there is no log, else a dict with ``file``, ``finished``,
    ``step``/``steps`` and, when printed, ``N``, ``wt``, ``wt_me`` (value, error) and
    ``qst``.
    """
    candidates = [p_dir / name for name in _LOG_NAMES]
    candidates += sorted(p_dir.glob("*.log")) + sorted(p_dir.glob("slurm-*.out"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "OUTPUT:" not in text:
            continue
        out: dict = {"file": path.name, "finished": "simulation exiting successfully" in text}
        keys = {"N": "N", "wt %": "wt", "wt % (ME)": "wt_me", "qst": "qst"}
        for name, value, err in _LOG_VALUE.findall(text):  # the last block wins
            try:
                out[keys[name]] = (float(value), float(err) if err else None)
            except ValueError:
                continue
        steps = _LOG_STEP.findall(text)
        if steps:
            out["step"], out["steps"] = int(steps[-1][0]), int(steps[-1][1])
        return out
    return {}


def isotherm_log_table(base_dir: str) -> str:
    """MPMC's own uptake, error bars, wt% and Qst for each pressure folder, as text, with a
    line for every run that has not finished. '' when no folder has a log."""
    base = Path(base_dir)
    if not base.is_dir():
        return ""
    rows = []
    for p_dir in sorted(base.iterdir()):
        if not p_dir.is_dir() or not p_dir.name.startswith("P_"):
            continue
        p_str = p_dir.name.replace("P_", "").replace("_atm", "").replace("p", ".").replace("m", "-")
        try:
            pressure = float(p_str)
        except ValueError:
            continue
        log = read_mpmc_log(p_dir)
        if log:
            rows.append((pressure, log))
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])

    def pm(log: dict, key: str, fmt: str) -> str:
        if key not in log:
            return "-"
        value, err = log[key]
        return f"{value:{fmt}}" + (f" ± {err:{fmt}}" if err is not None else "")

    lines = ["MPMC's own averages (whole run, from the log files):", "",
             f"{'P (atm)':>10}  {'N (molecules)':>20}  {'wt %':>18}  {'Qst (kJ/mol)':>12}"]
    unfinished = []
    for pressure, log in rows:
        lines.append(f"{pressure:>10g}  {pm(log, 'N', '.3f'):>20}  {pm(log, 'wt', '.4f'):>18}  "
                     f"{pm(log, 'qst', '.3f'):>12}")
        if not log["finished"]:
            done = f" (step {log['step']}/{log['steps']})" if "step" in log else ""
            unfinished.append(f"⚠ {pressure:g} atm has not finished{done}; its numbers will "
                              f"still change.")
    lines += [""] + unfinished
    lines.append("The plot shows the average of N over the last quarter of energy.dat; MPMC's "
                 "average covers the whole run, including equilibration, so the two differ "
                 "until the run is long.")
    return "\n".join(lines)
