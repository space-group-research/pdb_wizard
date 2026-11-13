"""Template-based input file generator.

Templates live in src/pdb_wizard/templates/ as plain text files with
{{{varname}}} placeholders. Substitution is intentionally minimal — no
conditionals, no loops, no inheritance. For computed/multi-line content
(like CP2K's per-element &KIND blocks) the generator builds the block in
Python and passes it in as a single string variable.

Users may edit the shipped templates in place to tweak defaults; the
Input Generator UI re-reads them every time the modal opens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdb_wizard.sorbates import SORBATE_MODELS

TEMPLATES_DIR = Path(__file__).parent

_PLACEHOLDER = re.compile(r"\{\{\{(\w+)\}\}\}")

# Populated by build.py for the amalgamated single-file dist (which has no
# templates/ folder next to it). Source installs always read from disk so
# users can edit the shipped templates in place.
_BUNDLED_TEMPLATES: dict[str, str] = {}


def load_template(name: str) -> str:
    """Load a template by filename (relative to templates/).

    Prefers an on-disk file under TEMPLATES_DIR so editable installs can
    customise the shipped templates. Falls back to _BUNDLED_TEMPLATES when
    running from the amalgamated single-file dist.
    """
    p = TEMPLATES_DIR / name
    if p.is_file():
        return p.read_text()
    if name in _BUNDLED_TEMPLATES:
        return _BUNDLED_TEMPLATES[name]
    raise FileNotFoundError(f"template not found: {p}")


def find_variables(template_str: str) -> list[str]:
    """Return the unique {{{var}}} names in `template_str`, in first-seen order."""
    seen: dict[str, None] = {}
    for m in _PLACEHOLDER.finditer(template_str):
        seen.setdefault(m.group(1), None)
    return list(seen.keys())


def render(template_str: str, vars: dict, *, strict: bool = True) -> str:
    """Substitute {{{var}}} placeholders with `str(vars[var])`.

    Raises KeyError when strict=True and a placeholder has no value.
    Extra keys in `vars` are silently ignored.
    """
    missing: list[str] = []

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in vars:
            missing.append(name)
            return m.group(0)
        return str(vars[name])

    out = _PLACEHOLDER.sub(_sub, template_str)
    if missing and strict:
        raise KeyError(
            "unfilled template variables: " + ", ".join(sorted(set(missing)))
        )
    return out


# --- Engine registry (UI metadata) ---


@dataclass
class VarSpec:
    """One user-editable variable in an engine form.

    kind:
        'float'       — SpinBox over reals
        'int'         — SpinBox over ints
        'str'         — single-line Input
        'select'      — Select with `choices`
        'csv_floats'  — Input parsed as comma-sep list of floats (multi_run)
    """
    name: str
    label: str
    kind: str
    default: Any
    choices: tuple[str, ...] | None = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    help: str = ""


@dataclass
class EngineSpec:
    """One engine: template + UI form + behavior."""
    key: str
    label: str
    description: str
    template_file: str
    output_filename: str
    user_vars: list[VarSpec] = field(default_factory=list)
    # multi_run=True iff the user picks a 'csv_floats' variable that
    # produces one output directory per value (currently only MPMC isotherm
    # over pressures).
    multi_run: bool = False
    multi_run_var: str = ""   # name of the csv_floats var that drives the loop


ENGINES: dict[str, EngineSpec] = {
    "mpmc_isotherm": EngineSpec(
        key="mpmc_isotherm",
        label="MPMC μVT isotherm",
        description="Grand-canonical Monte Carlo isotherm over multiple pressures.",
        template_file="mpmc_uvt.inp",
        output_filename="mpmc.inp",
        multi_run=True,
        multi_run_var="pressures",
        user_vars=[
            VarSpec("sorbate", "Sorbate model", "select", "h2_bss",
                    choices=tuple(SORBATE_MODELS)),  # every built-in model
            VarSpec("temperature", "Temperature (K)", "float", 77.0,
                    min_val=1.0, max_val=2000.0),
            VarSpec("pressures", "Pressures (atm, comma-sep)", "csv_floats",
                    "0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0"),
            VarSpec("numsteps", "MC steps per point", "int", 500_000,
                    min_val=1000, max_val=100_000_000, step=100_000),
            VarSpec("corrtime", "Correlation time", "int", 1000,
                    min_val=10, max_val=100_000),
            VarSpec("ensemble", "Ensemble", "select", "uvt",
                    choices=("uvt", "nvt")),
            VarSpec("insert_probability", "Insert probability", "float", 0.667,
                    min_val=0.0, max_val=1.0),
            VarSpec("move_factor", "Move factor", "float", 0.01,
                    min_val=0.0, max_val=1.0),
            VarSpec("rot_factor", "Rotation factor", "float", 0.01,
                    min_val=0.0, max_val=1.0),
        ],
    ),
    "cp2k_cell_opt": EngineSpec(
        key="cp2k_cell_opt",
        label="CP2K cell optimization",
        description="Relax atomic positions and the unit cell with CP2K LBFGS.",
        template_file="cp2k_cell_opt.inp",
        output_filename="input.inp",
        user_vars=[
            VarSpec("project", "Project name", "str", "pdbwizard"),
            VarSpec("cutoff_ry", "Plane-wave cutoff (Ry)", "float", 400.0,
                    min_val=100.0, max_val=2000.0, step=10.0),
            VarSpec("rel_cutoff_ry", "Relative cutoff (Ry)", "float", 60.0,
                    min_val=20.0, max_val=200.0, step=5.0),
            VarSpec("xc_functional", "XC functional", "select", "PBE",
                    choices=("PBE", "BLYP", "PBE0", "BEEFVDW")),
            VarSpec("eps_scf", "SCF tolerance", "float", 1.0e-7,
                    min_val=1.0e-12, max_val=1.0e-3),
            VarSpec("max_scf", "Max SCF iters", "int", 50,
                    min_val=5, max_val=500),
            VarSpec("n_opt_steps", "Cell-opt steps", "int", 200,
                    min_val=1, max_val=5000),
        ],
    ),
    "cp2k_dft_md": EngineSpec(
        key="cp2k_dft_md",
        label="CP2K DFT-MD (NVT)",
        description="Born-Oppenheimer MD with Nose-Hoover thermostat.",
        template_file="cp2k_dft_md.inp",
        output_filename="input.inp",
        user_vars=[
            VarSpec("project", "Project name", "str", "pdbwizard"),
            VarSpec("temperature", "Temperature (K)", "float", 300.0,
                    min_val=1.0, max_val=5000.0),
            VarSpec("timestep_fs", "Timestep (fs)", "float", 0.5,
                    min_val=0.1, max_val=10.0, step=0.1),
            VarSpec("n_md_steps", "MD steps", "int", 10_000,
                    min_val=1, max_val=10_000_000, step=1000),
            VarSpec("md_traj_interval", "Trajectory print every N steps",
                    "int", 10, min_val=1, max_val=10_000),
            VarSpec("cutoff_ry", "Plane-wave cutoff (Ry)", "float", 400.0,
                    min_val=100.0, max_val=2000.0, step=10.0),
            VarSpec("rel_cutoff_ry", "Relative cutoff (Ry)", "float", 60.0,
                    min_val=20.0, max_val=200.0, step=5.0),
            VarSpec("xc_functional", "XC functional", "select", "PBE",
                    choices=("PBE", "BLYP", "PBE0", "BEEFVDW")),
            VarSpec("eps_scf", "SCF tolerance", "float", 1.0e-7,
                    min_val=1.0e-12, max_val=1.0e-3),
            VarSpec("max_scf", "Max SCF iters", "int", 50,
                    min_val=5, max_val=500),
        ],
    ),
    "openmm_npt": EngineSpec(
        key="openmm_npt",
        label="OpenMM NPT MD",
        description="Langevin NPT with Monte-Carlo barostat, amber14/tip3pfb default.",
        template_file="openmm_npt.py",
        output_filename="run.py",
        user_vars=[
            VarSpec("forcefield", "Force field XMLs (comma-sep)", "str",
                    "amber14-all.xml,amber14/tip3pfb.xml"),
            VarSpec("temperature_k", "Temperature (K)", "float", 300.0,
                    min_val=1.0, max_val=2000.0),
            VarSpec("pressure_atm", "Pressure (atm)", "float", 1.0,
                    min_val=0.001, max_val=10000.0),
            VarSpec("timestep_fs", "Timestep (fs)", "float", 2.0,
                    min_val=0.1, max_val=10.0, step=0.5),
            VarSpec("n_steps", "MD steps", "int", 100_000,
                    min_val=10, max_val=100_000_000, step=10_000),
            VarSpec("report_interval", "Report every N steps", "int", 1000,
                    min_val=10, max_val=1_000_000, step=100),
            VarSpec("nonbonded_cutoff_nm", "Nonbonded cutoff (nm)", "float",
                    1.0, min_val=0.5, max_val=5.0, step=0.1),
        ],
    ),
    "openmm_nvt": EngineSpec(
        key="openmm_nvt",
        label="OpenMM NVT MD",
        description="Langevin NVT, amber14/tip3pfb default.",
        template_file="openmm_nvt.py",
        output_filename="run.py",
        user_vars=[
            VarSpec("forcefield", "Force field XMLs (comma-sep)", "str",
                    "amber14-all.xml,amber14/tip3pfb.xml"),
            VarSpec("temperature_k", "Temperature (K)", "float", 300.0,
                    min_val=1.0, max_val=2000.0),
            VarSpec("timestep_fs", "Timestep (fs)", "float", 2.0,
                    min_val=0.1, max_val=10.0, step=0.5),
            VarSpec("n_steps", "MD steps", "int", 100_000,
                    min_val=10, max_val=100_000_000, step=10_000),
            VarSpec("report_interval", "Report every N steps", "int", 1000,
                    min_val=10, max_val=1_000_000, step=100),
            VarSpec("nonbonded_cutoff_nm", "Nonbonded cutoff (nm)", "float",
                    1.0, min_val=0.5, max_val=5.0, step=0.1),
        ],
    ),
}


def engine_defaults(key: str) -> dict[str, Any]:
    """Return {var_name: default} for the engine identified by `key`."""
    spec = ENGINES[key]
    return {v.name: v.default for v in spec.user_vars}
