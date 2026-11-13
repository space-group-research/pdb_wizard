"""Simulation-input generation shared by the terminal viewer and the desktop GUI.

The forms come from ``templates.ENGINES``; this module turns form values into files and
produces the status line and preview text both front ends show.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdb_wizard.molecule import Molecule
from pdb_wizard.templates import ENGINES, VarSpec, load_template

_FOLDERS = {
    "mpmc_isotherm": "isotherm",
    "cp2k_cell_opt": "cp2k_cell_opt",
    "cp2k_dft_md": "cp2k_md",
    "openmm_npt": "openmm_npt",
    "openmm_nvt": "openmm_nvt",
}


@dataclass
class GenOutcome:
    """What to show after an action. ``preview`` None leaves the preview as it was."""

    ok: bool
    status: str
    preview: Optional[str] = None


def _plural_s(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def default_engine_outdir(filepath: str, engine: str) -> str:
    """Folder next to the loaded file (or the current one), named after the engine."""
    base = Path(filepath).parent if filepath else Path(".")
    return str(base / _FOLDERS.get(engine, "sim_input"))


def parse_form_value(var: VarSpec, raw: Any) -> Any:
    """A form entry converted to the variable's type; empty numbers raise ValueError."""
    if var.kind == "int":
        return int(raw)
    if var.kind == "float":
        text = str(raw).strip()
        if not text:
            raise ValueError(f"{var.label} is empty")
        return float(text)
    if var.kind == "csv_floats":
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{var.label} is empty")
        return [float(p) for p in parts]
    return str(raw)


def format_validation(errors: List[str], warnings: List[str]) -> str:
    if not errors and not warnings:
        return ("✓ All checks passed\n\n"
                "Charges, LJ params, box size, and sorbate placement all look fine.")
    lines: List[str] = []
    if errors:
        lines += [f"✗ {_plural_s(len(errors), 'error')} (must fix before generating):", ""]
        lines += [f"  ✗ {e}" for e in errors] + [""]
    if warnings:
        lines += [f"⚠ {_plural_s(len(warnings), 'warning')} (generation will proceed):", ""]
        lines += [f"  ⚠ {w}" for w in warnings]
    return "\n".join(lines)


def validate_mpmc_inputs(mol: Molecule,
                         sorbate: Optional[str] = None) -> Tuple[List[str], List[str]]:
    from pdb_wizard.isotherm import validate_for_mpmc
    return validate_for_mpmc(mol, sorbate_model=sorbate)


def check_engine(mol: Molecule, engine: str,
                 values: Optional[Dict[str, Any]] = None) -> GenOutcome:
    """The Validate button: MPMC's full checks, or the periodic-box check for the others.

    ``values`` are the form's current values (the MPMC checks depend on the sorbate)."""
    if engine not in ENGINES:
        return GenOutcome(False, f"Unknown engine: {engine}")
    if engine == "mpmc_isotherm":
        errors, warnings = validate_mpmc_inputs(mol, (values or {}).get("sorbate"))
        preview = format_validation(errors, warnings)
        if errors:
            return GenOutcome(False, f"{_plural_s(len(errors), 'error')} — fix before Generate",
                              preview)
        if warnings:
            return GenOutcome(True, f"{_plural_s(len(warnings), 'warning')} — review then Generate",
                           preview)
        return GenOutcome(True, "Ready to generate", preview)
    problems = []
    if engine.startswith("cp2k") and mol.pbc is None:
        problems.append("CP2K requires a periodic box. Add one in Update Unit Cell.")
    if engine == "openmm_npt" and mol.pbc is None:
        problems.append("OpenMM NPT requires a periodic box.")
    if problems:
        return GenOutcome(False, "Fix issues before Generate", "✗ " + "\n✗ ".join(problems))
    cautions = engine_cautions(mol, engine, values)
    if cautions:
        n = _plural_s(len(cautions), "warning")
        return GenOutcome(True, f"{n} — review then Generate", "⚠ " + "\n\n⚠ ".join(cautions))
    return GenOutcome(True, "Ready to generate", "✓ No engine-level problems detected.")


def engine_cautions(mol: Molecule, engine: str,
                    values: Optional[Dict[str, Any]] = None) -> List[str]:
    """Things the CP2K and OpenMM templates do not handle for this structure. They do not
    stop generation; they say what to change in the generated files."""
    from pdb_wizard.checks import open_shell_metals, openmm_template_warning
    out: List[str] = []
    if engine.startswith("cp2k"):
        metals = open_shell_metals(mol)
        if metals:
            out.append(
                f"{', '.join(metals)} usually has unpaired electrons, and the generated input "
                f"is closed-shell. Add UKS to &DFT and set MULTIPLICITY (and check the "
                f"charge) before running, or the SCF will converge to the wrong state or "
                f"not at all.")
    if engine.startswith("openmm"):
        from pdb_wizard.templates import engine_defaults
        ff = str((values or {}).get("forcefield") or engine_defaults(engine)["forcefield"])
        warning = openmm_template_warning(mol, ff)
        if warning:
            out.append(warning)
    return out


def engine_template_text(engine: str) -> GenOutcome:
    if engine not in ENGINES:
        return GenOutcome(False, f"Unknown engine: {engine}")
    spec = ENGINES[engine]
    try:
        tpl = load_template(spec.template_file)
    except Exception as e:  # missing or unreadable template file
        return GenOutcome(False, f"Error loading template: {e}")
    return GenOutcome(True, f"Template: {spec.template_file} "
                         f"(edit the file in src/pdb_wizard/templates/ to change defaults)",
                   f"-- {spec.template_file} --\n\n{tpl}")


def generate_engine_inputs(mol: Molecule, engine: str, values: Dict[str, Any],
                           outdir: str) -> GenOutcome:
    """Write the input files for ``engine`` into ``outdir``."""
    if not outdir:
        return GenOutcome(False, "Output dir is required")
    try:
        if engine == "mpmc_isotherm":
            return _generate_mpmc(mol, values, outdir)
        if engine in ("cp2k_cell_opt", "cp2k_dft_md"):
            return _generate_cp2k(mol, values, outdir, engine.split("_", 1)[1])
        if engine in ("openmm_npt", "openmm_nvt"):
            return _generate_openmm(mol, values, outdir, engine[-3:].upper())
    except Exception as e:  # reported in the status line
        return GenOutcome(False, f"Error: {e}")
    return GenOutcome(False, f"Unknown engine: {engine}")


def _generate_mpmc(mol: Molecule, values: Dict[str, Any], outdir: str) -> GenOutcome:
    from pdb_wizard.isotherm import generate_isotherm
    errors, warnings = validate_mpmc_inputs(mol, values.get("sorbate"))
    if errors:  # refuse; warnings are allowed
        return GenOutcome(False, f"Refusing: {_plural_s(len(errors), 'error')} "
                              f"would produce a meaningless run",
                       format_validation(errors, warnings))
    dirs = generate_isotherm(
        mol, sorbate_model=values["sorbate"], temperature=values["temperature"],
        pressures=values["pressures"], output_dir=outdir, numsteps=values["numsteps"],
        corrtime=values["corrtime"], ensemble=values["ensemble"],
        insert_probability=values["insert_probability"], move_factor=values["move_factor"],
        rot_factor=values["rot_factor"],
    )
    lines: List[str] = []
    if warnings:
        lines.append(f"⚠ Proceeded with {_plural_s(len(warnings), 'warning')}:")
        lines += [f"  ⚠ {w}" for w in warnings] + [""]
    lines += [f"Generated {len(dirs)} pressure points:", f"  Sorbate: {values['sorbate']}",
              f"  Temperature: {values['temperature']} K", f"  Steps: {values['numsteps']}",
              "", "Directories:"]
    for d in dirs:
        lines += [f"  {d}/", "    input.pqr, mpmc.inp, insert.pqr, run.sh"]
    lines += ["", f"Master script: {outdir}/run_all.sh", "",
              f"To run: cd {outdir} && bash run_all.sh"]
    tail = f" ({len(warnings)} warnings)" if warnings else ""
    return GenOutcome(True, f"Generated {len(dirs)} directories in {outdir}/{tail}",
                      "\n".join(lines))


def _caution_block(mol: Molecule, engine: str, values: Dict[str, Any]) -> str:
    cautions = engine_cautions(mol, engine, values)
    return ("\n\n⚠ " + "\n\n⚠ ".join(cautions)) if cautions else ""


def _generate_cp2k(mol: Molecule, values: Dict[str, Any], outdir: str, mode: str) -> GenOutcome:
    from pdb_wizard.sim_inputs import generate_cp2k_input
    if mol.pbc is None:
        return GenOutcome(False, "CP2K requires a periodic box")
    kwargs = dict(project_name=values["project"], cutoff_Ry=values["cutoff_ry"],
                  rel_cutoff_Ry=values["rel_cutoff_ry"], xc_functional=values["xc_functional"],
                  eps_scf=values["eps_scf"], max_scf=values["max_scf"])
    if mode == "dft_md":
        kwargs.update(temperature_K=values["temperature"], timestep_fs=values["timestep_fs"],
                      n_md_steps=values["n_md_steps"], md_traj_interval=values["md_traj_interval"])
    else:
        kwargs["n_opt_steps"] = values["n_opt_steps"]
    result = generate_cp2k_input(mol, outdir, mode=mode, **kwargs)
    return GenOutcome(True, f"Wrote {result['input']}",
                   f"Wrote CP2K {mode} input:\n  {result['input']}\n  {result['xyz']}\n\n"
                   f"Elements: {', '.join(result['elements'])}\n\n"
                   f"To run: cd {outdir} && cp2k -i {Path(result['input']).name} -o out.log"
                   + _caution_block(mol, f"cp2k_{mode}", values))


def _generate_openmm(mol: Molecule, values: Dict[str, Any], outdir: str,
                     ensemble: str) -> GenOutcome:
    from pdb_wizard.sim_inputs import generate_openmm_script
    kwargs = dict(forcefield=values["forcefield"], temperature_K=values["temperature_k"],
                  timestep_fs=values["timestep_fs"], n_steps=values["n_steps"],
                  report_interval=values["report_interval"],
                  nonbonded_cutoff_nm=values["nonbonded_cutoff_nm"])
    if ensemble == "NPT":
        kwargs["pressure_atm"] = values["pressure_atm"]
    result = generate_openmm_script(mol, outdir, ensemble=ensemble, **kwargs)
    return GenOutcome(True, f"Wrote {result['script']}",
                   f"Wrote OpenMM {ensemble} driver:\n  {result['script']}\n  {result['pdb']}\n\n"
                   f"To run: cd {outdir} && python {Path(result['script']).name}"
                   + _caution_block(mol, f"openmm_{ensemble.lower()}", values))
