"""Generate input scripts for external simulation packages.

Currently supported:
  * OpenMM (Python script): NPT or NVT MD with TIP3P / amber14 by default
  * CP2K (input file): cell optimization and DFT-MD

All input files are rendered from templates under `src/pdb_wizard/templates/`
using the placeholder syntax `{{{name}}}`. This module is responsible only
for (1) writing the companion coordinate file, (2) computing derived
template variables from the loaded Molecule, and (3) invoking the
template renderer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

try:  # Literal moved into typing in 3.8; typing_extensions backports it for 3.7
    from typing import Literal
except ImportError:  # pragma: no cover - exercised only on Python 3.7
    from typing_extensions import Literal

from pdb_wizard.io import site_element, write_standard_pdb
from pdb_wizard.molecule import Molecule
from pdb_wizard.templates import load_template, render

# ---------------------------------------------------------------------------
# OpenMM
# ---------------------------------------------------------------------------

OpenMMEnsemble = Literal["NPT", "NVT"]


def _ff_python_literal(forcefield: str) -> str:
    """Convert 'amber14-all.xml,amber14/tip3pfb.xml' into a Python argument
    list: `'amber14-all.xml', 'amber14/tip3pfb.xml'` — suitable for splicing
    into `ForceField({...})` in the template."""
    return ", ".join(repr(x.strip()) for x in forcefield.split(",") if x.strip())


def _nvt_nonbonded_setup(has_pbc: bool, cutoff_nm: float) -> str:
    """Build the createSystem() nonbonded kwargs block for NVT.

    With PBC we use PME and pass a cutoff. Without PBC, OpenMM's NoCutoff
    method doesn't accept `nonbondedCutoff`, so we emit just the method.
    """
    if has_pbc:
        return (
            f"    nonbondedMethod=PME,\n"
            f"    nonbondedCutoff={cutoff_nm} * nanometer,\n"
        )
    return "    nonbondedMethod=NoCutoff,\n"


def generate_openmm_script(
    mol: Molecule,
    output_dir: str | Path,
    ensemble: OpenMMEnsemble = "NPT",
    *,
    forcefield: str = "amber14-all.xml,amber14/tip3pfb.xml",
    temperature_K: float = 300.0,
    pressure_atm: float = 1.0,
    timestep_fs: float = 2.0,
    n_steps: int = 100_000,
    report_interval: int = 1000,
    nonbonded_cutoff_nm: float = 1.0,
    pdb_filename: str = "system.pdb",
    script_filename: str = "run.py",
) -> dict:
    """Write an OpenMM Python driver + a PDB topology file.

    Returns a dict with the paths written.
    """
    if ensemble not in ("NPT", "NVT"):
        raise ValueError(f"ensemble must be NPT or NVT, got {ensemble!r}")
    if mol.pbc is None and ensemble == "NPT":
        raise ValueError("NPT simulation requires a periodic box (mol.pbc).")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # PDB topology — skip_mols_step=False so atoms group into named residues
    # (HOH, MET, etc.), which amber-style force fields need to find templates.
    pdb_path = out / pdb_filename
    write_standard_pdb(mol, str(pdb_path), skip_mols_step=False)

    common_vars = {
        "pdb_filename": pdb_filename,
        "script_filename": script_filename,
        "forcefield_xmls": _ff_python_literal(forcefield),
        "temperature_k": temperature_K,
        "timestep_fs": timestep_fs,
        "n_steps": n_steps,
        "report_interval": report_interval,
        "nonbonded_cutoff_nm": nonbonded_cutoff_nm,
    }

    if ensemble == "NPT":
        template = load_template("openmm_npt.py")
        rendered = render(template, {
            **common_vars,
            "pressure_atm": pressure_atm,
        })
    else:  # NVT
        template = load_template("openmm_nvt.py")
        rendered = render(template, {
            **common_vars,
            "nonbonded_setup": _nvt_nonbonded_setup(
                mol.pbc is not None, nonbonded_cutoff_nm,
            ),
        })

    script_path = out / script_filename
    script_path.write_text(rendered)

    return {
        "script": str(script_path),
        "pdb": str(pdb_path),
        "ensemble": ensemble,
        "n_steps": n_steps,
        "temperature_K": temperature_K,
        "pressure_atm": pressure_atm if ensemble == "NPT" else None,
    }


# ---------------------------------------------------------------------------
# CP2K
# ---------------------------------------------------------------------------

CP2KMode = Literal["cell_opt", "dft_md"]


def _cp2k_atoms(mol: Molecule) -> list:
    """(symbol, atom) for the real atoms. Sites of MPMC sorbate models get their true
    element ('COG' is carbon, not cobalt) and massless sites are left out: CP2K needs
    atoms, not model sites."""
    out = []
    for atom in mol.atoms:
        element = site_element(atom)
        if element is not None:
            out.append((element.symbol, atom))
    return out


def _formula_summary(mol: Molecule) -> str:
    counts = Counter(symbol for symbol, _ in _cp2k_atoms(mol))
    return "".join(f"{el}{n}" if n > 1 else el for el, n in sorted(counts.items()))


def _kind_section(symbol: str, basis: str = "DZVP-MOLOPT-SR-GTH",
                  potential: str = "GTH-PBE") -> str:
    """Build a CP2K &KIND block for one element. Tries to choose a sensible
    Q-value (number of valence electrons) for the GTH pseudopotential; falls
    back to '-q' suffix omitted (CP2K will pick the default)."""
    Q = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
        "Ne": 8, "Na": 9, "Mg": 10, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
        "Ar": 8, "K": 9, "Ca": 10, "Ti": 12, "V": 13, "Cr": 14, "Mn": 15,
        "Fe": 16, "Co": 17, "Ni": 18, "Cu": 11, "Zn": 12, "Ga": 13, "Br": 7,
        "I": 7, "Pt": 18, "Au": 11,
    }
    pot = potential
    if symbol in Q:
        pot = f"{potential}-q{Q[symbol]}"
    return (
        f"    &KIND {symbol}\n"
        f"      BASIS_SET {basis}\n"
        f"      POTENTIAL {pot}\n"
        f"    &END KIND\n"
    )


def _build_kind_blocks(mol: Molecule) -> str:
    elements_present = sorted({symbol for symbol, _ in _cp2k_atoms(mol)})
    return "".join(_kind_section(el) for el in elements_present)


def _write_xyz(mol: Molecule, path: Path) -> None:
    atoms = _cp2k_atoms(mol)
    lines = [str(len(atoms)), f"{_formula_summary(mol)} from pdb_wizard"]
    for symbol, a in atoms:
        lines.append(
            f"{symbol}  {a.x[0]:.6f}  {a.x[1]:.6f}  {a.x[2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _cell_vector_vars(pbc) -> dict:
    """{'cell_vec_a': 'ax ay az', ...}: the rows of the cell's basis matrix, in Å."""
    def fmt(row):
        return " ".join(f"{round(float(v), 6) + 0.0:.6f}" for v in row)

    a, b, c = pbc.basis_matrix
    return {"cell_vec_a": fmt(a), "cell_vec_b": fmt(b), "cell_vec_c": fmt(c)}


def generate_cp2k_input(
    mol: Molecule,
    output_dir: str | Path,
    mode: CP2KMode = "cell_opt",
    *,
    project_name: str = "pdbwizard",
    cutoff_Ry: float = 400.0,
    rel_cutoff_Ry: float = 60.0,
    xc_functional: str = "PBE",
    eps_scf: float = 1.0e-7,
    max_scf: int = 50,
    temperature_K: float = 300.0,
    timestep_fs: float = 0.5,
    n_md_steps: int = 10_000,
    n_opt_steps: int = 200,
    md_traj_interval: int = 10,
    coords_filename: str = "system.xyz",
    input_filename: str = "input.inp",
) -> dict:
    """Write a CP2K input file (`.inp`) plus an XYZ coordinate file.

    `mode='cell_opt'` runs CP2K's CELL_OPT to relax both atomic positions
    and the unit cell. `mode='dft_md'` runs Born-Oppenheimer MD in NVT.
    """
    if mode not in ("cell_opt", "dft_md"):
        raise ValueError(f"mode must be cell_opt or dft_md, got {mode!r}")
    if mol.pbc is None:
        raise ValueError("CP2K requires a periodic cell (mol.pbc).")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz_path = out / coords_filename
    _write_xyz(mol, xyz_path)

    # The cell vectors themselves, in the frame of the XYZ coordinates. CP2K's
    # ALPHA_BETA_GAMMA only applies together with ABC, so writing orthogonal A/B/C plus
    # angles (as before) gave a rectangular box for hexagonal or monoclinic cells.
    common_vars = {
        "project": project_name,
        **_cell_vector_vars(mol.pbc),
        "coords_file": coords_filename,
        "kind_blocks": _build_kind_blocks(mol),
        "cutoff_ry": cutoff_Ry,
        "rel_cutoff_ry": rel_cutoff_Ry,
        "xc_functional": xc_functional,
        "eps_scf": eps_scf,
        "max_scf": max_scf,
    }

    if mode == "cell_opt":
        template = load_template("cp2k_cell_opt.inp")
        rendered = render(template, {
            **common_vars,
            "n_opt_steps": n_opt_steps,
        })
    else:  # dft_md
        template = load_template("cp2k_dft_md.inp")
        rendered = render(template, {
            **common_vars,
            "temperature": temperature_K,
            "timestep_fs": timestep_fs,
            "n_md_steps": n_md_steps,
            "md_traj_interval": md_traj_interval,
        })

    inp_path = out / input_filename
    inp_path.write_text(rendered)

    return {
        "input": str(inp_path),
        "xyz": str(xyz_path),
        "mode": mode,
        "project": project_name,
        "elements": sorted({symbol for symbol, _ in _cp2k_atoms(mol)}),
    }
