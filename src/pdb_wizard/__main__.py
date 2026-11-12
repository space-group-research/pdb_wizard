"""PDB Wizard entry point — defaults to interactive TUI viewer, --classic for text menu.

Batch CLI flags (--info, --rdf, --msd, --convert, --surface, --psd) run
analysis from the command line and exit without launching the TUI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Prevent OpenBLAS/MKL threading conflicts with Python asyncio threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _ensure_installed(package: str, import_name: str | None = None) -> bool:
    """Try to import a package; if missing, offer to pip install it."""
    import_name = import_name or package
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    print(f"'{package}' is not installed.")
    try:
        answer = input("Install it now with pip? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in ("", "y", "yes"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    return False


# ------------------------------------------------------------------
# Batch-mode handlers
# ------------------------------------------------------------------

def _check_file(filepath: str) -> None:
    """Exit with error if file doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)


def _read_or_exit(filepath: str):
    """read_file with a clean one-line error instead of a traceback."""
    from pdb_wizard.io import read_file
    try:
        return read_file(filepath)
    except (ValueError, OSError) as e:
        print(f"Error: could not read {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def _batch_info(filepath: str) -> None:
    """Print system info and exit."""
    from pdb_wizard.operations import print_info

    _check_file(filepath)
    mol = _read_or_exit(filepath)
    print_info(mol, filepath)


def _warn_fallback_cell(frames) -> None:
    """Tell the user when a trajectory carried no cell (a huge box was assumed)."""
    from pdb_wizard.io import NO_CELL_FALLBACK_LENGTH, is_fallback_cell
    if frames and is_fallback_cell(frames[0].pbc):
        print(
            f"Warning: trajectory has no cell information; assuming a "
            f"{NO_CELL_FALLBACK_LENGTH:.0f} A cubic box (non-periodic). "
            "Periodic quantities such as g(r) normalisation will be meaningless.",
            file=sys.stderr,
        )


def _batch_rdf(filepath: str, element1: str, element2: str, csv_path: str | None) -> None:
    """Compute RDF and optionally write CSV."""
    _check_file(filepath)
    from pdb_wizard.io import read_file_trajectory
    from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory

    frames = read_file_trajectory(filepath)
    _warn_fallback_cell(frames)
    if frames is not None:
        r, g = compute_rdf_trajectory(frames, element1, element2)
    else:
        mol = _read_or_exit(filepath)
        r, g = compute_rdf(mol, element1, element2)

    if len(r) == 0:
        print(f"No RDF data for {element1}-{element2} (check elements or PBC).",
              file=sys.stderr)
        sys.exit(1)

    if csv_path:
        import numpy as np
        np.savetxt(csv_path, np.column_stack([r, g]),
                   header="r(A),g(r)", delimiter=",", comments="")
        print(f"RDF written to {csv_path}")
    else:
        for ri, gi in zip(r, g):
            print(f"{ri:.4f}  {gi:.6f}")


def _batch_msd(filepath: str, element: str | None, csv_path: str | None) -> None:
    """Compute MSD and optionally write CSV."""
    _check_file(filepath)
    from pdb_wizard.io import read_file_trajectory
    from pdb_wizard.msd import compute_msd

    frames = read_file_trajectory(filepath)
    if frames is None:
        print("MSD requires a trajectory file with multiple frames.",
              file=sys.stderr)
        sys.exit(1)
    _warn_fallback_cell(frames)

    lags, msd = compute_msd(frames, element=element)
    from pdb_wizard.checks import mc_trajectory_warning
    caution = mc_trajectory_warning(frames, "a diffusion constant")
    if caution:
        print(caution, file=sys.stderr)

    if len(lags) == 0:
        print("No MSD data (check element or trajectory).", file=sys.stderr)
        sys.exit(1)

    if csv_path:
        import numpy as np
        np.savetxt(csv_path, np.column_stack([lags, msd]),
                   header="lag(frames),MSD(A^2)", delimiter=",", comments="")
        print(f"MSD written to {csv_path}")
    else:
        for li, mi in zip(lags, msd):
            print(f"{li:.0f}  {mi:.6f}")


# Formats whose writers cannot work without a unit cell. A standard PDB can be
# written without CRYST1, so .pdb/.ent are not listed.
_CELL_REQUIRED_SUFFIXES = (".pqr", ".cif", ".vasp", ".poscar")


def _batch_convert(filepath: str, output: str) -> None:
    """Convert between molecular file formats.

    ``.pdb``/``.ent`` write a standardized PDB; ``.pqr`` writes an MPMC-format
    PDB (charges and force-field columns preserved); ``.xyz``, ``.cif``,
    ``.vasp``/``.poscar`` and LAMMPS data are also supported. Formats that
    need a unit cell refuse to write when the structure has none, instead of
    leaving an empty file behind.
    """
    _check_file(filepath)
    from pathlib import Path

    from pdb_wizard.io import write_mpmc_pdb, write_standard_pdb, write_xyz

    suffix = Path(output).suffix.lower()
    supported = (".xyz", ".pdb", ".ent", ".lmp", ".lammps", ".data") + _CELL_REQUIRED_SUFFIXES
    if suffix not in supported:
        print(f"Unsupported output format: {suffix}", file=sys.stderr)
        sys.exit(1)

    mol = _read_or_exit(filepath)

    if suffix in _CELL_REQUIRED_SUFFIXES and mol.pbc is None:
        print(
            f"Error: {suffix} output requires a unit cell, but {filepath} has none. "
            "Convert to .xyz or add a cell to the input first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.exists(output):
        print(f"Warning: overwriting existing file {output}", file=sys.stderr)

    if suffix == ".pqr":
        write_mpmc_pdb(mol, output)
    else:
        with open(output, "w") as out:
            if suffix == ".xyz":
                write_xyz(mol, out)
            elif suffix in (".pdb", ".ent"):
                write_standard_pdb(mol, out, skip_mols_step=True)
            elif suffix == ".cif":
                from pdb_wizard.formats import write_cif
                write_cif(mol, out)
            elif suffix in (".vasp", ".poscar"):
                from pdb_wizard.formats import write_poscar
                write_poscar(mol, out)
            else:
                from pdb_wizard.formats import write_lammps_data
                write_lammps_data(mol, out)

    print(f"Converted {filepath} -> {output}")


def _batch_surface(filepath: str) -> None:
    """Compute and print surface area."""
    _check_file(filepath)
    from pdb_wizard.operations import surface_area

    mol = _read_or_exit(filepath)
    result = surface_area(mol)
    print(f"Surface area: {result['surface_area']:.2f} A^2")
    print(f"Area per volume: {result['area_per_volume']:.2f} m^2/g")


def _batch_psd(filepath: str) -> None:
    """Compute and print pore size distribution."""
    _check_file(filepath)
    from pdb_wizard.operations import pore_size_distribution

    mol = _read_or_exit(filepath)
    centers, hist = pore_size_distribution(mol)

    if hist.sum() == 0:
        print("No pore size data (check PBC or atom count).", file=sys.stderr)
        sys.exit(1)

    print("pore_radius(A)  probability")
    for c, h in zip(centers, hist):
        print(f"{c:.4f}  {h:.6f}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    if not _ensure_installed("numpy"):
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="pdb_wizard",
        description="Molecular structure processor and interactive 3D terminal viewer",
    )
    parser.add_argument(
        "file", nargs="?",
        help="molecular structure file (.pdb, .xyz, .ent, .pqr, .zmat). "
             "Optional: launch with no file to start with an empty buffer "
             "and pick one later via File > Open.",
    )
    parser.add_argument(
        "--classic", action="store_true",
        help="use the classic text menu instead of the interactive TUI viewer",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="open the desktop window (needs PySide6: pip install \"pdb-wizard[gui]\")",
    )

    # Batch-mode analysis flags
    batch = parser.add_argument_group("batch analysis (run and exit, no TUI)")
    batch.add_argument(
        "--info", action="store_true",
        help="print system info and exit",
    )
    batch.add_argument(
        "--rdf", action="store_true",
        help="compute radial distribution function (requires -e1 and -e2)",
    )
    batch.add_argument(
        "--msd", action="store_true",
        help="compute mean squared displacement (requires -e for element, trajectory file)",
    )
    batch.add_argument(
        "--convert", action="store_true",
        help="convert file to another format (requires -o for output path; "
             ".pqr output is MPMC format)",
    )
    batch.add_argument(
        "--surface", action="store_true",
        help="compute solvent-accessible surface area",
    )
    batch.add_argument(
        "--psd", action="store_true",
        help="compute pore size distribution",
    )

    # Element selectors
    batch.add_argument("-e", dest="element", default=None, help="element symbol (for --msd)")
    batch.add_argument("-e1", dest="element1", default=None, help="first element (for --rdf)")
    batch.add_argument("-e2", dest="element2", default=None, help="second element (for --rdf)")

    # Output options
    batch.add_argument("-o", "--output", default=None, help="output file path (for --convert)")
    batch.add_argument("--csv", default=None, help="CSV output file (for --rdf, --msd)")

    args = parser.parse_args()

    # ---- batch mode dispatch ----
    # Batch flags require a file argument.
    batch_flags = (args.info, args.rdf, args.msd, args.convert,
                   args.surface, args.psd)
    if any(batch_flags) and not args.file:
        parser.error("Batch flags require a file argument.")

    if args.info:
        _batch_info(args.file)
        sys.exit(0)

    if args.rdf:
        if not args.element1 or not args.element2:
            parser.error("--rdf requires -e1 and -e2")
        _batch_rdf(args.file, args.element1, args.element2, args.csv)
        sys.exit(0)

    if args.msd:
        _batch_msd(args.file, args.element, args.csv)
        sys.exit(0)

    if args.convert:
        if not args.output:
            parser.error("--convert requires -o / --output")
        _batch_convert(args.file, args.output)
        sys.exit(0)

    if args.surface:
        _batch_surface(args.file)
        sys.exit(0)

    if args.psd:
        _batch_psd(args.file)
        sys.exit(0)

    # ---- interactive mode (existing) ----
    if args.gui:
        from pdb_wizard.gui import run_gui
        sys.exit(run_gui(args.file))
    elif args.classic:
        if not args.file:
            parser.error("Classic menu requires a file argument.")
        from pdb_wizard.menu import run_classic
        run_classic(args.file)
    else:
        if not _ensure_installed("textual"):
            # textual unavailable (missing and not installed) — fall back to the
            # classic text menu rather than exiting, so the tool still works.
            if not args.file:
                parser.error(
                    "The interactive TUI needs 'textual'. Install it, or pass a "
                    "file to use the classic text menu."
                )
            print(
                "'textual' unavailable — falling back to the classic text menu "
                "(use --classic to select it directly).",
                file=sys.stderr,
            )
            from pdb_wizard.menu import run_classic
            run_classic(args.file)
            return
        from pdb_wizard.molecule import Molecule
        from pdb_wizard.viewer import PdbWizardApp

        if not args.file:
            # No-file launch: empty buffer, user picks via File > Open
            empty = Molecule(atoms=[])
            app = PdbWizardApp(molecule=empty, filepath="")
        else:
            from pdb_wizard.io import (
                check_pdb_trajectory,
                check_xyz_trajectory,
                detect_filetype,
            )
            if not os.path.exists(args.file):
                parser.error(f"file not found: {args.file}")
            try:
                ft = detect_filetype(args.file)
            except ValueError as e:
                parser.error(str(e))
            is_traj = False
            if ft == "pdb":
                is_traj = check_pdb_trajectory(args.file)
            elif ft == "xyz":
                is_traj = check_xyz_trajectory(args.file)
            elif ft == "dcd":
                from pdb_wizard.dcd import check_dcd_trajectory
                is_traj = check_dcd_trajectory(args.file)
                if not is_traj:
                    parser.error(f"{args.file} is not a readable DCD trajectory")

            if is_traj:
                # Launch app empty, load trajectory in background
                empty = Molecule(atoms=[])
                app = PdbWizardApp(molecule=empty, filepath=args.file, load_trajectory=True)
            else:
                from pdb_wizard.io import read_file
                load_messages: list = []
                try:
                    molecule = read_file(args.file, on_message=load_messages.append)
                except (ValueError, OSError) as e:
                    parser.error(f"could not read {args.file}: {e}")
                app = PdbWizardApp(molecule=molecule, filepath=args.file,
                                   load_messages=load_messages)
        app.run()


if __name__ == "__main__":
    main()
