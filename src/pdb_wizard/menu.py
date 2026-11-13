"""Classic text menu interface (from original pdb_wizard)."""

from __future__ import annotations

import os
import sys

from pdb_wizard.checks import load_notes, lone_atom_warning
from pdb_wizard.constants import get_element
from pdb_wizard.forcefields import (
    apply_ff_to_system_report,
    get_forcefield,
    missing_ff_messages,
)
from pdb_wizard.geometry import (
    adjust_h_distances,
    get_angles_list,
    get_bonds_list,
    get_close_contacts,
    get_lone_atoms,
    overlap_detector,
    remove_lone_atoms,
)
from pdb_wizard.io import (
    assign_framework_charges,
    check_pdb_trajectory,
    check_xyz_trajectory,
    detect_filetype,
    next_mpmc_mol_id,
    read_charges_file,
    read_file,
    read_pdb,
    read_pdb_trajectory,
    read_xyz,
    read_xyz_trajectory,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.menu_tools import (
    CmHistory,
    cm_dihedrals_list,
    cm_menu_analysis,
    cm_menu_database,
    cm_menu_edit,
    cm_menu_inputs,
    cm_menu_isotherm,
    cm_menu_measure,
    cm_menu_write_other,
    cm_place_sorbates,
    cm_prompt_sorbate_count,
    cm_prompt_sorbate_model,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import (
    extend_axis,
    print_info,
    print_info_movie,
    wrap_atoms,
)
from pdb_wizard.pbc import PBC
from pdb_wizard.sorbates import format_inserted_sorbates
from pdb_wizard.trajectory import read_trajectory


def _list_coords(mol: Molecule) -> None:
    for atom in mol.atoms:
        print(f"{atom.element.symbol} {atom.x}")


def _vmd_preview(mol: Molecule) -> None:
    write_mpmc_pdb(mol, "pdb_wizard.tmp.pdb")
    os.system("vmd pdb_wizard.tmp.pdb")
    os.system("rm pdb_wizard.tmp.pdb")


def _prompt_cell() -> tuple[float, float, float, float, float, float]:
    while True:
        try:
            a = float(input("Enter cell information\na>     "))
            b = float(input("b>     "))
            c = float(input("c>     "))
            alpha = float(input("alpha> "))
            beta = float(input("beta>  "))
            gamma = float(input("gamma> "))
            return a, b, c, alpha, beta, gamma
        except ValueError:
            print("!!! Error converting input to float !!!\n")


def _menu_update_pbc(pbc: PBC) -> PBC:
    pbc.update(*_prompt_cell())
    return pbc


def _prompt_axis_times(pbc: PBC) -> tuple[int, int] | None:
    """Prompt for an axis (0/1/2 or x/y/z) and a replication count, showing the
    current cell. Returns (axis, times), or None if the user cancels."""
    print(
        f"\nCurrent cell:\n{round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6}\n"
    )
    for row in pbc.basis_matrix:
        print(f"{row[0]:20.14f} {row[1]:20.14f} {row[2]:20.14f}")
    while True:
        try:
            axis_in = input(
                "\nWhat axis would you like to extend? 0, 1, 2 or x, y, z or q(uit)\n\n> "
            )
            if axis_in.lower() in ("q", "quit"):
                return None
            axis_map = {"x": 0, "y": 1, "z": 2}
            # Don't use dict.get(..., int(axis_in)): the default is evaluated
            # eagerly, so a letter axis would raise before the lookup.
            if axis_in.lower() in axis_map:
                axis = axis_map[axis_in.lower()]
            else:
                axis = int(axis_in)
            if axis < 0 or axis > 2:
                raise ValueError
            times = int(input("\nHow many times would you like to extend it?\n\n> "))
            if times < 1:
                raise ValueError
            return axis, times
        except ValueError:
            print("!!! Error converting input to int or x, y, z !!!")


def _prompt_frame(n_frames: int) -> int | None:
    """Prompt for a 1-based frame number, returning the 0-based index (or None
    if the user cancels)."""
    while True:
        try:
            raw = input(f"\nWhich frame? (1-{n_frames}, q to cancel)\n\n> ")
            if raw.lower() in ("q", "quit"):
                return None
            idx = int(raw) - 1
            if idx < 0 or idx >= n_frames:
                raise ValueError
            return idx
        except ValueError:
            print(f"!!! Error: enter a frame number from 1 to {n_frames} !!!")


def _menu_extend_axis(mol: Molecule) -> Molecule:
    if mol.pbc is None:
        print("No PBC data available")
        return mol
    res = _prompt_axis_times(mol.pbc)
    if res is None:
        return mol
    axis, times = res
    return extend_axis(mol, axis, times)


def _write_mpmc_options(mol: Molecule) -> None:
    if mol.pbc is None:
        return
    while True:
        try:
            wc = input(
                "\nWould you like to read in charges?\n"
                "('yes', 'y', 1 or 'no', 'n', 0)\n\n> "
            )
            write_charges = wc.lower() in ("yes", "y", "1")
            if wc.lower() not in ("yes", "y", "1", "no", "n", "0"):
                raise ValueError
            break
        except ValueError:
            print("!!! Error reading input !!!")

    if write_charges:
        while True:
            charges_filename = input(
                "\nEnter a resp file or a valid column of raw charges\n"
                "charges file name > "
            )
            try:
                charges = read_charges_file(charges_filename)
                problem = assign_framework_charges(mol, charges)
                if problem:
                    raise ValueError(problem)
                break
            except (TypeError, ValueError, FileNotFoundError) as e:
                print(f"!!! Error reading charges: {e} !!!")

    while True:
        try:
            wf = input(
                "\nWould you like to automatically apply a forcefield?\n"
                "('yes', 'y', 1 or 'no', 'n', 0)\n\n> "
            )
            write_ff = wf.lower() in ("yes", "y", "1")
            if wf.lower() not in ("yes", "y", "1", "no", "n", "0"):
                raise ValueError
            break
        except ValueError:
            print("!!! Error reading input !!!")

    if write_ff:
        while True:
            try:
                ff_in = input(
                    "\nWhich force field?\n"
                    "valid answers are 'OPLSAA' (0) or 'PHAHST' (1)\n\n> "
                )
                ff_map = {"OPLSAA": 0, "PHAHST": 1}
                key = ff_in.strip().upper()
                if key in ff_map:
                    ff_idx = ff_map[key]
                elif key.isdigit():
                    ff_idx = int(key)
                else:
                    raise ValueError
                if ff_idx not in (0, 1):
                    raise ValueError
                _, missing = apply_ff_to_system_report(mol.atoms, get_forcefield(ff_idx))
                for msg in missing_ff_messages(missing):
                    print(msg)
                break
            except ValueError:
                print("!!! Error reading input !!!")

    filename = input("\noutput filename > ")
    write_mpmc_pdb(mol, filename, write_charges=write_charges, write_params=write_ff)
    print(f"\nwrote {filename}\n")


def _menu_geom_analysis(mol: Molecule) -> Molecule:
    while True:
        option = 0
        try:
            option = int(input(
                "\nWhat would you like to do?\n\n"
                "1 = list bonds\n"
                "2 = list close vdw contacts\n"
                "3 = list angles\n"
                "4 = list lone atoms\n"
                "5 = delete lone atoms\n"
                "6 = list coordinates\n"
                "7 = edit hydrogen bond distances\n"
                "8 = preview with VMD\n"
                "9 = back to main menu\n"
                "10 = list dihedrals\n"
                "11 = measure a distance, angle or dihedral\n\n> "
            ))
        except ValueError:
            print("!!! Error converting input to int !!!")
        if option == 1:
            print("\nBonded atoms:\n")
            for msg in get_bonds_list(mol):
                print(msg)
        elif option == 2:
            print("\nClose contacts:\n")
            for msg in get_close_contacts(mol):
                print(msg)
        elif option == 3:
            print("\nAngles:\n")
            for msg in get_angles_list(mol):
                print(msg)
        elif option == 4:
            lone = get_lone_atoms(mol)
            if not lone:
                print("\nNo lone atoms found\n")
            else:
                print("\nLone atoms:\n")
                for atom in lone:
                    print(f"{atom.element.symbol:>3} {atom.id:>5} {atom.x}")
        elif option == 5:
            caution = lone_atom_warning(mol)
            mol, deleted = remove_lone_atoms(mol)
            if deleted and caution:
                print(f"\n!!! Warning: {caution} (main menu 17 undoes this) !!!")
            if not deleted:
                print("\nNo lone atoms found\n")
            else:
                print("\nDeleting lone atoms:\n")
                for atom in deleted:
                    print(f"{atom.element.symbol:>3} {atom.id:>5} {atom.x}")
        elif option == 6:
            _list_coords(mol)
        elif option == 7:
            while True:
                try:
                    el = input(
                        "\nLook for hydrogens bonded with which element? "
                        "(e.g. C, O, N, etc)\n\n> "
                    ).strip()
                    if not el or get_element(el).symbol == "X":
                        raise ValueError
                    el = get_element(el).symbol
                    dist = float(input(
                        f"\nWhat distance (in angstroms) shall {el}-H bonds be set to?\n\n> "
                    ))
                    break
                except ValueError:
                    print("!!! Error finding element or reading distance !!!")
            mol, messages = adjust_h_distances(mol, el, dist)
            if not messages:
                print(f"\nNo {el}-H bonds found\n")
            for msg in messages:
                print(msg)
        elif option == 8:
            _vmd_preview(mol)
        elif option == 9:
            return mol
        elif option == 10:
            print("\nDihedrals:\n")
            for msg in cm_dihedrals_list(mol):
                print(msg)
        elif option == 11:
            cm_menu_measure(mol)
    return mol


_NEW_OPTIONS = (
    "11 = analysis{scope}\n"
    "12 = edit structure{one}\n"
    "13 = write other formats and pictures{one}\n"
    "14 = generate simulation inputs{one}\n"
    "15 = isotherm results\n"
    "16 = fetch a structure from a database\n"
    "17 = undo the last change\n"
    "18 = open another file\n\n> "
)


def _prompt_open_file() -> str | None:
    """A readable structure file to switch to, or None."""
    path = input("\nfile to open (q to cancel) > ").strip()
    if not path or path.lower() in ("q", "quit"):
        return None
    if not os.path.isfile(path):
        print(f"!!! Error: {path} not found !!!")
        return None
    try:
        detect_filetype(path)
    except ValueError as e:
        print(f"!!! Error: {e} !!!")
        return None
    return path


def _main_loop_single(mol: Molecule, filename: str) -> str | None:
    """The single-structure menu. Returns a file to open next, or None to quit."""
    mol = overlap_detector(mol)
    history = CmHistory()

    while True:
        print_info(mol, filename)
        try:
            option = int(input(
                "\nWhat would you like to do?\n\n"
                "1 = geometry analysis\n"
                "2 = extend along axis\n"
                "3 = wrap atoms from (0, 0, 0) to (1, 1, 1)\n"
                "4 = wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2)\n"
                "5 = update cell dimensions\n"
                "6 = write .xyz\n"
                "7 = write MPMC .pdb\n"
                "8 = write standardized .pdb\n"
                "9 = quit\n"
                "10 = insert sorbates and write MPMC .pdb\n"
                + _NEW_OPTIONS.format(scope="", one="")
            ))
        except ValueError:
            print("!!! Error converting input to int !!!")
            continue
        if option in (1, 2, 3, 4, 5, 12):
            history.before(mol)
        if option == 1:
            mol = _menu_geom_analysis(mol)
        elif option == 2:
            mol = _menu_extend_axis(mol)
        elif option == 3:
            wrap_atoms(mol, forward=True)
            print("\nWrapped atoms forward of origin")
        elif option == 4:
            wrap_atoms(mol)
            print("\nWrapped atoms around origin")
        elif option == 5:
            if mol.pbc is not None:
                mol.pbc = _menu_update_pbc(mol.pbc)
                mol.bonds = []  # bonds depend on the cell; recompute lazily
        elif option == 6:
            out_filename = input("\noutput filename > ")
            with open(out_filename, "w") as out:
                write_xyz(mol, out)
            print(f"wrote {out_filename}")
        elif option == 7:
            _write_mpmc_options(mol)
        elif option == 8:
            out_filename = input("\noutput filename > ")
            with open(out_filename, "w") as out:
                write_standard_pdb(mol, out)
            print(f"wrote {out_filename}")
        elif option == 9:
            return None
        elif option == 10:
            _menu_insert_sorbates(mol)
        elif option == 11:
            cm_menu_analysis(mol, None, filename)
        elif option == 12:
            mol = cm_menu_edit(mol)
        elif option == 13:
            cm_menu_write_other(mol, filename)
        elif option == 14:
            cm_menu_inputs(mol, filename)
        elif option == 15:
            cm_menu_isotherm()
        elif option == 16:
            path = cm_menu_database()
            if path:
                return path
        elif option == 17:
            previous = history.undo()
            if previous is None:
                print("\nNothing to undo")
            else:
                mol = previous
                print("\nUndid the last change")
        elif option == 18:
            path = _prompt_open_file()
            if path:
                return path
        else:
            print("\nInvalid option!")
        if option in (1, 2, 3, 4, 5, 12):
            history.after(mol)


def _menu_insert_sorbates(mol: Molecule) -> None:
    """Classic-menu flow: pick a model, pack N copies (or fill), write MPMC .pdb."""
    if mol.pbc is None:
        print("\n!!! Inserting sorbates needs a unit cell (update cell dimensions first) !!!")
        return
    model = cm_prompt_sorbate_model()
    if model is None:
        return
    n = cm_prompt_sorbate_count()
    placed = cm_place_sorbates(mol, model, n)
    if not placed:
        return
    lines = format_inserted_sorbates(model, placed, start_atom_id=len(mol.atoms) + 1,
                                     start_mol_id=next_mpmc_mol_id(mol))
    filename = input("\noutput filename > ")
    write_mpmc_pdb(mol, filename, sorbate_lines=lines)
    print(f"\nwrote {filename}\n")


def _main_loop_movie(molecules: list[Molecule], filename: str) -> str | None:
    """The trajectory menu. Returns a file to open next, or None to quit."""
    history = CmHistory(depth=3)
    while True:
        print_info_movie(len(molecules), filename)
        n = len(molecules)
        try:
            option = int(input(
                "\nWhat would you like to do?\n\n"
                "1 = geometry analysis (single frame)\n"
                "2 = extend along axis (all frames)\n"
                "3 = wrap atoms from (0, 0, 0) to (1, 1, 1) (all frames)\n"
                "4 = wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2) (all frames)\n"
                "5 = update cell dimensions (all frames)\n"
                "6 = write .xyz (all frames)\n"
                "7 = write MPMC .pdb (single frame)\n"
                "8 = write standardized .pdb (all frames)\n"
                "9 = quit\n"
                "10 = insert sorbates and write MPMC .pdb (single frame)\n"
                + _NEW_OPTIONS.format(scope=" (whole trajectory)", one=" (single frame)")
            ))
        except ValueError:
            print("!!! Error converting input to int !!!")
            continue
        if option in (1, 2, 3, 4, 5, 12):
            history.before(molecules)
        if option == 1:
            idx = _prompt_frame(n)
            if idx is not None:
                molecules[idx] = _menu_geom_analysis(molecules[idx])
        elif option == 2:
            if molecules[0].pbc is None:
                print("No PBC data available")
            else:
                res = _prompt_axis_times(molecules[0].pbc)
                if res is not None:
                    axis, times = res
                    molecules = [extend_axis(m, axis, times) for m in molecules]
                    print(f"\nExtended all {len(molecules)} frames")
        elif option == 3:
            for m in molecules:
                wrap_atoms(m, forward=True)
            print(f"\nWrapped atoms forward of origin in all {n} frames")
        elif option == 4:
            for m in molecules:
                wrap_atoms(m)
            print(f"\nWrapped atoms around origin in all {n} frames")
        elif option == 5:
            if molecules[0].pbc is None:
                print("No PBC data available")
            else:
                vals = _prompt_cell()
                for m in molecules:
                    if m.pbc is not None:
                        m.pbc.update(*vals)
                        m.bonds = []
                print(f"\nUpdated cell dimensions in all {n} frames")
        elif option == 6:
            out_filename = input("\noutput filename > ")
            with open(out_filename, "w") as out:
                for m in molecules:
                    write_xyz(m, out)
            print(f"wrote {n} frames to {out_filename}")
        elif option == 7:
            idx = _prompt_frame(n)
            if idx is not None:
                _write_mpmc_options(molecules[idx])
        elif option == 8:
            out_filename = input("\noutput filename > ")
            with open(out_filename, "w") as out:
                for m in molecules:
                    write_standard_pdb(m, out)
            print(f"wrote {n} frames to {out_filename}")
        elif option == 9:
            return None
        elif option == 10:
            idx = _prompt_frame(n)
            if idx is not None:
                _menu_insert_sorbates(molecules[idx])
        elif option == 11:
            cm_menu_analysis(molecules[0], molecules, filename)
        elif option == 12:
            idx = _prompt_frame(n)
            if idx is not None:
                molecules[idx] = cm_menu_edit(molecules[idx])
        elif option == 13:
            idx = _prompt_frame(n)
            if idx is not None:
                cm_menu_write_other(molecules[idx], filename)
        elif option == 14:
            idx = _prompt_frame(n)
            if idx is not None:
                cm_menu_inputs(molecules[idx], filename)
        elif option == 15:
            cm_menu_isotherm()
        elif option == 16:
            path = cm_menu_database()
            if path:
                return path
        elif option == 17:
            previous = history.undo()
            if previous is None:
                print("\nNothing to undo")
            else:
                molecules = previous
                print("\nUndid the last change")
        elif option == 18:
            path = _prompt_open_file()
            if path:
                return path
        else:
            print("\nInvalid option!")
        if option in (1, 2, 3, 4, 5, 12):
            history.after(molecules)


def run_classic(filename: str) -> None:
    """Run the menu on ``filename``, then on any file the user opens from it."""
    next_file: str | None = filename
    while next_file:
        next_file = _run_classic_file(next_file)


def _run_classic_file(filename: str) -> str | None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    is_trajectory = False
    other_format = False
    file_notes: list[str] = []
    if ext == "xyz":
        is_trajectory = check_xyz_trajectory(filename)
    elif ext in ("pdb", "ent", "pqr"):
        is_trajectory = check_pdb_trajectory(filename)
    else:
        # Formats the original menu could not read (.cif, POSCAR, LAMMPS data, .dcd, ...)
        try:
            other_format = detect_filetype(filename) != "dcd"
        except ValueError:
            sys.exit("Unable to determine if input file is xyz or pdb (please rename)")
        is_trajectory = not other_format

    if is_trajectory:
        if ext == "xyz":
            with open(filename) as f:
                molecules, pbcs = read_xyz_trajectory(f)
        elif ext in ("pdb", "ent", "pqr"):
            with open(filename) as f:
                molecules, pbcs = read_pdb_trajectory(f)
        else:
            molecules = read_trajectory(filename)
        return _main_loop_movie(molecules, filename)

    if other_format:
        try:
            loaded = read_file(filename, remove_overlaps=False)
        except (OSError, ValueError) as e:
            sys.exit(f"Could not read {filename}: {e}")
        file_notes = load_notes(loaded)
        for note in file_notes:
            print(f"\n!!! Warning: {note} !!!\n")
        system, pbc = loaded.atoms, loaded.pbc
    else:
        with open(filename) as f:
            if ext == "xyz":
                system, pbc = read_xyz(f, filename)
            else:
                system, pbc = read_pdb(f, filename)

    if pbc is None:
        while True:
            try:
                print(f"Cell information not found in {filename}")
                a = float(input("a>     "))
                b = float(input("b>     "))
                c = float(input("c>     "))
                alpha = float(input("alpha> "))
                beta = float(input("beta>  "))
                gamma = float(input("gamma> "))
                break
            except ValueError:
                print("!!! Error converting input to float !!!\n")
        pbc = PBC(a, b, c, alpha, beta, gamma)

    mol = Molecule(atoms=system, pbc=pbc)
    if file_notes:
        mol.load_notes = file_notes  # so Check Structure repeats them
    mol.detect_bonds()
    return _main_loop_single(mol, filename)
