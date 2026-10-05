"""Classic text menu interface (from original pdb_wizard)."""

from __future__ import annotations

import os
import sys

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
    check_pdb_trajectory,
    check_xyz_trajectory,
    read_charges_file,
    read_pdb,
    read_pdb_trajectory,
    read_xyz,
    read_xyz_trajectory,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import (
    extend_axis,
    print_info,
    print_info_movie,
    wrap_atoms,
)
from pdb_wizard.pbc import PBC


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
                n = len(mol.atoms)
                if len(charges) == n:
                    for ind, atom in enumerate(mol.atoms):
                        atom.charge = charges[ind]
                elif n % len(charges) == 0:
                    for ind, atom in enumerate(mol.atoms):
                        atom.charge = charges[ind % len(charges)]
                else:
                    raise ValueError
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
                "9 = back to main menu\n\n> "
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
            mol, deleted = remove_lone_atoms(mol)
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
    return mol


def _main_loop_single(mol: Molecule, filename: str) -> None:
    mol = overlap_detector(mol)

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
                "9 = quit\n\n> "
            ))
        except ValueError:
            print("!!! Error converting input to int !!!")
            continue
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
            break
        else:
            print("\nInvalid option!")


def _main_loop_movie(molecules: list[Molecule], filename: str) -> None:
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
                "9 = quit\n\n> "
            ))
        except ValueError:
            print("!!! Error converting input to int !!!")
            continue
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
            return
        else:
            print("\nInvalid option!")


def run_classic(filename: str) -> None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    is_trajectory = False
    if ext == "xyz":
        is_trajectory = check_xyz_trajectory(filename)
    elif ext in ("pdb", "ent", "pqr"):
        is_trajectory = check_pdb_trajectory(filename)
    elif ext == "cif":
        sys.exit("Use Mercury to save a .cif as a .xyz or .pdb first")
    else:
        sys.exit("Unable to determine if input file is xyz or pdb (please rename)")

    if is_trajectory:
        with open(filename) as f:
            if ext == "xyz":
                molecules, pbcs = read_xyz_trajectory(f)
            else:
                molecules, pbcs = read_pdb_trajectory(f)
        _main_loop_movie(molecules, filename)
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
        mol.detect_bonds()
        _main_loop_single(mol, filename)
