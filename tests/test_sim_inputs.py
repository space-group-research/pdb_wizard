"""Tests for sim_inputs.py — OpenMM and CP2K input generators."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.sim_inputs import generate_cp2k_input, generate_openmm_script


def _mof():
    atoms = []
    for i, el in enumerate(["Cu", "C", "C", "O", "O", "H"]):
        atoms.append(Atom(float(i), 0.0, 0.0, el))
    return Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))


def _no_pbc_mol():
    return Molecule(atoms=[Atom(0.0, 0.0, 0.0, "C")], pbc=None)


# ---------------------------------------------------------------------------
# OpenMM generator
# ---------------------------------------------------------------------------

class TestOpenMMGenerator:
    def test_npt_writes_script_and_pdb(self, tmp_path):
        result = generate_openmm_script(_mof(), tmp_path, ensemble="NPT")
        script = Path(result["script"])
        pdb = Path(result["pdb"])
        assert script.exists()
        assert pdb.exists()
        assert script.read_text().startswith('"""')

    def test_npt_script_contains_barostat(self, tmp_path):
        generate_openmm_script(_mof(), tmp_path, ensemble="NPT", pressure_atm=1.5)
        script = (tmp_path / "run.py").read_text()
        # Barostat block must be present and reference the requested pressure
        assert "MonteCarloBarostat" in script
        assert "1.5 * atmosphere" in script

    def test_nvt_script_omits_barostat(self, tmp_path):
        generate_openmm_script(_mof(), tmp_path, ensemble="NVT")
        script = (tmp_path / "run.py").read_text()
        # The class may be imported, but we must not actually add it as a force
        assert "addForce(MonteCarloBarostat" not in script, (
            "NVT must not add a barostat force"
        )
        # But the integrator should still be Langevin
        assert "LangevinMiddleIntegrator" in script

    def test_step_count_temperature_in_script(self, tmp_path):
        generate_openmm_script(
            _mof(), tmp_path, ensemble="NVT",
            temperature_K=350.0, n_steps=12345,
        )
        script = (tmp_path / "run.py").read_text()
        assert "350.0 * kelvin" in script
        assert "simulation.step(12345)" in script

    def test_nonpbc_mol_uses_nocutoff(self, tmp_path):
        generate_openmm_script(_no_pbc_mol(), tmp_path, ensemble="NVT")
        script = (tmp_path / "run.py").read_text()
        assert "nonbondedMethod=NoCutoff" in script

    def test_npt_on_nonpbc_mol_raises(self, tmp_path):
        with pytest.raises(ValueError, match="periodic"):
            generate_openmm_script(_no_pbc_mol(), tmp_path, ensemble="NPT")

    def test_invalid_ensemble_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ensemble"):
            generate_openmm_script(_mof(), tmp_path, ensemble="NVE")  # type: ignore

    def test_force_field_xml_list_makes_it_into_script(self, tmp_path):
        """Custom force fields should be quoted into the ForceField(...) call."""
        generate_openmm_script(
            _mof(), tmp_path, ensemble="NVT",
            forcefield="my_custom.xml,water/tip4p.xml",
        )
        script = (tmp_path / "run.py").read_text()
        assert "'my_custom.xml'" in script
        assert "'water/tip4p.xml'" in script

    def test_script_compiles_to_python_ast(self, tmp_path):
        """The generated script must parse — not a templating error."""
        import ast
        generate_openmm_script(_mof(), tmp_path, ensemble="NPT")
        script_text = (tmp_path / "run.py").read_text()
        ast.parse(script_text)  # raises SyntaxError if broken


# ---------------------------------------------------------------------------
# CP2K generator
# ---------------------------------------------------------------------------

class TestCP2KGenerator:
    def test_cell_opt_writes_input_and_xyz(self, tmp_path):
        r = generate_cp2k_input(_mof(), tmp_path, mode="cell_opt")
        assert Path(r["input"]).exists()
        assert Path(r["xyz"]).exists()

    def test_dft_md_writes_input_and_xyz(self, tmp_path):
        r = generate_cp2k_input(_mof(), tmp_path, mode="dft_md")
        assert Path(r["input"]).exists()
        assert Path(r["xyz"]).exists()

    def test_no_pbc_raises(self, tmp_path):
        with pytest.raises(ValueError, match="periodic"):
            generate_cp2k_input(_no_pbc_mol(), tmp_path, mode="cell_opt")

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            generate_cp2k_input(_mof(), tmp_path, mode="bogus")  # type: ignore

    def test_input_records_run_type(self, tmp_path):
        generate_cp2k_input(_mof(), tmp_path, mode="cell_opt", project_name="abc123")
        text = (tmp_path / "input.inp").read_text()
        assert "RUN_TYPE CELL_OPT" in text
        assert "PROJECT abc123" in text

    def test_dft_md_records_md_block(self, tmp_path):
        generate_cp2k_input(
            _mof(), tmp_path, mode="dft_md",
            temperature_K=400.0, n_md_steps=2500, timestep_fs=0.25,
        )
        text = (tmp_path / "input.inp").read_text()
        assert "RUN_TYPE MD" in text
        assert "ENSEMBLE NVT" in text
        assert "TEMPERATURE 400.0" in text
        assert "STEPS 2500" in text
        assert "TIMESTEP 0.25" in text

    def test_cell_block_contains_pbc_dimensions(self, tmp_path):
        mol = _mof()
        generate_cp2k_input(mol, tmp_path, mode="cell_opt")
        text = (tmp_path / "input.inp").read_text()
        # Box 20×20×20 was set in _mof()
        assert "A   20.000000 0.000000 0.000000" in text
        assert "B   0.000000 20.000000 0.000000" in text
        assert "C   0.000000 0.000000 20.000000" in text
        # angles are only read with ABC; the vectors above already define the cell
        assert "ALPHA_BETA_GAMMA" not in text

    def test_non_orthogonal_cell_vectors(self, tmp_path):
        """A hexagonal cell (gamma = 120) used to be written as a rectangular box: CP2K
        ignores ALPHA_BETA_GAMMA unless ABC is given."""
        import numpy as np

        from pdb_wizard.io import read_file
        mol = read_file(str(Path(__file__).resolve().parent.parent / "Ni-MOF-74.pqr"))
        generate_cp2k_input(mol, tmp_path, mode="dft_md")
        rows = {}
        for line in (tmp_path / "input.inp").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in ("A", "B", "C") and len(parts) == 4:
                rows[parts[0]] = [float(v) for v in parts[1:]]
        h = np.array([rows["A"], rows["B"], rows["C"]])
        np.testing.assert_allclose(h, mol.pbc.basis_matrix, atol=1e-5)
        lengths = np.linalg.norm(h, axis=1)
        gamma = np.degrees(np.arccos(h[0] @ h[1] / (lengths[0] * lengths[1])))
        assert gamma == pytest.approx(mol.pbc.gamma, abs=1e-3) and abs(gamma - 120) < 0.01

    def test_kind_block_per_element(self, tmp_path):
        """Each unique element must get a &KIND ... &END KIND block."""
        mol = _mof()
        generate_cp2k_input(mol, tmp_path, mode="cell_opt")
        text = (tmp_path / "input.inp").read_text()
        for el in ("Cu", "C", "O", "H"):
            assert f"&KIND {el}" in text, f"missing &KIND {el}"
            assert f"BASIS_SET DZVP-MOLOPT-SR-GTH" in text
        # No duplicate entries
        for el in ("Cu", "C", "O", "H"):
            assert text.count(f"&KIND {el}\n") == 1, (
                f"duplicate &KIND {el} block"
            )

    def test_kind_block_picks_correct_q_for_known_elements(self, tmp_path):
        """Cu should use q11 (3d¹⁰ 4s¹), O should use q6 (2s² 2p⁴)."""
        generate_cp2k_input(_mof(), tmp_path, mode="cell_opt")
        text = (tmp_path / "input.inp").read_text()
        # Find each &KIND block + its POTENTIAL line
        kinds = re.findall(r"&KIND (\w+)\s+BASIS_SET .+?\s+POTENTIAL (\S+)", text)
        kinds_dict = dict(kinds)
        assert kinds_dict["Cu"] == "GTH-PBE-q11", kinds_dict
        assert kinds_dict["O"] == "GTH-PBE-q6", kinds_dict
        assert kinds_dict["C"] == "GTH-PBE-q4", kinds_dict
        assert kinds_dict["H"] == "GTH-PBE-q1", kinds_dict

    def test_xyz_atom_count_and_elements(self, tmp_path):
        mol = _mof()
        generate_cp2k_input(mol, tmp_path, mode="cell_opt")
        xyz = (tmp_path / "system.xyz").read_text().splitlines()
        assert int(xyz[0]) == len(mol.atoms)
        # Each atom line: element + 3 coords
        for i, atom in enumerate(mol.atoms):
            parts = xyz[2 + i].split()
            assert parts[0] == atom.element.symbol, f"line {i}: {parts}"
            # Coordinate sanity
            assert float(parts[1]) == pytest.approx(atom.x[0])
            assert float(parts[2]) == pytest.approx(atom.x[1])
            assert float(parts[3]) == pytest.approx(atom.x[2])

    def test_xc_functional_records_in_input(self, tmp_path):
        generate_cp2k_input(
            _mof(), tmp_path, mode="cell_opt", xc_functional="BLYP",
        )
        text = (tmp_path / "input.inp").read_text()
        assert "&XC_FUNCTIONAL BLYP" in text

    def test_cutoff_values_record_in_input(self, tmp_path):
        generate_cp2k_input(
            _mof(), tmp_path, mode="cell_opt",
            cutoff_Ry=600.0, rel_cutoff_Ry=80.0,
        )
        text = (tmp_path / "input.inp").read_text()
        assert "CUTOFF 600.0" in text
        assert "REL_CUTOFF 80.0" in text
