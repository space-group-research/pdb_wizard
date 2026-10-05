"""Tests for CIF reader in pdb_wizard.io."""

import io
import textwrap

import numpy as np
import pytest

from pdb_wizard.io import (
    _apply_symops,
    _frac_to_cart,
    _parse_symop,
    _parse_symop_full,
    _strip_cif_uncertainty,
    _tokenize_cif_line,
    detect_filetype,
    read_cif,
    read_file,
)
from pdb_wizard.pbc import PBC

# ---------------------------------------------------------------------------
# Helper: CIF content fixtures
# ---------------------------------------------------------------------------

P1_MOF_CIF = textwrap.dedent("""\
    data_MOF5
    _cell_length_a                 25.832(2)
    _cell_length_b                 25.832(2)
    _cell_length_c                 25.832(2)
    _cell_angle_alpha              90.000
    _cell_angle_beta               90.000
    _cell_angle_gamma              90.000
    _symmetry_space_group_name_H-M 'P 1'
    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Zn1  Zn  0.29410  0.29410  0.29410
    O1   O   0.25000  0.25000  0.25000
    C1   C   0.21950  0.28310  0.32070
    H1   H   0.21200  0.30500  0.35100
""")

NON_P1_CIF = textwrap.dedent("""\
    data_test_symops
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'
    '-x, -y, z'
    'x, -y, -z'
    '-x, y, -z'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Cu1  Cu  0.25  0.25  0.25
""")

CARTESIAN_CIF = textwrap.dedent("""\
    data_cartesian
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_Cartn_x
    _atom_site_Cartn_y
    _atom_site_Cartn_z
    O1  O   1.500  2.500  3.500
    H1  H   2.460  2.500  3.500
    H2  H   1.260  3.430  3.500
""")

BOTH_COORDS_CIF = textwrap.dedent("""\
    data_both
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_Cartn_x
    _atom_site_Cartn_y
    _atom_site_Cartn_z
    O1  O  0.15  0.25  0.35  1.5  2.5  3.5
""")

NO_TYPE_SYMBOL_CIF = textwrap.dedent("""\
    data_label_only
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Zn1   0.25  0.25  0.25
    Cu12  0.50  0.50  0.50
    O3    0.10  0.20  0.30
""")

TRICLINIC_CIF = textwrap.dedent("""\
    data_triclinic
    _cell_length_a   8.0
    _cell_length_b   9.0
    _cell_length_c   10.0
    _cell_angle_alpha 80.0
    _cell_angle_beta  85.0
    _cell_angle_gamma 75.0
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Fe1  Fe  0.5  0.5  0.5
""")

SPACE_GROUP_SYMOP_TAG_CIF = textwrap.dedent("""\
    data_new_tag
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _space_group_symop_operation_xyz
    'x, y, z'
    '-x, -y, -z'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Zn1  Zn  0.25  0.25  0.25
""")

COMMENT_CIF = textwrap.dedent("""\
    # This is a test CIF file with comments
    data_comments
    # Cell parameters follow
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    # Atom data
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    # More comments in the loop
    O1  O  0.10  0.20  0.30
""")

EXTRA_COLUMNS_CIF = textwrap.dedent("""\
    data_extra_cols
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_U_iso_or_equiv
    _atom_site_occupancy
    O1  O   0.10  0.20  0.30  0.025  1.0
    C1  C   0.40  0.50  0.60  0.030  1.0
""")

FRACTIONAL_SYMOP_CIF = textwrap.dedent("""\
    data_frac_symops
    _cell_length_a   10.0
    _cell_length_b   10.0
    _cell_length_c   10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'
    '-x+1/2, y+1/2, -z+1/2'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Cu1  Cu  0.10  0.20  0.30
""")


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestStripCifUncertainty:
    def test_no_uncertainty(self):
        assert _strip_cif_uncertainty("25.832") == pytest.approx(25.832)

    def test_with_uncertainty(self):
        assert _strip_cif_uncertainty("25.832(10)") == pytest.approx(25.832)

    def test_integer_value(self):
        assert _strip_cif_uncertainty("90") == pytest.approx(90.0)

    def test_integer_with_uncertainty(self):
        assert _strip_cif_uncertainty("90(1)") == pytest.approx(90.0)

    def test_negative(self):
        assert _strip_cif_uncertainty("-3.14(2)") == pytest.approx(-3.14)


class TestTokenizeCifLine:
    def test_simple_tokens(self):
        assert _tokenize_cif_line("Zn1 Zn 0.25 0.25 0.25") == [
            "Zn1", "Zn", "0.25", "0.25", "0.25"
        ]

    def test_single_quoted(self):
        tokens = _tokenize_cif_line("_tag 'P 1'")
        assert tokens == ["_tag", "P 1"]

    def test_double_quoted(self):
        tokens = _tokenize_cif_line('_tag "some value"')
        assert tokens == ["_tag", "some value"]

    def test_quoted_symop(self):
        tokens = _tokenize_cif_line("'-x+1/2, y, -z+1/2'")
        assert tokens == ["-x+1/2, y, -z+1/2"]

    def test_tabs_and_spaces(self):
        tokens = _tokenize_cif_line("O1\tO\t0.1  0.2  0.3")
        assert tokens == ["O1", "O", "0.1", "0.2", "0.3"]


class TestParseSymop:
    def test_identity_x(self):
        coeffs, trans = _parse_symop("x")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(0.0)

    def test_identity_y(self):
        coeffs, trans = _parse_symop("y")
        np.testing.assert_array_equal(coeffs, [0, 1, 0])
        assert trans == pytest.approx(0.0)

    def test_negative_x(self):
        coeffs, trans = _parse_symop("-x")
        np.testing.assert_array_equal(coeffs, [-1, 0, 0])
        assert trans == pytest.approx(0.0)

    def test_negative_x_plus_half(self):
        coeffs, trans = _parse_symop("-x+1/2")
        np.testing.assert_array_equal(coeffs, [-1, 0, 0])
        assert trans == pytest.approx(0.5)

    def test_y_plus_half(self):
        coeffs, trans = _parse_symop("y+1/2")
        np.testing.assert_array_equal(coeffs, [0, 1, 0])
        assert trans == pytest.approx(0.5)

    def test_negative_z_plus_third(self):
        coeffs, trans = _parse_symop("-z+1/3")
        np.testing.assert_array_equal(coeffs, [0, 0, -1])
        assert trans == pytest.approx(1 / 3)

    def test_with_spaces(self):
        coeffs, trans = _parse_symop(" -x + 1/2 ")
        np.testing.assert_array_equal(coeffs, [-1, 0, 0])
        assert trans == pytest.approx(0.5)

    def test_two_thirds(self):
        coeffs, trans = _parse_symop("x+2/3")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(2 / 3)

    def test_negative_translation(self):
        coeffs, trans = _parse_symop("x-1/2")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(-0.5)


class TestParseSymopFull:
    def test_identity(self):
        rot, trans = _parse_symop_full("x, y, z")
        np.testing.assert_array_equal(rot, np.eye(3))
        np.testing.assert_array_equal(trans, [0, 0, 0])

    def test_inversion(self):
        rot, trans = _parse_symop_full("-x, -y, -z")
        np.testing.assert_array_equal(rot, -np.eye(3))
        np.testing.assert_array_equal(trans, [0, 0, 0])

    def test_with_translations(self):
        rot, trans = _parse_symop_full("-x+1/2, y+1/2, -z+1/2")
        expected_rot = np.diag([-1, 1, -1]).astype(float)
        np.testing.assert_array_equal(rot, expected_rot)
        np.testing.assert_array_almost_equal(trans, [0.5, 0.5, 0.5])

    def test_wrong_number_of_components(self):
        with pytest.raises(ValueError, match="Expected 3"):
            _parse_symop_full("x, y")


class TestApplySymops:
    def test_identity_only(self):
        coords = np.array([[0.25, 0.25, 0.25]])
        rot_id = np.eye(3)
        trans_id = np.zeros(3)
        result, labels, symbols = _apply_symops(
            coords, ["Cu1"], ["Cu"], [(rot_id, trans_id)]
        )
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], [0.25, 0.25, 0.25])

    def test_inversion_generates_new_atom(self):
        coords = np.array([[0.25, 0.25, 0.25]])
        symops = [
            (np.eye(3), np.zeros(3)),
            (-np.eye(3), np.zeros(3)),
        ]
        result, labels, symbols = _apply_symops(
            coords, ["Cu1"], ["Cu"], symops
        )
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], [0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(result[1], [0.75, 0.75, 0.75])

    def test_duplicate_removal(self):
        """Atom at origin is invariant under inversion."""
        coords = np.array([[0.0, 0.0, 0.0]])
        symops = [
            (np.eye(3), np.zeros(3)),
            (-np.eye(3), np.zeros(3)),
        ]
        result, labels, symbols = _apply_symops(
            coords, ["O1"], ["O"], symops
        )
        assert len(result) == 1

    def test_atom_at_half(self):
        """Atom at (0.5, 0.5, 0.5) mapped by inversion to (-0.5,-0.5,-0.5) = (0.5,0.5,0.5) mod 1."""
        coords = np.array([[0.5, 0.5, 0.5]])
        symops = [
            (np.eye(3), np.zeros(3)),
            (-np.eye(3), np.zeros(3)),
        ]
        result, labels, symbols = _apply_symops(
            coords, ["Fe1"], ["Fe"], symops
        )
        assert len(result) == 1

    def test_four_symops(self):
        """4 symops on a general position should give 4 atoms."""
        coords = np.array([[0.1, 0.2, 0.3]])
        symops = [
            _parse_symop_full("x, y, z"),
            _parse_symop_full("-x, -y, z"),
            _parse_symop_full("x, -y, -z"),
            _parse_symop_full("-x, y, -z"),
        ]
        result, labels, symbols = _apply_symops(
            coords, ["O1"], ["O"], symops
        )
        assert len(result) == 4


class TestFracToCart:
    def test_cubic(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        frac = np.array([[0.5, 0.0, 0.0]])
        cart = _frac_to_cart(frac, pbc.basis_matrix)
        np.testing.assert_array_almost_equal(cart[0], [5.0, 0.0, 0.0])

    def test_cubic_corner(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        frac = np.array([[1.0, 1.0, 1.0]])
        cart = _frac_to_cart(frac, pbc.basis_matrix)
        np.testing.assert_array_almost_equal(cart[0], [10.0, 10.0, 10.0])

    def test_orthogonal_different_lengths(self):
        pbc = PBC(5.0, 8.0, 12.0, 90.0, 90.0, 90.0)
        frac = np.array([[0.5, 0.5, 0.5]])
        cart = _frac_to_cart(frac, pbc.basis_matrix)
        np.testing.assert_array_almost_equal(cart[0], [2.5, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Integration tests: read_cif
# ---------------------------------------------------------------------------

class TestReadCifP1:
    def test_reads_atoms(self):
        f = io.StringIO(P1_MOF_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 4
        symbols = [a.element.symbol for a in system]
        assert symbols == ["Zn", "O", "C", "H"]

    def test_reads_pbc(self):
        f = io.StringIO(P1_MOF_CIF)
        system, pbc = read_cif(f)
        assert pbc is not None
        assert pbc.a == pytest.approx(25.832)
        assert pbc.b == pytest.approx(25.832)
        assert pbc.c == pytest.approx(25.832)
        assert pbc.alpha == pytest.approx(90.0)
        assert pbc.beta == pytest.approx(90.0)
        assert pbc.gamma == pytest.approx(90.0)

    def test_positions_are_cartesian(self):
        """Positions should be in Cartesian coordinates, not fractional."""
        f = io.StringIO(P1_MOF_CIF)
        system, pbc = read_cif(f)
        # Zn1 at frac (0.29410, 0.29410, 0.29410) in cubic cell a=25.832
        # Cartesian should be ~(7.597, 7.597, 7.597)
        expected = 0.29410 * 25.832
        assert system[0].x[0] == pytest.approx(expected, abs=0.01)
        assert system[0].x[1] == pytest.approx(expected, abs=0.01)
        assert system[0].x[2] == pytest.approx(expected, abs=0.01)

    def test_atom_labels(self):
        f = io.StringIO(P1_MOF_CIF)
        system, pbc = read_cif(f)
        assert system[0].name == "Zn1"
        assert system[1].name == "O1"


class TestReadCifNonP1:
    def test_symmetry_expansion(self):
        f = io.StringIO(NON_P1_CIF)
        system, pbc = read_cif(f)
        # Cu at (0.25, 0.25, 0.25) with 4 symops should give 4 atoms
        # (general position, no special position coincidences)
        assert len(system) == 4
        for atom in system:
            assert atom.element.symbol == "Cu"

    def test_symmetry_positions_correct(self):
        """Check the actual positions after symmetry expansion."""
        f = io.StringIO(NON_P1_CIF)
        system, pbc = read_cif(f)
        # In a 10x10x10 cubic cell:
        # x,y,z -> (0.25, 0.25, 0.25) -> (2.5, 2.5, 2.5)
        # -x,-y,z -> (0.75, 0.75, 0.25) -> (7.5, 7.5, 2.5)
        # x,-y,-z -> (0.25, 0.75, 0.75) -> (2.5, 7.5, 7.5)
        # -x,y,-z -> (0.75, 0.25, 0.75) -> (7.5, 2.5, 7.5)
        positions = sorted([tuple(a.x.round(1)) for a in system])
        expected = sorted([
            (2.5, 2.5, 2.5),
            (7.5, 7.5, 2.5),
            (2.5, 7.5, 7.5),
            (7.5, 2.5, 7.5),
        ])
        for pos, exp in zip(positions, expected):
            np.testing.assert_array_almost_equal(pos, exp, decimal=1)


class TestReadCifCartesian:
    def test_reads_cartesian(self):
        f = io.StringIO(CARTESIAN_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 3
        np.testing.assert_array_almost_equal(system[0].x, [1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(system[1].x, [2.46, 2.5, 3.5])

    def test_elements(self):
        f = io.StringIO(CARTESIAN_CIF)
        system, pbc = read_cif(f)
        assert system[0].element.symbol == "O"
        assert system[1].element.symbol == "H"


class TestReadCifBothCoords:
    def test_prefers_fractional(self):
        """When both fractional and Cartesian are present, fractional should be used."""
        f = io.StringIO(BOTH_COORDS_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 1
        # Fractional (0.15, 0.25, 0.35) in 10x10x10 cubic = (1.5, 2.5, 3.5) Cartesian
        np.testing.assert_array_almost_equal(system[0].x, [1.5, 2.5, 3.5])


class TestReadCifNoTypeSymbol:
    def test_derives_symbol_from_label(self):
        f = io.StringIO(NO_TYPE_SYMBOL_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 3
        assert system[0].element.symbol == "Zn"
        assert system[1].element.symbol == "Cu"
        assert system[2].element.symbol == "O"


class TestReadCifTriclinic:
    def test_triclinic_cell(self):
        f = io.StringIO(TRICLINIC_CIF)
        system, pbc = read_cif(f)
        assert pbc is not None
        assert pbc.a == pytest.approx(8.0)
        assert pbc.b == pytest.approx(9.0)
        assert pbc.c == pytest.approx(10.0)
        assert pbc.alpha == pytest.approx(80.0)
        assert pbc.beta == pytest.approx(85.0)
        assert pbc.gamma == pytest.approx(75.0)

    def test_triclinic_position_not_simple(self):
        """In triclinic, (0.5, 0.5, 0.5) should NOT be (4, 4.5, 5) due to off-diagonal terms."""
        f = io.StringIO(TRICLINIC_CIF)
        system, pbc = read_cif(f)
        # The position is frac @ basis_matrix, which accounts for non-orthogonal axes
        assert len(system) == 1
        # Just check it's not a simple scale (which would mean orthogonal)
        assert not np.allclose(system[0].x, [4.0, 4.5, 5.0])


class TestReadCifNewSymopTag:
    def test_space_group_symop_operation_xyz(self):
        """CIF files using the newer _space_group_symop_operation_xyz tag."""
        f = io.StringIO(SPACE_GROUP_SYMOP_TAG_CIF)
        system, pbc = read_cif(f)
        # Zn at (0.25, 0.25, 0.25) + inversion -> also (0.75, 0.75, 0.75)
        assert len(system) == 2
        for atom in system:
            assert atom.element.symbol == "Zn"


class TestReadCifComments:
    def test_comments_ignored(self):
        f = io.StringIO(COMMENT_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 1
        assert system[0].element.symbol == "O"
        assert pbc is not None


class TestReadCifExtraColumns:
    def test_extra_columns_ignored(self):
        f = io.StringIO(EXTRA_COLUMNS_CIF)
        system, pbc = read_cif(f)
        assert len(system) == 2
        assert system[0].element.symbol == "O"
        assert system[1].element.symbol == "C"


class TestReadCifFractionalSymops:
    def test_half_translation_symops(self):
        f = io.StringIO(FRACTIONAL_SYMOP_CIF)
        system, pbc = read_cif(f)
        # Cu at (0.1, 0.2, 0.3) + (-x+1/2, y+1/2, -z+1/2) -> (0.4, 0.7, 0.2)
        assert len(system) == 2
        # In 10x10x10 cubic: (0.1, 0.2, 0.3) -> (1.0, 2.0, 3.0)
        # and (0.4, 0.7, 0.2) -> (4.0, 7.0, 2.0)
        assert any(
            np.allclose(a.x, [1.0, 2.0, 3.0], atol=0.01) for a in system
        )
        assert any(
            np.allclose(a.x, [4.0, 7.0, 2.0], atol=0.01) for a in system
        )


class TestReadCifEmptyFile:
    def test_empty_returns_nothing(self):
        f = io.StringIO("")
        system, pbc = read_cif(f)
        assert system == []
        assert pbc is None

    def test_cell_only_no_atoms(self):
        content = textwrap.dedent("""\
            data_empty
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
        """)
        f = io.StringIO(content)
        system, pbc = read_cif(f)
        assert system == []
        assert pbc is not None


class TestReadCifAtomIds:
    def test_atom_ids_set(self):
        f = io.StringIO(P1_MOF_CIF)
        system, pbc = read_cif(f)
        ids = [a.id for a in system]
        assert ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Integration: detect_filetype and read_file
# ---------------------------------------------------------------------------

class TestDetectFiletypeCif:
    def test_cif_detected(self):
        assert detect_filetype("structure.cif") == "cif"

    def test_cif_uppercase(self):
        assert detect_filetype("structure.CIF") == "cif"


class TestReadFileCif:
    def test_read_file_cif(self, tmp_path):
        p = tmp_path / "test.cif"
        p.write_text(P1_MOF_CIF)
        mol = read_file(str(p))
        assert len(mol.atoms) == 4
        assert mol.pbc is not None
        assert mol.pbc.a == pytest.approx(25.832)

    def test_read_file_detects_bonds(self, tmp_path):
        p = tmp_path / "test.cif"
        p.write_text(CARTESIAN_CIF)
        mol = read_file(str(p))
        # Water molecule: O-H bonds should be detected
        assert len(mol.bonds) > 0


# ---------------------------------------------------------------------------
# Edge case: special positions on cell boundary
# ---------------------------------------------------------------------------

class TestReadCifSemicolonTextField:
    def test_semicolon_field_outside_loop(self):
        """CIF with a semicolon text field (multi-line value) outside a loop."""
        cif = textwrap.dedent("""\
            data_semicolon
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            _some_text_field
            ;
            This is a multi-line
            text field value
            ;
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            O1  O  0.10  0.20  0.30
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        assert len(system) == 1
        assert system[0].element.symbol == "O"

    def test_semicolon_field_inside_loop(self):
        """CIF with a semicolon text field inside a non-atom loop."""
        cif = textwrap.dedent("""\
            data_semicolon_loop
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            loop_
            _publ_author_name
            ;
            John Smith
            ;
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Cu1  Cu  0.25  0.25  0.25
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        assert len(system) == 1
        assert system[0].element.symbol == "Cu"


class TestReadCifFracNoPbc:
    def test_frac_without_cell(self):
        """Fractional coordinates but no cell parameters -- treated as raw coords."""
        cif = textwrap.dedent("""\
            data_no_cell
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            O1  O  0.10  0.20  0.30
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        assert pbc is None
        assert len(system) == 1
        # Without a cell, fractional coords are used as-is
        np.testing.assert_array_almost_equal(system[0].x, [0.10, 0.20, 0.30])


class TestReadCifBadCoordFallback:
    def test_bad_frac_coord(self):
        """Unparseable fractional coordinate falls back to (0,0,0)."""
        cif = textwrap.dedent("""\
            data_bad_coord
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            O1  O  0.10  0.20  0.30
            C1  C  ???   0.50  0.60
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        assert len(system) == 2
        # C1 with bad coords should fall back to (0, 0, 0) Cartesian
        np.testing.assert_array_almost_equal(system[1].x, [0.0, 0.0, 0.0])


class TestReadCifMissingLabelAndSymbol:
    def test_row_with_no_label_skipped(self):
        """Rows that yield neither label nor symbol are skipped."""
        cif = textwrap.dedent("""\
            data_no_label
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            loop_
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            0.10  0.20  0.30
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        # The loop has no _atom_site_label or _atom_site_type_symbol tag
        # so atom_tag detection won't match this loop
        assert system == []


class TestReadCifDecimalSymopTranslation:
    def test_decimal_translation(self):
        """Symop with a non-fraction decimal translation."""
        coeffs, trans = _parse_symop("x+0.5")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(0.5)


class TestParseSymopEdgeCases:
    def test_trailing_sign_only(self):
        """Edge case: expression ending after sign with nothing following."""
        coeffs, trans = _parse_symop("x+")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(0.0)

    def test_integer_translation(self):
        """Plain integer as translation, like 'x+1'."""
        coeffs, trans = _parse_symop("x+1")
        np.testing.assert_array_equal(coeffs, [1, 0, 0])
        assert trans == pytest.approx(1.0)


class TestSymmetryEdgeCases:
    def test_atom_on_cell_boundary(self):
        """Atom at (0, 0, 0.5) with inversion -> (0, 0, 0.5) (same after wrap)."""
        cif = textwrap.dedent("""\
            data_edge
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            loop_
            _symmetry_equiv_pos_as_xyz
            'x, y, z'
            '-x, -y, -z'
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Fe1  Fe  0.0  0.0  0.5
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        # (0, 0, 0.5) under inversion becomes (0, 0, -0.5) = (0, 0, 0.5) mod 1
        assert len(system) == 1

    def test_multiple_atoms_with_symops(self):
        """Two distinct atoms each generating images under inversion."""
        cif = textwrap.dedent("""\
            data_multi
            _cell_length_a   10.0
            _cell_length_b   10.0
            _cell_length_c   10.0
            _cell_angle_alpha 90.0
            _cell_angle_beta  90.0
            _cell_angle_gamma 90.0
            loop_
            _symmetry_equiv_pos_as_xyz
            'x, y, z'
            '-x, -y, -z'
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Cu1  Cu  0.1  0.2  0.3
            O1   O   0.5  0.5  0.5
        """)
        f = io.StringIO(cif)
        system, pbc = read_cif(f)
        # Cu at (0.1, 0.2, 0.3) -> also (0.9, 0.8, 0.7): 2 Cu atoms
        # O at (0.5, 0.5, 0.5) -> inversion gives same: 1 O atom
        cu_atoms = [a for a in system if a.element.symbol == "Cu"]
        o_atoms = [a for a in system if a.element.symbol == "O"]
        assert len(cu_atoms) == 2
        assert len(o_atoms) == 1
