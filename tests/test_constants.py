"""Tests for pdb_wizard.constants."""

import pytest

from pdb_wizard.constants import (
    ATOMIC_NUMBER_TO_SYMBOL,
    BOHR_TO_ANGSTROM,
    DEFAULT_ELEMENT,
    ELEMENTS,
    get_element,
    get_element_by_number,
)


class TestElement:
    def test_element_is_frozen(self):
        e = ELEMENTS["H"]
        with pytest.raises(AttributeError):
            e.symbol = "X"

    def test_element_fields(self):
        carbon = ELEMENTS["C"]
        assert carbon.symbol == "C"
        assert carbon.name == "Carbon"
        assert carbon.atomic_number == 6
        assert carbon.mass == pytest.approx(12.011)
        assert carbon.cpk_color == (144, 144, 144)
        assert carbon.covalent_radius == pytest.approx(0.76)
        assert carbon.vdw_radius == pytest.approx(1.70)
        assert carbon.bond_r == pytest.approx(1.7)


class TestElementsDict:
    def test_hydrogen_present(self):
        assert "H" in ELEMENTS

    def test_all_periods_covered(self):
        for sym in ("H", "C", "Na", "Fe", "Ag", "Au", "U"):
            assert sym in ELEMENTS

    def test_placeholders(self):
        assert "Da" in ELEMENTS
        assert "X" in ELEMENTS
        assert ELEMENTS["Da"].atomic_number == 0
        assert ELEMENTS["X"].mass == 0.0

    def test_heavy_elements_present(self):
        for sym in ("Am", "Cm", "Og", "Ds"):
            assert sym in ELEMENTS

    def test_bond_r_thresholds(self):
        assert ELEMENTS["H"].bond_r == pytest.approx(0.9)
        assert ELEMENTS["He"].bond_r == pytest.approx(1.2)
        assert ELEMENTS["B"].bond_r == pytest.approx(1.2)
        assert ELEMENTS["C"].bond_r == pytest.approx(1.7)
        assert ELEMENTS["N"].bond_r == pytest.approx(1.7)
        assert ELEMENTS["O"].bond_r == pytest.approx(1.5)
        assert ELEMENTS["F"].bond_r == pytest.approx(1.5)
        assert ELEMENTS["Ne"].bond_r == pytest.approx(1.2)
        assert ELEMENTS["Si"].bond_r == pytest.approx(1.8)
        assert ELEMENTS["Fe"].bond_r == pytest.approx(1.8)

    def test_element_count(self):
        assert len(ELEMENTS) >= 120


class TestGetElement:
    def test_basic_lookup(self):
        assert get_element("C").symbol == "C"

    def test_case_insensitive(self):
        assert get_element("c").symbol == "C"
        assert get_element("CU").symbol == "Cu"
        assert get_element("cu").symbol == "Cu"

    def test_strips_whitespace(self):
        assert get_element("  C  ").symbol == "C"

    def test_unknown_returns_default(self):
        assert get_element("Zz") == DEFAULT_ELEMENT

    def test_default_element(self):
        assert DEFAULT_ELEMENT.symbol == "X"
        assert DEFAULT_ELEMENT.atomic_number == 0


class TestGetElementByNumber:
    def test_hydrogen(self):
        assert get_element_by_number(1).symbol == "H"

    def test_carbon(self):
        assert get_element_by_number(6).symbol == "C"

    def test_unknown(self):
        assert get_element_by_number(999) == DEFAULT_ELEMENT


class TestAtomicNumberToSymbol:
    def test_mappings(self):
        assert ATOMIC_NUMBER_TO_SYMBOL[1] == "H"
        assert ATOMIC_NUMBER_TO_SYMBOL[6] == "C"
        assert ATOMIC_NUMBER_TO_SYMBOL[79] == "Au"

    def test_no_zero(self):
        assert 0 not in ATOMIC_NUMBER_TO_SYMBOL


class TestBohrConversion:
    def test_value(self):
        assert BOHR_TO_ANGSTROM == pytest.approx(0.529177249)
