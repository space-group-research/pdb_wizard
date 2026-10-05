"""Tests for pdb_wizard.atom."""

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.constants import DEFAULT_ELEMENT


class TestAtomConstruction:
    def test_basic_creation(self):
        a = Atom(1.0, 2.0, 3.0, "C1")
        assert a.element.symbol == "C"
        assert a.name == "C1"
        np.testing.assert_array_almost_equal(a.x, [1.0, 2.0, 3.0])

    def test_position_alias(self):
        a = Atom(1.0, 2.0, 3.0, "O")
        np.testing.assert_array_equal(a.position, a.x)

    def test_element_parsing_two_char(self):
        a = Atom(0, 0, 0, "Cu1")
        assert a.element.symbol == "Cu"

    def test_element_parsing_single_char(self):
        a = Atom(0, 0, 0, "H")
        assert a.element.symbol == "H"

    def test_element_parsing_strips(self):
        a = Atom(0, 0, 0, "  N  ")
        assert a.element.symbol == "N"

    def test_element_fallback_to_single_char(self):
        a = Atom(0, 0, 0, "Cx")
        assert a.element.symbol == "C"

    def test_unknown_element(self):
        a = Atom(0, 0, 0, "99")
        assert a.element == DEFAULT_ELEMENT


class TestAtomProperties:
    def test_mass(self):
        a = Atom(0, 0, 0, "C")
        assert a.mass == pytest.approx(12.011)

    def test_atomic_number(self):
        a = Atom(0, 0, 0, "Fe")
        assert a.atomic_number == 26

    def test_bond_r(self):
        h = Atom(0, 0, 0, "H")
        assert h.bond_r == pytest.approx(0.9)
        c = Atom(0, 0, 0, "C")
        assert c.bond_r == pytest.approx(1.7)

    def test_vdw(self):
        o = Atom(0, 0, 0, "O")
        assert o.vdw == pytest.approx(1.52)

    def test_ff_defaults(self):
        a = Atom(0, 0, 0, "C")
        assert a.charge == 0.0
        assert a.alpha == 0.0
        assert a.epsilon == 0.0
        assert a.sigma == 0.0
        assert a.c6 == 0.0
        assert a.c8 == 0.0
        assert a.c10 == 0.0

    def test_id_default(self):
        a = Atom(0, 0, 0, "C")
        assert a.id == 0


class TestAtomStr:
    def test_str(self):
        a = Atom(0, 0, 0, "C1")
        a.id = 5
        assert "C" in str(a)
        assert "5" in str(a)


class TestAtomCpkColor:
    def test_renderer_compatible(self):
        a = Atom(0, 0, 0, "O")
        assert a.element.cpk_color == (255, 13, 13)
        assert a.element.covalent_radius == pytest.approx(0.66)
