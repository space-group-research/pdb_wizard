"""Tests for pdb_wizard.forcefields."""

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.forcefields import apply_ff_to_system, get_forcefield


class TestGetForcefield:
    def test_oplsaa(self):
        ff = get_forcefield(0)
        assert "H" in ff
        assert "C" in ff
        assert "Cu" in ff
        assert ff["H"]["alpha"] == pytest.approx(0.41380)
        assert ff["C"]["c6"] == 0.0

    def test_phahst(self):
        ff = get_forcefield(1)
        assert "Cu" in ff
        assert "C" in ff
        assert ff["Cu"]["c6"] == pytest.approx(6.96956)
        assert ff["H"]["c10"] == pytest.approx(202.99322)

    def test_invalid_index(self):
        with pytest.raises(IndexError):
            get_forcefield(5)


class TestApplyFfToSystem:
    def test_applies_params(self):
        atoms = [Atom(0, 0, 0, "C"), Atom(1, 0, 0, "H")]
        ff = get_forcefield(0)
        result = apply_ff_to_system(atoms, ff)
        assert result[0].alpha == pytest.approx(1.2866)
        assert result[0].sigma == pytest.approx(3.55)
        assert result[1].alpha == pytest.approx(0.41380)

    def test_skips_missing_elements(self):
        atoms = [Atom(0, 0, 0, "Og")]
        ff = get_forcefield(0)
        result = apply_ff_to_system(atoms, ff)
        assert result[0].alpha == 0.0

    def test_returns_same_list(self):
        atoms = [Atom(0, 0, 0, "C")]
        ff = get_forcefield(0)
        result = apply_ff_to_system(atoms, ff)
        assert result is atoms
