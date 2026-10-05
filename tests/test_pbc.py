"""Tests for pdb_wizard.pbc."""

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.pbc import PBC


class TestPBCConstruction:
    def test_cubic(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        assert pbc.a == 10.0
        assert pbc.volume == pytest.approx(1000.0, rel=1e-6)
        np.testing.assert_array_almost_equal(
            pbc.basis_matrix, np.diag([10.0, 10.0, 10.0]), decimal=6
        )

    def test_orthorhombic(self):
        pbc = PBC(5.0, 8.0, 12.0, 90.0, 90.0, 90.0)
        assert pbc.volume == pytest.approx(480.0, rel=1e-6)

    def test_non_ortho(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
        assert pbc.volume > 0
        assert pbc.basis_matrix[1, 0] != pytest.approx(0.0)

    def test_reciprocal_is_inverse(self):
        pbc = PBC(10.0, 12.0, 8.0, 85.0, 95.0, 100.0)
        product = pbc.reciprocal_basis_matrix @ pbc.basis_matrix
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)


class TestPBCUpdate:
    def test_update_changes_matrices(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        old_vol = pbc.volume
        pbc.update(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        assert pbc.volume == pytest.approx(8000.0, rel=1e-6)
        assert pbc.volume != old_vol


class TestMinImage:
    def test_within_cell(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        dx = np.array([1.0, 2.0, 3.0])
        r = pbc.min_image(dx)
        assert r == pytest.approx(np.linalg.norm(dx), rel=1e-6)

    def test_across_boundary(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        dx = np.array([9.0, 0.0, 0.0])
        r = pbc.min_image(dx)
        assert r == pytest.approx(1.0, rel=1e-6)


class TestWrap:
    def test_wrap_centered(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        dx = np.array([15.0, 0.0, 0.0])
        wrapped = pbc.wrap(dx)
        np.testing.assert_array_almost_equal(wrapped, [-5.0, 0.0, 0.0])

    def test_wrap_forward(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        dx = np.array([-3.0, 0.0, 0.0])
        wrapped = pbc.wrap_forward(dx)
        np.testing.assert_array_almost_equal(wrapped, [7.0, 0.0, 0.0])


class TestGetAllRsMinImage:
    def test_two_atoms(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        a1 = Atom(1.0, 0.0, 0.0, "C")
        a2 = Atom(9.0, 0.0, 0.0, "C")
        i_idx, j_idx, rs = pbc.get_all_rs_min_image([a1, a2])
        assert len(rs) == 1
        assert rs[0] == pytest.approx(2.0, rel=1e-6)

    def test_three_atoms(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(0, 0, 0, "C"), Atom(1, 0, 0, "C"), Atom(0, 1, 0, "C")]
        i_idx, j_idx, rs = pbc.get_all_rs_min_image(atoms)
        assert len(rs) == 3


class TestCorners:
    def test_cubic_corners(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        corners = pbc.corners()
        assert corners.shape == (8, 3)
        np.testing.assert_array_almost_equal(corners[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(corners[7], [10, 10, 10])

    def test_corner_count(self):
        pbc = PBC(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        assert len(pbc.corners()) == 8


class TestEdges:
    def test_edge_count(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        edges = pbc.edges()
        assert len(edges) == 12

    def test_edges_valid_indices(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        for i, j in pbc.edges():
            assert 0 <= i < 8
            assert 0 <= j < 8
