"""Tests for pdb_wizard.renderer."""

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.renderer import (
    ImageRenderer,
    classify_periodic_bonds,
    render_scene,
    rotation_matrix,
)

C_COLOR = (144, 144, 144)
O_COLOR = (255, 13, 13)


class TestRotationMatrix:
    def test_identity(self):
        rot = rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(rot, np.eye(3), decimal=6)

    def test_orthogonal(self):
        rot = rotation_matrix(0.3, 0.5, 0.7)
        product = rot @ rot.T
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)


class TestImageRenderer:
    def test_clear(self):
        r = ImageRenderer(10, 10)
        assert r.pixels.shape == (10, 10, 3)
        assert r.z_buf.shape == (10, 10)

    def test_render_sphere(self):
        r = ImageRenderer(100, 100)
        center = np.array([0.0, 0.0, 5.0])
        r.render_sphere(center, 1.0, (255, 0, 0))
        assert np.any(r.pixels[:, :, 0] > 0)

    def test_render_bond(self):
        r = ImageRenderer(100, 100)
        p1 = np.array([-1.0, 0.0, 5.0])
        p2 = np.array([1.0, 0.0, 5.0])
        r.render_bond(p1, p2, (255, 0, 0), (0, 0, 255))
        assert np.any(r.pixels > 0)


class TestRenderMolecule:
    def test_renders_water(self):
        atoms = [
            Atom(0, 0, 0, "O"),
            Atom(0.96, 0, 0, "H"),
            Atom(-0.24, 0.93, 0, "H"),
        ]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (0, 2)])
        r = ImageRenderer(100, 100)
        rot = rotation_matrix(0, 0, 0)
        r.render_molecule(mol, rot, 5.0)
        assert np.any(r.pixels > 0)

    def test_renders_with_pbc(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(1, 1, 1, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        r = ImageRenderer(100, 100)
        rot = rotation_matrix(0, 0, 0)
        r.render_molecule(mol, rot, 30.0, pbc=pbc)
        assert np.any(r.pixels > 0)


class TestRenderScene:
    def test_returns_correct_shapes(self):
        atoms = [Atom(0, 0, 0, "C")]
        mol = Molecule(atoms=atoms)
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(50, 50, mol, rot, 5.0, ssaa=1)
        assert pixels.shape == (50, 50, 3)
        assert hit.shape == (50, 50)

    def test_ssaa_downsamples(self):
        atoms = [Atom(0, 0, 0, "C")]
        mol = Molecule(atoms=atoms)
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(50, 50, mol, rot, 5.0, ssaa=2)
        assert pixels.shape == (50, 50, 3)

    def test_custom_lighting(self):
        atoms = [Atom(0, 0, 0, "C")]
        mol = Molecule(atoms=atoms)
        rot = rotation_matrix(0, 0, 0)
        pixels, _ = render_scene(
            50, 50, mol, rot, 5.0, ssaa=1,
            ambient=0.8, diffuse=0.2,
        )
        assert pixels.shape == (50, 50, 3)


class TestClassifyPeriodicBonds:
    def _pbc(self):
        return PBC(10, 10, 10, 90, 90, 90)

    def test_wrapped_bond_flagged_with_short_displacement(self):
        # atoms near opposite x faces; the real gap is 9, min-image is -1
        pos = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])
        is_p, d_min = classify_periodic_bonds(pos, [(0, 1)], self._pbc())
        assert bool(is_p[0]) is True
        np.testing.assert_allclose(d_min[0], [-1.0, 0.0, 0.0], atol=1e-9)

    def test_normal_bond_not_flagged(self):
        pos = np.array([[4.5, 0.0, 0.0], [5.5, 0.0, 0.0]])
        is_p, d_min = classify_periodic_bonds(pos, [(0, 1)], self._pbc())
        assert bool(is_p[0]) is False
        np.testing.assert_allclose(d_min[0], [1.0, 0.0, 0.0], atol=1e-9)

    def test_no_pbc_never_periodic(self):
        pos = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])
        is_p, d_min = classify_periodic_bonds(pos, [(0, 1)], None)
        assert bool(is_p[0]) is False
        np.testing.assert_allclose(d_min[0], [9.0, 0.0, 0.0], atol=1e-9)

    def test_empty_bonds(self):
        is_p, d_min = classify_periodic_bonds(np.zeros((2, 3)), [], self._pbc())
        assert is_p.shape == (0,)
        assert d_min.shape == (0, 3)


class TestPeriodicBondRendering:
    """A bond that wraps across the cell must be drawn as two short stubs,
    never as a single line streaking across the whole structure."""

    def _capture_bond_segments(self, mol, pbc):
        r = ImageRenderer(120, 120)
        segs = []
        orig = r.render_bond

        def spy(p1, p2, c1, c2):
            segs.append(
                (np.array(p1, float), np.array(p2, float), tuple(c1), tuple(c2))
            )
            return orig(p1, p2, c1, c2)

        r.render_bond = spy
        r.render_molecule(mol, rotation_matrix(0, 0, 0), 30.0, pbc=pbc)
        # keep only molecule bonds (both endpoints coloured C or O), which
        # excludes the white cell wireframe and the r/g/b axis arrows
        return [
            (p1, p2)
            for p1, p2, c1, c2 in segs
            if c1 in (C_COLOR, O_COLOR) and c2 in (C_COLOR, O_COLOR)
        ]

    def test_wrapped_bond_becomes_two_short_stubs(self):
        atoms = [Atom(0.5, 0, 0, "C"), Atom(9.5, 0, 0, "O")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        mol.bonds = [(0, 1)]
        segs = self._capture_bond_segments(mol, mol.pbc)
        assert len(segs) == 2, "wrapped bond should render as two stubs"
        lengths = [float(np.linalg.norm(p2 - p1)) for p1, p2 in segs]
        # min-image length is 1.0, so each stub is ~0.5; the buggy behaviour
        # would draw a single ~9-unit line across the cell
        assert max(lengths) < 2.0, f"bond streaks across the cell: {lengths}"

    def test_normal_bond_drawn_full(self):
        atoms = [Atom(4.5, 0, 0, "C"), Atom(5.5, 0, 0, "O")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        mol.bonds = [(0, 1)]
        segs = self._capture_bond_segments(mol, mol.pbc)
        assert len(segs) == 1, "non-wrapped bond is a single segment"
        length = float(np.linalg.norm(segs[0][1] - segs[0][0]))
        assert abs(length - 1.0) < 0.1, length
