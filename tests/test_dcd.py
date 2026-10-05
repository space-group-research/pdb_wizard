"""DCD trajectory reader tests.

We synthesize valid DCD files byte-by-byte rather than depending on a real
DCD writer — keeps the test self-contained and pins the parser against an
exact byte layout.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.dcd import (
    check_dcd_trajectory,
    read_dcd_trajectory,
)


# ---------------------------------------------------------------------------
# DCD synthesizer
# ---------------------------------------------------------------------------

def _record(payload: bytes, endian: str = "<") -> bytes:
    """Wrap a payload with Fortran 4-byte length brackets."""
    n = struct.pack(endian + "i", len(payload))
    return n + payload + n


def _make_dcd(
    coords: np.ndarray,                      # (n_frames, n_atoms, 3) float
    cell: np.ndarray | None = None,          # (n_frames, 6) [a, b, c, α, β, γ]
    endian: str = "<",
) -> bytes:
    """Build a complete DCD file in memory. Returns the bytes."""
    n_frames, n_atoms, _ = coords.shape
    has_box = cell is not None

    # Header: 4-byte 'CORD' + 20 int32
    icntrl = [0] * 20
    icntrl[0] = n_frames        # NSET
    icntrl[1] = 1               # ISTART
    icntrl[2] = 1               # NSAVC
    icntrl[3] = n_frames        # NSTEP
    icntrl[10] = 1 if has_box else 0
    icntrl[19] = 24             # CHARMM version
    header_payload = b"CORD" + struct.pack(endian + "20i", *icntrl)
    out = _record(header_payload, endian)

    # Title block: ntitle (4 bytes) + ntitle * 80 chars
    title = b"created by test_dcd.py".ljust(80, b" ")
    title_payload = struct.pack(endian + "i", 1) + title
    out += _record(title_payload, endian)

    # N_atoms record
    out += _record(struct.pack(endian + "i", n_atoms), endian)

    # Per-frame
    for fi in range(n_frames):
        if has_box:
            a, b, c, alpha, beta, gamma = cell[fi]
            # Pack in CHARMM/NAMD order: a, gamma, b, beta, alpha, c (as angles in degrees)
            box_payload = struct.pack(
                endian + "6d",
                float(a), float(gamma), float(b),
                float(beta), float(alpha), float(c),
            )
            out += _record(box_payload, endian)

        xs = coords[fi, :, 0].astype(endian + "f4").tobytes()
        ys = coords[fi, :, 1].astype(endian + "f4").tobytes()
        zs = coords[fi, :, 2].astype(endian + "f4").tobytes()
        out += _record(xs, endian)
        out += _record(ys, endian)
        out += _record(zs, endian)

    return out


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

class TestParseSyntheticDcd:
    def test_round_trip_coords_no_box(self, tmp_path):
        n_frames = 4
        n_atoms = 5
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (n_frames, n_atoms, 3)).astype(np.float32)
        path = tmp_path / "traj.dcd"
        path.write_bytes(_make_dcd(coords))

        frames, pbcs = read_dcd_trajectory(str(path))
        assert len(frames) == n_frames
        assert all(p is None for p in pbcs)
        for fi in range(n_frames):
            for ai in range(n_atoms):
                a = frames[fi].atoms[ai]
                # float32 round-trip — exact equality fine since we wrote float32
                assert pytest.approx(coords[fi, ai, 0], abs=1e-5) == a.x[0]
                assert pytest.approx(coords[fi, ai, 1], abs=1e-5) == a.x[1]
                assert pytest.approx(coords[fi, ai, 2], abs=1e-5) == a.x[2]

    def test_pbc_box_round_trip(self, tmp_path):
        n_frames, n_atoms = 3, 4
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        cell = np.tile([20.0, 30.0, 40.0, 90.0, 95.0, 90.0], (n_frames, 1))
        path = tmp_path / "traj.dcd"
        path.write_bytes(_make_dcd(coords, cell=cell))

        frames, pbcs = read_dcd_trajectory(str(path))
        assert len(frames) == n_frames
        for pbc in pbcs:
            assert pbc is not None
            assert pbc.a == pytest.approx(20.0)
            assert pbc.b == pytest.approx(30.0)
            assert pbc.c == pytest.approx(40.0)
            assert pbc.alpha == pytest.approx(90.0, abs=1e-3)
            assert pbc.beta == pytest.approx(95.0, abs=1e-3)
            assert pbc.gamma == pytest.approx(90.0, abs=1e-3)

    def test_check_dcd_trajectory_recognizes_valid_file(self, tmp_path):
        coords = np.zeros((2, 3, 3), dtype=np.float32)
        path = tmp_path / "traj.dcd"
        path.write_bytes(_make_dcd(coords))
        assert check_dcd_trajectory(str(path)) is True

    def test_check_rejects_garbage(self, tmp_path):
        path = tmp_path / "junk.dcd"
        path.write_bytes(b"not a real dcd file" * 20)
        assert check_dcd_trajectory(str(path)) is False

    def test_check_rejects_empty(self, tmp_path):
        path = tmp_path / "empty.dcd"
        path.write_bytes(b"")
        assert check_dcd_trajectory(str(path)) is False

    def test_big_endian_dcd_parsed(self, tmp_path):
        """Some legacy CHARMM machines wrote big-endian DCDs. Auto-detect."""
        coords = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        path = tmp_path / "be.dcd"
        path.write_bytes(_make_dcd(coords, endian=">"))
        frames, _ = read_dcd_trajectory(str(path))
        assert len(frames) == 1
        a = frames[0].atoms[0]
        assert a.x[0] == pytest.approx(1.0)
        assert a.x[1] == pytest.approx(2.0)
        assert a.x[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Topology integration
# ---------------------------------------------------------------------------

class TestTopologyAssignment:
    def _write_xyz_topology(self, path: Path, elements: list[str]) -> None:
        lines = [str(len(elements)), ""]
        for el in elements:
            lines.append(f"{el} 0.0 0.0 0.0")
        path.write_text("\n".join(lines) + "\n")

    def test_explicit_topology_assigns_elements(self, tmp_path):
        coords = np.zeros((1, 4, 3), dtype=np.float32)
        dcd = tmp_path / "t.dcd"
        dcd.write_bytes(_make_dcd(coords))
        topo = tmp_path / "topology.xyz"
        self._write_xyz_topology(topo, ["O", "H", "H", "Na"])

        frames, _ = read_dcd_trajectory(str(dcd), topology_file=str(topo))
        elements = [a.element.symbol for a in frames[0].atoms]
        assert elements == ["O", "H", "H", "Na"]
        # Topology was provided, so the missing-flag should be unset
        assert not getattr(frames[0], "_dcd_topology_missing", False)

    def test_sibling_topology_auto_discovered(self, tmp_path):
        coords = np.zeros((1, 3, 3), dtype=np.float32)
        dcd = tmp_path / "sim.dcd"
        dcd.write_bytes(_make_dcd(coords))
        # Write topology with the SAME stem, .xyz extension
        sibling = tmp_path / "sim.xyz"
        self._write_xyz_topology(sibling, ["C", "N", "O"])

        frames, _ = read_dcd_trajectory(str(dcd))
        elements = [a.element.symbol for a in frames[0].atoms]
        assert elements == ["C", "N", "O"]

    def test_no_topology_falls_back_to_carbon_with_flag(self, tmp_path):
        coords = np.zeros((1, 7, 3), dtype=np.float32)
        dcd = tmp_path / "lonely.dcd"
        dcd.write_bytes(_make_dcd(coords))

        frames, _ = read_dcd_trajectory(str(dcd))
        assert len(frames) == 1
        elements = [a.element.symbol for a in frames[0].atoms]
        assert elements == ["C"] * 7
        assert getattr(frames[0], "_dcd_topology_missing", False) is True

    def test_topology_with_wrong_atom_count_is_ignored(self, tmp_path):
        """If the sibling topology has the wrong atom count, fall back to
        carbon — don't silently mis-assign elements."""
        coords = np.zeros((1, 5, 3), dtype=np.float32)
        dcd = tmp_path / "mismatch.dcd"
        dcd.write_bytes(_make_dcd(coords))
        topo = tmp_path / "mismatch.xyz"
        self._write_xyz_topology(topo, ["C", "N"])  # only 2 atoms

        frames, _ = read_dcd_trajectory(str(dcd))
        elements = [a.element.symbol for a in frames[0].atoms]
        assert elements == ["C"] * 5
        assert getattr(frames[0], "_dcd_topology_missing", False) is True


# ---------------------------------------------------------------------------
# io.py integration: read_file_trajectory dispatches correctly
# ---------------------------------------------------------------------------

class TestIoIntegration:
    def test_read_file_trajectory_handles_dcd(self, tmp_path):
        from pdb_wizard.io import read_file_trajectory
        coords = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ], dtype=np.float32)
        dcd = tmp_path / "x.dcd"
        dcd.write_bytes(_make_dcd(coords))

        # Sibling topology so atoms get real elements
        topo = tmp_path / "x.xyz"
        topo.write_text("2\n\nC 0 0 0\nO 0 0 0\n")

        frames = read_file_trajectory(str(dcd))
        assert frames is not None
        assert len(frames) == 2
        assert [a.element.symbol for a in frames[0].atoms] == ["C", "O"]

    def test_detect_filetype_recognizes_dcd(self):
        from pdb_wizard.io import detect_filetype
        assert detect_filetype("/tmp/anything.dcd") == "dcd"


# ---------------------------------------------------------------------------
# Real OpenMM DCD fixture
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
OPENMM_DCD = FIXTURE_DIR / "openmm_water.dcd"
OPENMM_PDB = FIXTURE_DIR / "openmm_water.pdb"


@pytest.mark.skipif(
    not (OPENMM_DCD.exists() and OPENMM_PDB.exists()),
    reason="Run scripts/make_openmm_dcd_fixture.py first",
)
class TestOpenMMDcdFixture:
    """Parse a DCD generated by OpenMM (NPT, 3 H2O molecules, 20 frames)."""

    def test_frame_count_matches_openmm_setup(self):
        # Generator writes 100 steps with reporter every 5 steps → 20 frames
        frames, pbcs = read_dcd_trajectory(str(OPENMM_DCD))
        assert len(frames) == 20

    def test_atom_count_matches_topology(self):
        # 3 waters × 3 atoms = 9
        frames, _ = read_dcd_trajectory(str(OPENMM_DCD))
        assert len(frames[0].atoms) == 9

    def test_sibling_pdb_assigns_water_elements(self):
        """The .pdb fixture sits next to the .dcd — the reader should
        auto-discover it and assign O/H elements."""
        frames, _ = read_dcd_trajectory(str(OPENMM_DCD))
        elements = [a.element.symbol for a in frames[0].atoms]
        # 3 waters: O,H,H,O,H,H,O,H,H
        assert elements == ["O", "H", "H"] * 3, elements
        assert not getattr(frames[0], "_dcd_topology_missing", False)

    def test_npt_box_changes_across_frames(self):
        """NPT means the box volume fluctuates — the box dimensions in the
        DCD must NOT be constant across all frames."""
        frames, pbcs = read_dcd_trajectory(str(OPENMM_DCD))
        boxes = [(p.a, p.b, p.c) for p in pbcs if p is not None]
        assert len(boxes) == len(frames), "every frame should have a PBC under NPT"
        # Compute volume per frame; require variance > 0
        volumes = [a * b * c for (a, b, c) in boxes]
        assert max(volumes) > min(volumes), (
            f"NPT box should fluctuate; volumes = {volumes}"
        )

    def test_box_dimensions_are_physically_plausible(self):
        """1.5 nm = 15 Å initial box. Sub-1 Å or huge values mean the
        endianness or unit detection is wrong."""
        frames, pbcs = read_dcd_trajectory(str(OPENMM_DCD))
        for pbc in pbcs:
            assert pbc is not None
            for L in (pbc.a, pbc.b, pbc.c):
                assert 5.0 < L < 50.0, f"implausible box length {L} Å"
            for ang in (pbc.alpha, pbc.beta, pbc.gamma):
                # NPT cubic-only barostat keeps angles at 90°
                assert abs(ang - 90.0) < 1.0, f"unexpected angle {ang}"

    def test_coords_evolve_between_frames(self):
        """A real MD trajectory has motion — frames must not be identical."""
        frames, _ = read_dcd_trajectory(str(OPENMM_DCD))
        first = np.array([a.x for a in frames[0].atoms])
        last = np.array([a.x for a in frames[-1].atoms])
        # Each atom should move at least a small amount over 100 steps × 1 fs
        # at 300 K — but coordinates are also affected by NPT scaling.
        max_disp = float(np.linalg.norm(first - last, axis=1).max())
        assert max_disp > 0.001, (
            f"frame 0 and frame -1 too similar: max displacement {max_disp:.4f} Å"
        )


# ---------------------------------------------------------------------------
# Error handling: corrupt / mismatched files surface real errors
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_short_file_raises(self, tmp_path):
        path = tmp_path / "short.dcd"
        path.write_bytes(b"\x54\x00\x00\x00")  # only the 4-byte length
        with pytest.raises(Exception):
            read_dcd_trajectory(str(path))

    def test_record_bracket_mismatch_raises(self, tmp_path):
        """Tamper with the trailing bracket — parser must catch it."""
        coords = np.zeros((1, 2, 3), dtype=np.float32)
        path = tmp_path / "tampered.dcd"
        good = _make_dcd(coords)
        # Flip the last record's tail length so it doesn't match the head
        bad = good[:-4] + struct.pack("<i", 999999)
        path.write_bytes(bad)
        with pytest.raises(Exception):
            read_dcd_trajectory(str(path))
