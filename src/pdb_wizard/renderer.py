"""3D braille rendering engine (adapted from moltui, MO/isosurface code removed)."""

from __future__ import annotations

import math

import numpy as np

from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def classify_periodic_bonds(positions, bonds, pbc, tol: float = 1e-6):
    """Split bonds into normal vs. periodic (wrapped across the cell).

    Bonds are detected with the minimum-image convention, so a bond may connect
    two atoms sitting on opposite faces of the cell. Drawing a straight line
    between their real coordinates streaks across the whole structure; instead
    such bonds should be drawn as stubs toward each atom's nearest periodic
    image.

    Returns ``(is_periodic, d_min)`` where ``is_periodic[k]`` is True when bond
    ``k`` wraps, and ``d_min[k]`` is its minimum-image displacement
    ``pos_j - pos_i`` (real space). With no PBC, all bonds are non-periodic.
    """
    n = len(bonds)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 3))
    n_atoms = len(positions)
    bond_i = np.clip(np.array([b[0] for b in bonds]), 0, max(n_atoms - 1, 0))
    bond_j = np.clip(np.array([b[1] for b in bonds]), 0, max(n_atoms - 1, 0))
    d_raw = positions[bond_j] - positions[bond_i]
    if pbc is None:
        return np.zeros(n, dtype=bool), d_raw
    d_min = pbc.wrap(d_raw)
    is_periodic = np.linalg.norm(d_raw - d_min, axis=1) > tol
    return is_periodic, d_min


class ImageRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        self.width = width
        self.height = height
        self.fov = 1.5
        # When True, use parallel-projection (orthographic) instead of
        # perspective. The orthographic scale is set by `ortho_scale` —
        # render_scene picks a value matching the camera distance so that
        # toggling perspective/ortho keeps the molecule roughly the same
        # apparent size.
        self.orthographic = False
        self.ortho_scale = 1.0
        self.light_dir = np.array([0.4, 0.7, -0.6])
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.ambient = 0.35
        self.diffuse_strength = 0.6
        self.specular_strength = 0.4
        self.shininess = 32.0
        self.view_dir = np.array([0.0, 0.0, -1.0])
        self.half_vec = self.light_dir + self.view_dir
        self.half_vec /= np.linalg.norm(self.half_vec)
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.bg_color = np.array(bg_color, dtype=np.uint8)
        self.clear()

    def clear(self) -> None:
        self.pixels = np.tile(self.bg_color, (self.height, self.width, 1))
        self.z_buf = np.full((self.height, self.width), float("inf"))

    def _project(self, point: np.ndarray) -> tuple[float, float, float]:
        x, y, z = point
        scale = min(self.width, self.height) / 2
        if self.orthographic:
            # Parallel projection: px,py independent of z. Anything in front
            # of the camera (z > 0) renders; we still need a valid z for the
            # depth buffer, but no perspective divide.
            sx = self.width / 2 + x * self.ortho_scale * scale
            sy = self.height / 2 - y * self.ortho_scale * scale
            return (sx, sy, z if z > 0 else 0.1)
        if z <= 0.1:
            return (float("nan"), float("nan"), z)
        px = x * self.fov / z
        py = y * self.fov / z
        sx = self.width / 2 + px * scale
        sy = self.height / 2 - py * scale
        return (sx, sy, z)

    def render_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[int, int, int],
    ) -> None:
        sx, sy, sz = self._project(center)
        if math.isnan(sx):
            return

        scale = min(self.width, self.height) / 2
        if self.orthographic:
            pr = radius * self.ortho_scale * scale
        else:
            pr = radius * self.fov / sz * scale
        if pr < 0.5:
            return

        x_min = max(0, int(sx - pr - 1))
        x_max = min(self.width - 1, int(sx + pr + 1))
        y_min = max(0, int(sy - pr - 1))
        y_max = min(self.height - 1, int(sy + pr + 1))

        if x_min > x_max or y_min > y_max:
            return

        ys = np.arange(y_min, y_max + 1)
        xs = np.arange(x_min, x_max + 1)
        px_grid, py_grid = np.meshgrid(xs, ys)

        dx = (px_grid - sx) / pr
        dy = (py_grid - sy) / pr
        dist_sq = dx * dx + dy * dy
        mask = dist_sq <= 1.0

        dz = np.sqrt(np.maximum(0.0, 1.0 - dist_sq))
        norm_len = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-10
        nx = dx / norm_len
        ny = -dy / norm_len
        nz = dz / norm_len

        n_dot_l = np.maximum(
            0.0,
            nx * self.light_dir[0] + ny * self.light_dir[1] + nz * self.light_dir[2],
        )
        n_dot_h = np.maximum(
            0.0,
            nx * self.half_vec[0] + ny * self.half_vec[1] + nz * self.half_vec[2],
        )
        specular = np.power(n_dot_h, self.shininess) * self.specular_strength
        diffuse = n_dot_l * self.diffuse_strength
        intensity = np.minimum(1.0, self.ambient + diffuse)

        point_z = sz - radius * dz

        z_slice = self.z_buf[y_min: y_max + 1, x_min: x_max + 1]
        valid = mask & (point_z < z_slice)

        z_slice[valid] = point_z[valid]

        color_arr = np.array(color, dtype=np.float64)
        shaded = np.minimum(
            255, color_arr[None, None, :] * intensity[:, :, None] + 255 * specular[:, :, None]
        ).astype(np.uint8)
        for c in range(3):
            channel = self.pixels[y_min: y_max + 1, x_min: x_max + 1, c]
            channel[valid] = shaded[:, :, c][valid]

    def render_bond(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        color1: tuple[int, int, int],
        color2: tuple[int, int, int],
    ) -> None:
        sx1, sy1, sz1 = self._project(p1)
        sx2, sy2, sz2 = self._project(p2)
        if math.isnan(sx1) or math.isnan(sx2):
            return

        scale = min(self.width, self.height) / 2
        mid_z = (sz1 + sz2) / 2
        if self.orthographic:
            pr = self.bond_radius * self.ortho_scale * scale
        else:
            pr = self.bond_radius * self.fov / mid_z * scale

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1.0:
            return

        # Skip if entirely off-screen
        margin = pr + 2
        if (min(sx1, sx2) > self.width + margin or max(sx1, sx2) < -margin or
                min(sy1, sy2) > self.height + margin or max(sy1, sy2) < -margin):
            return

        nx, ny = -dy / length, dx / length
        half_w = max(1.0, pr)
        steps = int(length * 2) + 1

        hw = int(half_w + 1)
        ts = np.linspace(0, 1, steps + 1)
        offsets = np.arange(-hw, hw + 1, dtype=np.float64)
        d_norm = offsets / half_w
        off_mask = np.abs(d_norm) <= 1.0
        offsets = offsets[off_mask]
        d_norm = d_norm[off_mask]

        cxs = sx1 + dx * ts
        cys = sy1 + dy * ts
        czs = sz1 + (sz2 - sz1) * ts

        all_px = np.round(cxs[:, None] + nx * offsets[None, :]).astype(int)
        all_py = np.round(cys[:, None] + ny * offsets[None, :]).astype(int)

        cyl_nz = np.sqrt(1.0 - d_norm * d_norm)
        cyl_nx_v = nx * d_norm
        cyl_ny_v = -ny * d_norm
        norm_len = np.sqrt(cyl_nx_v**2 + cyl_ny_v**2 + cyl_nz**2) + 1e-10
        cyl_nx_v /= norm_len
        cyl_ny_v /= norm_len
        cyl_nz /= norm_len
        diffuse = np.maximum(
            0.0,
            cyl_nx_v * self.light_dir[0]
            + cyl_ny_v * self.light_dir[1]
            + cyl_nz * self.light_dir[2],
        )
        intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)
        pz = czs[:, None] - self.bond_radius * cyl_nz[None, :]
        intensity_2d = np.broadcast_to(intensity[None, :], pz.shape)

        flat_px = all_px.ravel()
        flat_py = all_py.ravel()
        flat_pz = pz.ravel()
        flat_int = intensity_2d.ravel()

        n_steps = len(ts)
        n_off = len(offsets)
        step_idx = np.repeat(np.arange(n_steps), n_off)
        half = n_steps // 2
        is_first_half = step_idx <= half

        valid = (flat_px >= 0) & (flat_px < self.width) & (flat_py >= 0) & (flat_py < self.height)
        flat_px, flat_py, flat_pz, flat_int, is_first_half = (
            flat_px[valid], flat_py[valid], flat_pz[valid], flat_int[valid], is_first_half[valid],
        )

        if len(flat_px) == 0:
            return

        z_current = self.z_buf[flat_py, flat_px]
        z_pass = flat_pz < z_current
        flat_px = flat_px[z_pass]
        flat_py = flat_py[z_pass]
        flat_pz = flat_pz[z_pass]
        flat_int = flat_int[z_pass]
        is_first_half = is_first_half[z_pass]

        if len(flat_px) == 0:
            return

        self.z_buf[flat_py, flat_px] = flat_pz

        c1 = np.array(color1, dtype=np.float64)
        c2 = np.array(color2, dtype=np.float64)
        colors = np.where(is_first_half[:, None], c1[None, :], c2[None, :])
        shaded = np.minimum(255, colors * flat_int[:, None]).astype(np.uint8)
        self.pixels[flat_py, flat_px] = shaded

    @staticmethod
    def _highlight_color() -> tuple[int, int, int]:
        return (255, 255, 50)

    @staticmethod
    def _charge_color(charge: float) -> tuple[int, int, int]:
        """Map charge to color: red (negative) -> white (zero) -> blue (positive)."""
        c = max(-1.0, min(1.0, charge))
        if c >= 0:
            # Positive = blue
            r = int(255 * (1 - c))
            g = int(255 * (1 - c))
            b = 255
        else:
            # Negative = red
            r = 255
            g = int(255 * (1 + c))
            b = int(255 * (1 + c))
        return (r, g, b)

    # Palette for residue / chain / index coloring
    _PALETTE = [
        (50, 130, 255), (255, 100, 80), (80, 220, 100), (255, 200, 50),
        (200, 100, 255), (100, 220, 220), (255, 150, 200), (180, 180, 100),
        (255, 130, 50), (100, 180, 255), (220, 80, 180), (80, 200, 160),
        (200, 200, 80), (150, 100, 200), (80, 160, 255), (255, 180, 130),
    ]

    def _compute_atom_colors(
        self, molecule: Molecule, color_mode: str,
    ) -> list[tuple[int, int, int]]:
        """Compute per-atom colors based on the selected mode."""
        if color_mode == "charge":
            return [self._charge_color(a.charge) for a in molecule.atoms]
        if color_mode == "residue":
            # Group by residue_id if set, otherwise by name runs
            pal = self._PALETTE
            colors = []
            prev_name = None
            group_id = -1
            for a in molecule.atoms:
                rid = getattr(a, "residue_id", 0)
                if rid:
                    colors.append(pal[rid % len(pal)])
                else:
                    if a.name != prev_name:
                        group_id += 1
                        prev_name = a.name
                    colors.append(pal[group_id % len(pal)])
            return colors
        if color_mode == "chain":
            # Color by chain breaks (CA-CA distance > 5A or residue_id resets)
            pal = self._PALETTE
            chain_id = 0
            colors: list[tuple[int, int, int]] = []
            prev_pos = None
            for a in molecule.atoms:
                if prev_pos is not None:
                    d = float(np.linalg.norm(a.position - prev_pos))
                    if d > 8.0:
                        chain_id += 1
                prev_pos = a.position
                colors.append(pal[chain_id % len(pal)])
            return colors
        if color_mode == "index":
            # Rainbow gradient by atom index
            n = max(1, len(molecule.atoms))
            colors = []
            for i in range(len(molecule.atoms)):
                frac = i / n
                # HSV-like rainbow: red -> yellow -> green -> cyan -> blue -> magenta
                h = frac * 6.0
                x = int(255 * (1 - abs(h % 2 - 1)))
                if h < 1:
                    colors.append((255, x, 0))
                elif h < 2:
                    colors.append((x, 255, 0))
                elif h < 3:
                    colors.append((0, 255, x))
                elif h < 4:
                    colors.append((0, x, 255))
                elif h < 5:
                    colors.append((x, 0, 255))
                else:
                    colors.append((255, 0, x))
            return colors
        # Default: element CPK
        return [a.element.cpk_color for a in molecule.atoms]

    def render_molecule(
        self,
        molecule: Molecule,
        rot: np.ndarray,
        camera_distance: float,
        pbc: PBC | None = None,
        pan: tuple[float, float] = (0.0, 0.0),
        highlighted_atoms: set[int] | None = None,
        licorice: bool = False,
        vdw: bool = False,
        color_by_charge: bool = False,
        color_mode: str = "element",
        centroid_override: np.ndarray | None = None,
        ribbon: bool = False,
        show_polyhedra: bool = False,
        density_positions: np.ndarray | None = None,
        density_values: np.ndarray | None = None,
    ) -> None:
        self.clear()
        # color_by_charge is legacy — map to color_mode
        if color_by_charge and color_mode == "element":
            color_mode = "charge"
        centroid = centroid_override if centroid_override is not None else (
            molecule.center() if molecule.atoms else np.zeros(3)
        )
        hl = highlighted_atoms or set()
        has_hl = len(hl) > 0

        # Render unit cell wireframe if PBC available
        if pbc is not None:
            corners = pbc.corners()
            # Center the cell on the atom centroid
            cell_center = np.array([0.5, 0.5, 0.5]) @ pbc.basis_matrix
            cell_offset = centroid - cell_center
            corners = corners + cell_offset
            wire_color = (255, 255, 255)
            transformed_corners = []
            for corner in corners:
                pos = rot @ (corner - centroid)
                pos[0] += pan[0]
                pos[1] += pan[1]
                pos[2] += camera_distance
                transformed_corners.append(pos)
            for i, j in pbc.edges():
                self.render_bond(
                    transformed_corners[i], transformed_corners[j],
                    wire_color, wire_color,
                )

        # Snapshot atoms once — molecule may be mutated concurrently by
        # background threads (extend_axis, sort, etc.)
        snap_atoms = list(molecule.atoms)
        n_snap = len(snap_atoms)
        if n_snap > 0:
            positions = np.array([a.position for a in snap_atoms])
            transformed_arr = (positions - centroid) @ rot.T
            transformed_arr[:, 0] += pan[0]
            transformed_arr[:, 1] += pan[1]
            transformed_arr[:, 2] += camera_distance
            transformed = [transformed_arr[i] for i in range(n_snap)]
        else:
            transformed = []

        # Snapshot bonds too — bonds may reference indices that exist in our snapshot
        snap_bonds = list(molecule.bonds) if molecule.bonds else []

        # Ribbon mode: draw backbone trace through CA atoms
        if ribbon and snap_atoms:
            ca_indices = [i for i, a in enumerate(snap_atoms) if a.name.strip() == "CA"]
            if len(ca_indices) < 2:
                if snap_bonds:
                    neighbors: dict[int, set[int]] = {}
                    for a, b in snap_bonds:
                        if a < n_snap and b < n_snap:
                            neighbors.setdefault(a, set()).add(b)
                            neighbors.setdefault(b, set()).add(a)
                    ca_indices = []
                    for i, a in enumerate(snap_atoms):
                        if a.element.symbol != "C":
                            continue
                        nbrs = neighbors.get(i, set())
                        has_n = any(snap_atoms[j].element.symbol == "N" for j in nbrs if j < n_snap)
                        has_c = any(snap_atoms[j].element.symbol == "C" for j in nbrs if j < n_snap)
                        if has_n and has_c:
                            ca_indices.append(i)
            if len(ca_indices) >= 2:
                _chain_colors = [
                    (50, 130, 255), (255, 100, 80), (80, 220, 100),
                    (255, 200, 50), (200, 100, 255), (100, 220, 220),
                    (255, 150, 200), (180, 180, 100),
                ]
                chain_id = 0
                ca_chains: list[int] = [0]
                for k in range(1, len(ca_indices)):
                    d = np.linalg.norm(
                        snap_atoms[ca_indices[k]].position
                        - snap_atoms[ca_indices[k - 1]].position
                    )
                    if d > 5.0:
                        chain_id += 1
                    ca_chains.append(chain_id)
                old_br = self.bond_radius
                self.bond_radius = 0.25
                for k in range(len(ca_indices) - 1):
                    if ca_chains[k] != ca_chains[k + 1]:
                        continue
                    c = _chain_colors[ca_chains[k] % len(_chain_colors)]
                    p0 = transformed[ca_indices[max(k - 1, 0)]]
                    p1 = transformed[ca_indices[k]]
                    p2 = transformed[ca_indices[k + 1]]
                    p3 = transformed[ca_indices[min(k + 2, len(ca_indices) - 1)]]
                    n_seg = 4
                    prev = p1
                    for s in range(1, n_seg + 1):
                        t = s / n_seg
                        t2 = t * t
                        t3 = t2 * t
                        pt = 0.5 * (
                            (2 * p1)
                            + (-p0 + p2) * t
                            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                        )
                        self.render_bond(prev, pt, c, c)
                        prev = pt
                self.bond_radius = old_br
        else:
            # Normal atom/bond rendering (non-ribbon)
            # Use snapshot for atom colors to match transformed length
            if color_mode == "charge":
                atom_colors = [self._charge_color(a.charge) for a in snap_atoms]
            elif color_mode in ("residue", "chain", "index"):
                # Build a temporary molecule-like object from snapshot
                _snap_mol = type("M", (), {"atoms": snap_atoms})()
                atom_colors = self._compute_atom_colors(_snap_mol, color_mode)
            else:
                atom_colors = [a.element.cpk_color for a in snap_atoms]

            if not vdw:
                # Periodic bonds connect atoms on opposite faces of the cell.
                # Draw them as two short stubs (one from each atom toward its
                # partner's nearest image) instead of a line straight across.
                stub_view = None
                is_periodic = None
                if snap_bonds:
                    is_periodic, d_min = classify_periodic_bonds(
                        positions, snap_bonds, pbc
                    )
                    stub_view = (0.5 * d_min) @ rot.T  # half-bond, view space
                for k, (i, j) in enumerate(snap_bonds):
                    if i >= n_snap or j >= n_snap:
                        continue  # skip bonds that reference atoms outside snapshot
                    c1, c2 = atom_colors[i], atom_colors[j]
                    if has_hl and i in hl and j in hl:
                        c1 = self._highlight_color()
                        c2 = self._highlight_color()
                    if is_periodic is not None and is_periodic[k]:
                        self.render_bond(
                            transformed[i], transformed[i] + stub_view[k], c1, c1
                        )
                        self.render_bond(
                            transformed[j], transformed[j] - stub_view[k], c2, c2
                        )
                    else:
                        self.render_bond(transformed[i], transformed[j], c1, c2)

            atom_order = sorted(
                range(n_snap),
                key=lambda idx: -transformed[idx][2],
            )
            for i in atom_order:
                atom = snap_atoms[i]
                if vdw:
                    radius = atom.element.vdw_radius
                elif licorice:
                    radius = self.bond_radius
                else:
                    radius = atom.element.covalent_radius * self.atom_scale
                color = self._highlight_color() if has_hl and i in hl else atom_colors[i]
                self.render_sphere(transformed[i], radius, color)

        # Draw 3D density blobs
        if (density_positions is not None and density_values is not None
                and len(density_positions) > 0):
            for di in range(len(density_positions)):
                pos = rot @ (density_positions[di] - centroid)
                pos[0] += pan[0]
                pos[1] += pan[1]
                pos[2] += camera_distance
                v = float(density_values[di])
                # Color: low density = blue, high = red
                r_c = int(255 * v)
                b_c = int(255 * (1 - v))
                g_c = int(100 * min(v, 1 - v) * 2)
                radius = 0.2 + v * 0.4
                self.render_sphere(pos, radius, (r_c, g_c, b_c))

        # Draw coordination polyhedra around metal atoms
        if show_polyhedra and snap_atoms and transformed:
            _metals = {
                "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V",
                "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr",
                "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
                "Sn", "Cs", "Ba", "La", "Ce", "Hf", "Ta", "W", "Re", "Os",
                "Ir", "Pt", "Au", "Pb", "Bi",
            }
            for ci, atom in enumerate(snap_atoms):
                if atom.element.symbol not in _metals:
                    continue
                cutoff = atom.element.covalent_radius * 2.5 + 0.5
                center_pos = atom.position
                # Collect neighbours at their minimum-image positions so a metal
                # near a face gets a coherent polyhedron instead of edges
                # streaking across the cell to the atoms' real coordinates.
                nbr_view = []
                for ni, natom in enumerate(snap_atoms):
                    if ni == ci:
                        continue
                    diff = natom.position - center_pos
                    if pbc is not None:
                        frac = diff @ pbc.reciprocal_basis_matrix
                        frac -= np.round(frac)
                        diff = frac @ pbc.basis_matrix
                    d = float(np.linalg.norm(diff))
                    if d < cutoff:
                        pt = rot @ (center_pos + diff - centroid)
                        pt[0] += pan[0]
                        pt[1] += pan[1]
                        pt[2] += camera_distance
                        nbr_view.append(pt)
                if len(nbr_view) < 3:
                    continue
                # Draw edges between all neighboring atoms (polyhedron wireframe)
                color = atom.element.cpk_color
                dim = (color[0] // 2, color[1] // 2, color[2] // 2)
                for ai in range(len(nbr_view)):
                    for bj in range(ai + 1, len(nbr_view)):
                        self.render_bond(nbr_view[ai], nbr_view[bj], dim, dim)

        # Draw axis arrows in bottom-left corner (always, including ribbon)
        if pbc is not None:
            self._render_axes(rot, camera_distance, pbc)

    def _render_axes(self, rot: np.ndarray, camera_distance: float,
                     pbc: "PBC") -> None:
        """Draw a/b/c axis arrows in the bottom-left corner."""
        # Normalize basis vectors to equal visual length
        basis = pbc.basis_matrix.copy()
        for i in range(3):
            basis[i] = basis[i] / (np.linalg.norm(basis[i]) + 1e-10)

        # Place the axis origin in screen space: bottom-left
        arrow_len = camera_distance * 0.08
        # Origin in view space (offset to bottom-left)
        ox = -camera_distance * 0.85
        oy = -camera_distance * 0.45
        origin = np.array([ox, oy, camera_distance])

        colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]  # a=red, b=green, c=blue
        for i in range(3):
            direction = rot @ basis[i]
            tip = origin + direction * arrow_len
            self.render_bond(origin, tip, colors[i], colors[i])
            # Small sphere at tip
            self.render_sphere(tip, arrow_len * 0.15, colors[i])


def render_scene(
    width: int,
    height: int,
    molecule: Molecule,
    rot: np.ndarray,
    camera_distance: float,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    pbc: PBC | None = None,
    ssaa: int = 2,
    pan: tuple[float, float] = (0.0, 0.0),
    highlighted_atoms: set[int] | None = None,
    licorice: bool = False,
    vdw: bool = False,
    ambient: float | None = None,
    diffuse: float | None = None,
    specular: float | None = None,
    shininess: float | None = None,
    atom_scale: float | None = None,
    bond_radius: float | None = None,
    color_by_charge: bool = False,
    color_mode: str = "element",
    centroid_override: np.ndarray | None = None,
    ribbon: bool = False,
    show_polyhedra: bool = False,
    density_positions: np.ndarray | None = None,
    density_values: np.ndarray | None = None,
    orthographic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Render with supersampling anti-aliasing.

    Returns (pixels, hit_mask) where pixels is (height, width, 3) uint8
    and hit_mask is (height, width) bool indicating which pixels were drawn.
    """
    r = ImageRenderer(width * ssaa, height * ssaa, bg_color=bg_color)
    if orthographic:
        r.orthographic = True
        # Match the on-screen size of perspective at the chosen camera
        # distance: in perspective, an object at z=camera_distance is
        # scaled by fov/camera_distance; replicate that as ortho_scale.
        r.ortho_scale = r.fov / max(camera_distance, 1e-3)
    if ambient is not None:
        r.ambient = ambient
    if diffuse is not None:
        r.diffuse_strength = diffuse
    if specular is not None:
        r.specular_strength = specular
    if shininess is not None:
        r.shininess = shininess
    if atom_scale is not None:
        r.atom_scale = atom_scale
    if bond_radius is not None:
        r.bond_radius = bond_radius
    r.render_molecule(
        molecule, rot, camera_distance,
        pbc=pbc, pan=pan, highlighted_atoms=highlighted_atoms,
        licorice=licorice, vdw=vdw, color_by_charge=color_by_charge,
        color_mode=color_mode,
        centroid_override=centroid_override,
        ribbon=ribbon, show_polyhedra=show_polyhedra,
        density_positions=density_positions, density_values=density_values,
    )
    hit = np.isfinite(r.z_buf)
    if ssaa == 1:
        return r.pixels, hit
    downsampled = r.pixels.reshape(height, ssaa, width, ssaa, 3).mean(axis=(1, 3)).astype(np.uint8)
    hit_down = hit.reshape(height, ssaa, width, ssaa).any(axis=(1, 3))
    return downsampled, hit_down
