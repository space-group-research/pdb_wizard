"""Mean Squared Displacement (MSD) calculation for diffusion analysis."""

from __future__ import annotations

import numpy as np

from pdb_wizard.molecule import Molecule


def compute_msd(
    frames: list[Molecule],
    element: str | None = None,
    name: str | None = None,
    max_lag: int | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MSD vs time lag for selected atoms across trajectory frames.

    Uses the direct method: MSD(tau) = <|r(t+tau) - r(t)|^2> averaged over
    all atoms and all time origins t.

    Args:
        frames: List of Molecule objects (trajectory).
        element: Filter to this element (None = all moveable atoms).
        name: Filter to this atom name (None = all matching element).
        max_lag: Maximum lag in frames (default: half the trajectory).
        progress_callback: Optional callable(frac).

    Returns:
        (lag_frames, msd_values) arrays.
    """
    if len(frames) < 2:
        return np.array([]), np.array([])

    # Identify which atoms to track — must be present in all frames
    # Use first frame to get indices
    ref = frames[0]
    if element:
        if name and name != "(all)":
            def atom_filter(a):
                return a.element.symbol == element and a.name.strip() == name
        else:
            def atom_filter(a):
                return a.element.symbol == element
    else:
        # Track all non-framework atoms (moveable = charge != 0 or element is H2-like)
        def atom_filter(a):
            return True

    ref_indices = [i for i, a in enumerate(ref.atoms) if atom_filter(a)]
    if not ref_indices:
        return np.array([]), np.array([])

    # Build position arrays: find matching atoms in each frame by name+position proximity
    # For MPMC trajectories, atom count varies. Track by name matching within element.
    n_frames = len(frames)
    if max_lag is None:
        max_lag = n_frames // 2

    # Collect positions per frame for tracked atoms
    # Use index-based tracking (assumes atom ordering is consistent for matching atoms)
    n_track = len(ref_indices)
    positions = np.zeros((n_frames, n_track, 3))
    valid_count = np.zeros(n_frames, dtype=int)

    for fi, mol in enumerate(frames):
        indices = [i for i, a in enumerate(mol.atoms) if atom_filter(a)]
        n_avail = min(len(indices), n_track)
        for ti in range(n_avail):
            positions[fi, ti] = mol.atoms[indices[ti]].x
        valid_count[fi] = n_avail
        if progress_callback and fi % max(1, n_frames // 20) == 0:
            progress_callback(fi / n_frames * 0.5)

    # PBC unwrap: wrapped trajectories produce garbage MSD past first cell
    # crossing. Reconstruct unwrapped positions by accumulating min-imaged
    # frame-to-frame displacements.
    pbc = ref.pbc
    if pbc is not None:
        recip = pbc.reciprocal_basis_matrix
        basis = pbc.basis_matrix
        unwrapped = np.zeros_like(positions)
        unwrapped[0] = positions[0]
        for fi in range(1, n_frames):
            dx = positions[fi] - positions[fi - 1]
            frac = dx @ recip
            frac -= np.round(frac)
            dx_min = frac @ basis
            unwrapped[fi] = unwrapped[fi - 1] + dx_min
        positions = unwrapped

    # Compute MSD for each lag — only use atoms valid in both frames
    min_valid = int(valid_count.min()) if len(valid_count) > 0 else 0
    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(len(lags))

    for li, lag in enumerate(lags):
        displacements = positions[lag:, :min_valid] - positions[:n_frames - lag, :min_valid]
        sq_disp = np.sum(displacements ** 2, axis=2)
        msd[li] = sq_disp.mean() if sq_disp.size > 0 else 0.0
        if progress_callback and li % max(1, len(lags) // 20) == 0:
            progress_callback(0.5 + li / len(lags) * 0.5)

    if progress_callback:
        progress_callback(1.0)

    return lags.astype(float), msd


def compute_diffusion_constant(
    lags: np.ndarray,
    msd: np.ndarray,
    dt: float = 1.0,
    fit_start: float = 0.25,
    fit_end: float = 0.75,
) -> tuple[float, float]:
    """Compute diffusion constant from MSD via Einstein relation.

    D = slope / (2 * d) where d=3 for 3D diffusion, so D = slope / 6.
    slope is d(MSD)/d(t) in A^2/ps.

    Args:
        lags: Lag values in frames.
        msd: MSD values in A^2.
        dt: Time between frames in picoseconds.
        fit_start: Fraction of lag range to start fitting (skip ballistic regime).
        fit_end: Fraction of lag range to end fitting (skip noisy tail).

    Returns:
        (D_cm2_s, slope_A2_ps): Diffusion constant in cm^2/s and slope in A^2/ps.
    """
    if len(lags) < 4:
        return 0.0, 0.0
    n = len(lags)
    i_start = max(1, int(n * fit_start))
    i_end = max(i_start + 2, int(n * fit_end))
    t = lags[i_start:i_end] * dt  # time in ps
    m = msd[i_start:i_end]
    slope, _ = np.polyfit(t, m, 1)  # A^2/ps
    D_A2_ps = slope / 6.0
    D_cm2_s = D_A2_ps * 1e-16 / 1e-12  # 1 A^2 = 1e-16 cm^2, 1 ps = 1e-12 s
    return D_cm2_s, slope
