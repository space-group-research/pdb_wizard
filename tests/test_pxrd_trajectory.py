"""PXRD averaged over the frames of a trajectory.

A diffractometer records the time average of the scattered INTENSITY. So the pattern of a
trajectory is: each frame's pattern with that frame's own cell, intensities averaged on the
common 2-theta grid, and only then scaled so the highest point is 100. Averaging amplitudes,
or taking the pattern of the frame-averaged structure, would lose the Debye-Waller damping
that thermal motion gives high-angle peaks.

Also checks that a single structure gives exactly what it gave before, and that the GUI
dialog, the terminal viewer and the classic menu give the same curve and summary as
``run_pxrd`` for the same trajectory and options. The GUI parts skip without PySide6 and
pytest-qt, the terminal-viewer parts without textual.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import pdb_wizard.menu_tools as mt
import pdb_wizard.pxrd as pxrd
from pdb_wizard.analyses import run_pxrd
from pdb_wizard.atom import Atom
from pdb_wizard.io import fallback_cell
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.pxrd import compute_pxrd, pxrd_reflections

CU_KA = 1.5406


def _importable(name: str) -> bool:
    """A real module (a bare namespace package left by a partial install does not count)."""
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:  # the parent package is missing
        return False


HAVE_QT = _importable("PySide6.QtWidgets") and _importable("pytestqt.qtbot")
HAVE_TEXTUAL = _importable("textual.app")
needs_qt = pytest.mark.skipif(not HAVE_QT, reason="needs PySide6 and pytest-qt")
needs_textual = pytest.mark.skipif(not HAVE_TEXTUAL, reason="needs textual")


# ---- structures -------------------------------------------------------------------------


TRICLINIC = (9.1, 10.7, 12.3, 77.0, 84.0, 101.0)
FRACS = np.random.default_rng(1).random((12, 3))
ELEMENTS = ["C", "O", "N", "Cu"] * 3


def crystal(scale: float = 1.0, shift=(0.0, 0.0, 0.0), noise=None) -> Molecule:
    """A low-symmetry crystal (triclinic cell, mixed elements); ``scale`` stretches the
    cell, ``shift`` translates every atom (fractional), ``noise`` adds Cartesian offsets."""
    a, b, c, al, be, ga = TRICLINIC
    pbc = PBC(a * scale, b * scale, c * scale, al, be, ga)
    xyz = (FRACS + np.asarray(shift)) @ pbc.basis_matrix
    if noise is not None:
        xyz = xyz + noise
    return Molecule(atoms=[Atom(*x, el) for x, el in zip(xyz, ELEMENTS)], pbc=pbc)


def trajectory(n_frames: int = 6, seed: int = 3) -> list:
    """Frames with thermal noise and a cell that breathes by up to 1 %."""
    rng = np.random.default_rng(seed)
    return [crystal(1.0 + 0.002 * f, noise=rng.normal(0, 0.08, (12, 3)))
            for f in range(n_frames)]


def fcc(a: float, n: int, element: str = "Pb", disp=None) -> Molecule:
    """An n x n x n supercell of an fcc crystal; ``disp`` are per-atom displacements."""
    basis = np.array([(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)])
    cells = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)])
    xyz = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * a
    if disp is not None:
        xyz = xyz + disp
    side = a * n
    return Molecule(atoms=[Atom(*x, element) for x in xyz],
                    pbc=PBC(side, side, side, 90.0, 90.0, 90.0))


def normalised_mean(frames, **kw):
    """Independent reference: mean of the frames' un-normalised patterns, max scaled to 100."""
    raws = [pxrd.pxrd_raw_pattern(f, **kw) for f in frames]
    x = raws[0][0]
    mean = np.mean([r[1] for r in raws], axis=0)
    return x, mean / mean.max() * 100


def old_compute_pxrd(mol, wavelength=CU_KA, two_theta_min=5.0, two_theta_max=50.0,
                     n_points=2000, peak_width=0.1):
    """compute_pxrd as it was before trajectories (the reference for 'unchanged')."""
    tt_all, inten = pxrd_reflections(mol, wavelength, two_theta_min, two_theta_max)
    reflections = list(zip(tt_all.tolist(), inten.tolist()))
    if not reflections:
        return np.array([]), np.array([])
    x = np.linspace(two_theta_min, two_theta_max, n_points)
    y = np.zeros(n_points)
    sigma = peak_width / (2 * np.sqrt(2 * np.log(2)))
    for tt, i in reflections:
        y += i * np.exp(-0.5 * ((x - tt) / sigma) ** 2)
    return x, y / y.max() * 100


def peak_near(x, y, target, half_window=0.3) -> int:
    """Index of the highest point within ``half_window`` degrees of ``target``."""
    idx = np.flatnonzero(np.abs(x - target) <= half_window)
    return int(idx[np.argmax(y[idx])])


def is_local_max(y, i) -> bool:
    return y[i] >= y[i - 1] and y[i] >= y[i + 1]


# ---- a single structure is unchanged ----------------------------------------------------


@pytest.mark.parametrize("wl, tt_max, width", [(CU_KA, 50.0, 0.10), (0.7107, 30.0, 0.05),
                                               (1.7903, 90.0, 0.5)])
def test_single_structure_is_bit_for_bit_unchanged(wl, tt_max, width):
    mol = crystal()
    x, y = compute_pxrd(mol, wavelength=wl, two_theta_max=tt_max, peak_width=width)
    ox, oy = old_compute_pxrd(mol, wl, two_theta_max=tt_max, peak_width=width)
    np.testing.assert_array_equal(x, ox)
    np.testing.assert_array_equal(y, oy)
    single = run_pxrd(mol, wl, tt_max, width)
    np.testing.assert_array_equal(single.y, y)
    frames = trajectory(3)
    for res in (run_pxrd(mol, wl, tt_max, width, frames=frames),           # use_traj off
                run_pxrd(mol, wl, tt_max, width, frames=frames, use_traj=False),
                run_pxrd(mol, wl, tt_max, width, frames=[mol], use_traj=True)):
        np.testing.assert_array_equal(res.x, single.x)
        np.testing.assert_array_equal(res.y, single.y)
        assert (res.title, res.summary) == (single.title, single.summary)
    assert single.title == ("PXRD (Cu Ka)" if wl == CU_KA else f"PXRD ({wl:.4f} A)")


def test_raw_pattern_is_compute_pxrd_before_scaling():
    mol = crystal()
    x, raw = pxrd.pxrd_raw_pattern(mol)
    cx, cy = compute_pxrd(mol)
    np.testing.assert_array_equal(x, cx)
    np.testing.assert_array_equal(raw / raw.max() * 100, cy)
    empty = pxrd.pxrd_raw_pattern(Molecule(atoms=[], pbc=PBC(5, 5, 5, 90, 90, 90)))
    assert len(empty[0]) == 0 and len(empty[1]) == 0
    tiny = simple_cubic(1.0)  # first reflection at 2-theta = 100 deg, beyond the range
    assert [len(a) for a in pxrd.pxrd_raw_pattern(tiny)] == [0, 0]
    assert [len(a) for a in compute_pxrd(tiny)] == [0, 0]


# ---- physics ----------------------------------------------------------------------------


def test_identical_frames_give_the_single_frame_pattern():
    mol = crystal()
    single = run_pxrd(mol)
    for n in (2, 3, 5):
        avg = run_pxrd(mol, frames=[crystal() for _ in range(n)], use_traj=True)
        np.testing.assert_array_equal(avg.x, single.x)
        np.testing.assert_allclose(avg.y, single.y, rtol=1e-12, atol=1e-12)
        assert avg.summary == single.summary
        assert avg.title == f"PXRD (Cu Ka, frames 1-{n})"


def test_a_rigid_translation_changes_nothing_so_intensities_not_amplitudes_are_averaged():
    """Frame 2 is frame 1 moved by half a cell along a. Intensities do not see the origin;
    averaged amplitudes would carry (1 + e^{i pi h}) / 2 and cancel every odd-h peak."""
    mol = crystal()
    moved = crystal(shift=(0.5, 0.0, 0.0))
    single = run_pxrd(mol)
    avg = run_pxrd(mol, frames=[mol, moved], use_traj=True)
    np.testing.assert_allclose(avg.y, single.y, rtol=1e-9, atol=1e-9)
    # what an amplitude average would give: the even-h reflections only
    tt, inten, hkl = pxrd_reflections(mol, CU_KA, 5.0, 50.0, with_hkl=True)
    sigma = 0.10 / (2 * np.sqrt(2 * np.log(2)))
    even = hkl[:, 0] % 2 == 0
    amp = sum(i * np.exp(-0.5 * ((avg.x - t) / sigma) ** 2) for t, i in zip(tt[even], inten[even]))
    amp = amp / amp.max() * 100
    assert np.abs(avg.y - amp).max() > 20


@pytest.fixture(scope="module")
def thermal_fcc():
    """Perfect fcc Pb (2x2x2 supercell) and 40 frames of Gaussian thermal displacements."""
    a, n, sigma, n_frames = 4.95, 2, 0.12, 40
    rng = np.random.default_rng(11)
    perfect = fcc(a, n)
    n_atoms = len(perfect.atoms)
    disps = [rng.normal(0, sigma, (n_atoms, 3)) for _ in range(n_frames)]
    frames = [fcc(a, n, disp=d) for d in disps]
    mean_structure = fcc(a, n, disp=np.mean(disps, axis=0))
    opts = dict(wavelength=CU_KA, two_theta_max=90.0, peak_width=0.10)
    return dict(
        a=a, sigma=sigma, n_frames=n_frames,
        perfect=run_pxrd(perfect, **opts),
        thermal=run_pxrd(frames[0], frames=frames, use_traj=True, **opts),
        of_mean_structure=run_pxrd(mean_structure, **opts),
    )


FCC_PEAKS = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (3, 3, 1),
             (4, 2, 0)]


def _relative_heights(perfect, other, a):
    """(sin(theta)/lambda)^2 of each fcc peak and other/perfect at it, relative to (111)."""
    s2, ratio = [], []
    step = perfect.x[1] - perfect.x[0]
    for hkl in FCC_PEAKS:
        s = np.sqrt(sum(h * h for h in hkl)) / (2 * a)  # 1 / (2 d)
        bragg = 2 * np.degrees(np.arcsin(CU_KA * s))
        i = peak_near(perfect.x, perfect.y, bragg)
        assert abs(perfect.x[i] - bragg) <= step, hkl          # the peak is where Bragg says
        j = peak_near(other.x, other.y, bragg)
        assert abs(j - i) <= 1 and is_local_max(other.y, j), hkl  # and stays there
        s2.append(s * s)
        ratio.append(other.y[i] / perfect.y[i])
    s2, ratio = np.array(s2), np.array(ratio)
    return s2, ratio / ratio[0]


def test_thermal_motion_gives_debye_waller_damping(thermal_fcc):
    """Each atom displaced by a Gaussian of rms sigma per axis damps the intensity by
    exp(-B s^2), s = sin(theta)/lambda, with B = 16 pi^2 sigma^2 (twice the crystallographic
    B = 8 pi^2 <u_x^2>, which multiplies the amplitude)."""
    t = thermal_fcc
    s2, rel = _relative_heights(t["perfect"], t["thermal"], t["a"])
    B = 16 * np.pi ** 2 * t["sigma"] ** 2
    expected = np.exp(-B * (s2 - s2[0]))
    assert expected[-1] < 0.7                       # a strong effect at the high-angle end
    np.testing.assert_allclose(rel, expected, rtol=0.08)
    assert rel[-1] < 0.8 * rel[1]                   # high-angle peaks drop relative to low
    assert t["thermal"].y.max() == 100.0


def test_averaging_intensities_differs_from_the_pattern_of_the_averaged_structure(thermal_fcc):
    """The frame-averaged structure is nearly the perfect crystal: no damping. The test
    above would fail for it, so it tells the two methods apart."""
    t = thermal_fcc
    s2, rel_mean = _relative_heights(t["perfect"], t["of_mean_structure"], t["a"])
    _, rel_avg = _relative_heights(t["perfect"], t["thermal"], t["a"])
    B = 16 * np.pi ** 2 * t["sigma"] ** 2
    expected = np.exp(-B * (s2 - s2[0]))
    assert rel_mean[-1] > 0.95 and abs(rel_mean[-1] - expected[-1]) > 0.25
    assert abs(rel_avg[-1] - expected[-1]) < 0.08 * expected[-1]
    assert not np.allclose(t["thermal"].y, t["of_mean_structure"].y, atol=1.0)


def simple_cubic(a: float) -> Molecule:
    return Molecule(atoms=[Atom(0.0, 0.0, 0.0, "Cu")], pbc=PBC(a, a, a, 90.0, 90.0, 90.0))


def test_frames_with_different_cells_show_both_peak_positions():
    small, large = simple_cubic(4.0), simple_cubic(4.2)
    res = run_pxrd(small, frames=[small, large], use_traj=True)
    x, ref = normalised_mean([small, large])
    np.testing.assert_allclose(res.y, ref, rtol=1e-12, atol=1e-12)
    step = res.x[1] - res.x[0]
    for a in (4.0, 4.2):
        for hkl in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
            bragg = 2 * np.degrees(np.arcsin(CU_KA * np.sqrt(sum(h * h for h in hkl)) / (2 * a)))
            i = peak_near(res.x, res.y, bragg, 0.1)
            assert abs(res.x[i] - bragg) <= step and is_local_max(res.y, i), (a, hkl)
            assert res.y[i] > 20, (a, hkl)       # each frame's peak carries real weight


def test_peak_width_applies_to_the_averaged_pattern():
    """The FWHM of an isolated peak is the Peak width option, averaged or not."""
    small, large = simple_cubic(4.0), simple_cubic(4.2)
    for width in (0.10, 0.30):
        res = run_pxrd(small, CU_KA, 50.0, width, frames=[small, large], use_traj=True)
        same = run_pxrd(small, CU_KA, 50.0, width, frames=[small, simple_cubic(4.0)],
                        use_traj=True)
        np.testing.assert_allclose(same.y, run_pxrd(small, CU_KA, 50.0, width).y,
                                   rtol=1e-12, atol=1e-12)
        step = res.x[1] - res.x[0]
        for a in (4.0, 4.2):
            bragg = 2 * np.degrees(np.arcsin(CU_KA / (2 * a)))  # (100)
            i = peak_near(res.x, res.y, bragg, 0.5)
            lo = i
            while res.y[lo] > res.y[i] / 2:
                lo -= 1
            hi = i
            while res.y[hi] > res.y[i] / 2:
                hi += 1
            assert abs((hi - lo - 1) * step - width) <= 2 * step, (width, a)


# ---- normalisation, range, cells, progress ----------------------------------------------


def test_highest_point_is_exactly_100():
    frames = trajectory()
    assert run_pxrd(frames[0]).y.max() == 100.0
    assert run_pxrd(frames[0], frames=frames, use_traj=True).y.max() == 100.0
    assert run_pxrd(frames[0], frames=frames, start=2, end=4, use_traj=True).y.max() == 100.0


def test_frame_range_averages_only_the_chosen_frames():
    frames = trajectory()
    sub = run_pxrd(frames[0], frames=frames, start=2, end=4, use_traj=True)
    _, ref = normalised_mean(frames[1:4])
    np.testing.assert_allclose(sub.y, ref, rtol=1e-12, atol=1e-12)
    assert sub.title == "PXRD (Cu Ka, frames 2-4)"
    whole = run_pxrd(frames[0], frames=frames, use_traj=True)
    _, ref_all = normalised_mean(frames)
    np.testing.assert_allclose(whole.y, ref_all, rtol=1e-12, atol=1e-12)
    assert whole.title == "PXRD (Cu Ka, frames 1-6)"
    assert not np.allclose(whole.y, sub.y, atol=1e-3)
    tail = run_pxrd(frames[0], frames=frames, start=5, end=99, use_traj=True)
    assert tail.title == "PXRD (Cu Ka, frames 5-6)"


def test_an_empty_frame_range_is_explained():
    frames = trajectory()
    res = run_pxrd(frames[0], frames=frames, start=5, end=3, use_traj=True)
    assert res.empty
    assert res.summary == "No frames between 5 and 3 (the trajectory has 6)"


def test_frames_without_a_cell_are_left_out_and_none_is_refused():
    frames = trajectory(4)
    frames[1].pbc = fallback_cell()   # what a reader gives a frame with no cell
    frames[3].pbc = None
    res = run_pxrd(frames[0], frames=frames, use_traj=True)
    _, ref = normalised_mean([frames[0], frames[2]])
    np.testing.assert_allclose(res.y, ref, rtol=1e-12, atol=1e-12)
    assert "2 of 4 frames have no unit cell and were left out" in res.summary
    assert res.title == "PXRD (Cu Ka, frames 1-4)"
    for f in frames:
        f.pbc = fallback_cell()
    refused = run_pxrd(frames[0], frames=frames, start=2, end=3, use_traj=True)
    assert refused.empty
    assert refused.summary == "No unit cell in frames 2-3: PXRD needs a crystal structure"


def test_trajectory_summary_notes_and_frames_without_reflections():
    """The form-factor note covers every frame used; cells without atoms give nothing."""
    cu = simple_cubic(4.0)
    with_ar = simple_cubic(4.0)
    with_ar.atoms.append(Atom(2.0, 2.0, 2.0, "Ar"))  # no tabulated form factor
    res = run_pxrd(cu, frames=[cu, with_ar], use_traj=True)
    assert "no tabulated form factor for Ar" in res.summary
    assert "form factor" not in run_pxrd(cu).summary
    _, ref = normalised_mean([cu, with_ar])
    np.testing.assert_allclose(res.y, ref, rtol=1e-12, atol=1e-12)
    empty = [Molecule(atoms=[], pbc=PBC(5, 5, 5, 90, 90, 90)) for _ in range(2)]
    assert run_pxrd(cu, frames=empty, use_traj=True).summary == "No reflections found"


def test_progress_runs_across_the_frames():
    frames = trajectory(5)
    seen: list = []
    run_pxrd(frames[0], progress=seen.append, frames=frames, use_traj=True)
    assert len(seen) >= 5 and seen == sorted(seen)
    assert seen[0] <= 0.2 + 1e-12 and seen[-1] == 1.0


# ---- the three front ends give run_pxrd's numbers ---------------------------------------


def drive(func, responses, *args):
    """Run ``func`` with scripted input(); returns (result, prompts, printed text)."""
    prompts: list = []
    printed: list = []
    it = iter(responses)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"menu asked for more input than scripted: {prompt!r}")

    def fake_print(*a, **k):
        printed.append(" ".join(str(x) for x in a))

    with mock.patch("builtins.input", fake_input), mock.patch("builtins.print", fake_print):
        result = func(*args)
    assert next(it, None) is None, "scripted input left over"
    return result, prompts, "\n".join(printed)


PXRD_OPTION = str([k for k, _ in mt.CM_ANALYSES].index("pxrd") + 1)
BACK = str(len(mt.CM_ANALYSES) + 1)
PXRD_PROMPTS = ["Wavelength (A; Cu Ka 1.5406, Mo Ka 0.7107, Co Ka 1.7903) [1.5406] > ",
                "Maximum 2-theta (deg) [50.0] > ", "Peak width (deg) [0.1] > "]
CSV_PROMPT = "\nCSV filename (blank to skip) > "


def test_classic_menu_single_structure_prompts_are_unchanged(tmp_path):
    mol = crystal()
    csv = tmp_path / "one.csv"
    for frames in (None, [mol]):
        _, prompts, out = drive(mt.cm_menu_analysis, [PXRD_OPTION, "", "", "", str(csv), BACK],
                                mol, frames, "t.pdb")
        assert prompts[1:-1] == PXRD_PROMPTS + [CSV_PROMPT]
        ref = run_pxrd(mol)
        assert ref.title in out and ref.summary in out
        np.testing.assert_allclose(np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1], ref.y,
                                   atol=1e-6)


def test_classic_menu_trajectory_matches_run_pxrd(tmp_path):
    frames = trajectory()
    csv = tmp_path / "avg.csv"
    _, prompts, out = drive(mt.cm_menu_analysis,
                            [PXRD_OPTION, "", "", "0.2", "", "2", "5", str(csv), BACK],
                            frames[0], frames, "t.xyz")
    assert prompts[1:-1] == PXRD_PROMPTS + ["Average over the trajectory? (y/n) [y] > ",
                                            "First frame [1] > ", "Last frame [6] > ",
                                            CSV_PROMPT]
    ref = run_pxrd(frames[0], CU_KA, 50.0, 0.2, frames=frames, start=2, end=5, use_traj=True)
    assert "PXRD (Cu Ka, frames 2-5)" in out and ref.summary in out
    np.testing.assert_allclose(np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1], ref.y,
                               atol=1e-6)
    # "n" keeps the current frame, with no range questions
    _, prompts, out = drive(mt.cm_menu_analysis, [PXRD_OPTION, "", "", "", "n", "", BACK],
                            frames[0], frames, "t.xyz")
    assert prompts[4] == "Average over the trajectory? (y/n) [y] > "
    assert prompts[5] == CSV_PROMPT
    single = run_pxrd(frames[0])
    assert "\nPXRD (Cu Ka)\n" in out and single.summary in out


def pdb_trajectory(path: Path, frames: list) -> str:
    """A multi-MODEL PDB with a CRYST1 record (its own cell) in every model."""
    lines = []
    for n, mol in enumerate(frames, 1):
        p = mol.pbc
        lines += [f"MODEL     {n:4d}",
                  f"CRYST1{p.a:9.3f}{p.b:9.3f}{p.c:9.3f}{p.alpha:7.2f}{p.beta:7.2f}"
                  f"{p.gamma:7.2f} P 1           1"]
        for i, at in enumerate(mol.atoms, 1):
            el = at.element.symbol
            lines.append(f"HETATM{i:5d} {el:<4} MOF A   1    {at.x[0]:8.3f}{at.x[1]:8.3f}"
                         f"{at.x[2]:8.3f}  1.00  0.00          {el:>2}  ")
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


async def _tui_pxrd(path, setup=None, exp=None):
    """Open the file as ``pdb_wizard file`` does, run Analysis > PXRD and Compute."""
    from textual.widgets import Checkbox, Label

    from pdb_wizard import viewer as V
    from pdb_wizard.panels import OperationsPanel

    app = V.PdbWizardApp(molecule=Molecule(atoms=[]), filepath=str(path), load_trajectory=True)
    async with app.run_test(size=(220, 60)) as pilot:
        for _ in range(600):
            await pilot.pause(0.05)
            if (app._frames and len(app._frames) > 1 and app.molecule.atoms
                    and all(t.done() for t in list(app._bg_tasks))):
                break
        await pilot.pause()
        mol, frames = app.molecule, app._frames
        app.on_operations_panel_run_command(OperationsPanel.RunCommand("pxrd"))
        for _ in range(200):
            await pilot.pause(0.05)
            if isinstance(app.screen, V.PxrdScreen) and app.screen.is_mounted:
                break
        screen = app.screen
        await pilot.pause()
        assert screen._frames is frames
        defaults = (screen.query_one("#pxrd-traj", Checkbox).value,
                    screen.query_one("#pxrd-start", V.SpinBox).value,
                    screen.query_one("#pxrd-end", V.SpinBox).value)
        if setup is not None:
            setup(screen, V)
            await pilot.pause()
        screen.query_one("#pxrd-compute").press()
        for _ in range(1200):
            await asyncio.sleep(0.05)
            task = screen._compute_task
            if task is not None and task.done():
                break
        await pilot.pause()
        task.result()
        plot = screen.query_one("#pxrd-plot", V.PlotWidget)
        status = str(screen.query_one("#pxrd-status", Label).render())
        out = (plot._x.copy(), plot._y.copy(), plot._title, status)
        overlay = None
        if exp is not None:
            screen._load_experimental(str(exp))
            await pilot.pause()
            overlay = (plot._overlay_y, plot._y.copy(),
                       str(screen.query_one("#pxrd-status", Label).render()))
        return mol, frames, defaults, out, overlay


def assert_matches(out, ref):
    x, y, title, status = out
    assert len(x) > 0, status
    np.testing.assert_array_equal(x, ref.x)
    np.testing.assert_array_equal(y, ref.y)
    assert title == ref.title
    assert status == ref.summary


@needs_textual
def test_terminal_viewer_trajectory_matches_run_pxrd(tmp_path):
    path = pdb_trajectory(tmp_path / "traj.pdb", trajectory())
    exp = tmp_path / "exp.xy"
    exp.write_text("5 10\n20 100\n45 20\n")

    def setup(screen, V):
        screen.query_one("#pxrd-start", V.SpinBox)._update(2)
        screen.query_one("#pxrd-end", V.SpinBox)._update(5)
        screen.query_one("#pxrd-width", V.SpinBox)._update(20)

    mol, frames, defaults, out, overlay = asyncio.run(_tui_pxrd(path, setup, exp))
    assert defaults == (True, 1, 6)
    ref = run_pxrd(mol, CU_KA, 50.0, 0.2, frames=frames, start=2, end=5, use_traj=True)
    assert ref.title == "PXRD (Cu Ka, frames 2-5)"
    assert_matches(out, ref)
    exp_y, sim_y, status = overlay  # the experimental overlay sits on the averaged pattern
    assert exp_y is not None and exp_y.max() == pytest.approx(100.0)
    np.testing.assert_array_equal(sim_y, ref.y)
    assert status.startswith("Experimental data loaded (3 points)")


@needs_textual
def test_terminal_viewer_trajectory_box_off_gives_the_current_frame(tmp_path):
    path = pdb_trajectory(tmp_path / "traj.pdb", trajectory())

    def setup(screen, V):
        from textual.widgets import Checkbox
        screen.query_one("#pxrd-traj", Checkbox).value = False

    mol, frames, _, out, _ = asyncio.run(_tui_pxrd(path, setup))
    assert_matches(out, run_pxrd(mol))


@needs_textual
def test_terminal_viewer_single_structure_has_no_trajectory_controls():
    from textual.widgets import Checkbox

    from pdb_wizard import viewer as V

    mol = crystal()

    async def run():
        app = V.PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.pause()
            screen = V.PxrdScreen(mol)
            app.push_screen(screen)
            await pilot.pause()
            return [len(screen.query(sel)) for sel in ("#pxrd-traj", "#pxrd-start",
                                                       "#pxrd-end")] + [
                len(screen.query(Checkbox))]

    assert asyncio.run(run()) == [0, 0, 0, 0]


@needs_qt
def test_gui_dialog_trajectory_matches_run_pxrd(qtbot, tmp_path):
    from pdb_wizard.gui_tools import AnalysisDialog

    frames = trajectory()
    dlg = AnalysisDialog("pxrd", frames[0], frames)
    qtbot.addWidget(dlg)
    assert dlg.traj_box is not None and dlg.traj_box.isChecked()   # on for a trajectory
    assert (dlg.start_spin.value(), dlg.end_spin.value()) == (1, 6)
    dlg.run().wait()
    ref = run_pxrd(frames[0], CU_KA, 50.0, 0.10, frames=frames, use_traj=True)
    assert_matches((dlg.plot.x, dlg.plot.y, dlg.plot.title, dlg.status.text()), ref)
    exp = tmp_path / "exp.xy"
    exp.write_text("5 10\n20 100\n45 20\n")
    assert dlg.load_experimental(str(exp))
    dlg.start_spin.setValue(2)
    dlg.end_spin.setValue(5)
    dlg.inputs["width"].setValue(20)
    dlg.run().wait()
    ref = run_pxrd(frames[0], CU_KA, 50.0, 0.2, frames=frames, start=2, end=5, use_traj=True)
    assert_matches((dlg.plot.x, dlg.plot.y, dlg.plot.title, dlg.status.text()), ref)
    assert dlg.plot.overlay is not None   # the experimental pattern survives the recompute
    dlg.traj_box.setChecked(False)
    dlg.run().wait()
    assert_matches((dlg.plot.x, dlg.plot.y, dlg.plot.title, dlg.status.text()),
                   run_pxrd(frames[0], CU_KA, 50.0, 0.2))


@needs_qt
def test_gui_dialog_single_structure_has_no_trajectory_controls(qtbot):
    from pdb_wizard.gui_tools import AnalysisDialog

    mol = crystal()
    for frames in (None, [mol]):
        dlg = AnalysisDialog("pxrd", mol, frames)
        qtbot.addWidget(dlg)
        assert dlg.traj_box is None and dlg.start_spin is None
        dlg.run().wait()
        assert_matches((dlg.plot.x, dlg.plot.y, dlg.plot.title, dlg.status.text()),
                       run_pxrd(mol))
