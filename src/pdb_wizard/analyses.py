"""Analysis runs shared by the terminal viewer and the desktop GUI.

Each ``run_*`` function takes plain parameters (elements, atom names, frame range, ...),
calls the numerical routine and returns an :class:`AnalysisResult` with the curve, its
labels, the one-line summary the viewers show and the CSV header. Keeping this out of the
UI code means both front ends compute and report exactly the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from pdb_wizard.checks import mc_trajectory_warning, structure_check_text
from pdb_wizard.coordination import compute_coordination, compute_coordination_trajectory
from pdb_wizard.crystal import symmetry_report
from pdb_wizard.density3d import compute_density_3d
from pdb_wizard.density_profile import compute_density_profile, compute_density_profile_trajectory
from pdb_wizard.gyration import compute_gyration_trajectory
from pdb_wizard.hbonds import count_hbonds_trajectory, detect_hbonds
from pdb_wizard.io import is_fallback_cell
from pdb_wizard.isotherm import isotherm_log_table, parse_isotherm_results
from pdb_wizard.molecule import Molecule
from pdb_wizard.msd import compute_diffusion_constant, compute_msd
from pdb_wizard.operations import (
    formula_unit,
    formula_unit_reduced,
    pore_size_distribution,
    surface_area,
    system_info,
    void_volume,
)
from pdb_wizard.pxrd import compute_pxrd, pxrd_untabulated_elements
from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory
from pdb_wizard.rmsd import compute_rmsd

ALL_ATOMS = "(all)"
PXRD_SOURCES = (("Cu Ka 1.5406", 1.5406), ("Mo Ka 0.7107", 0.7107), ("Co Ka 1.7903", 1.7903))


@dataclass
class AnalysisResult:
    """One analysis outcome. ``x``/``y`` are empty when there is nothing to plot."""

    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    title: str = ""
    x_label: str = ""
    summary: str = ""
    csv_header: str = ""
    text: str = ""
    y_min: Optional[float] = 0.0
    y_max: Optional[float] = None
    extra: Dict = field(default_factory=dict)
    csv_data: Optional[tuple] = None  # columns for CSV when they differ from (x, y)

    @property
    def empty(self) -> bool:
        return len(self.x) == 0

    def csv_arrays(self) -> tuple:
        return self.csv_data if self.csv_data is not None else (self.x, self.y)


def _opt(value: Optional[str]) -> Optional[str]:
    """'(all)' and '' mean no filter."""
    return None if value in (None, "", ALL_ATOMS) else value


def _grow(values: np.ndarray) -> float:
    return float(values.max()) * 1.1 if len(values) and values.max() > 0 else 1.0


def frame_slice(frames: Optional[list], start: int = 1,
                end: Optional[int] = None) -> Optional[list]:
    """Frames ``start``..``end`` (1-based, inclusive), as the viewers' Start/End boxes mean."""
    if not frames or len(frames) < 2:
        return frames
    return frames[max(start, 1) - 1:len(frames) if end is None else end]


def elements_of(mol: Molecule) -> List[str]:
    return sorted({a.element.symbol for a in mol.atoms})


def names_of(mol: Molecule, element: Optional[str] = None) -> List[str]:
    """Atom names, optionally only those of one element (for the Name filters)."""
    el = _opt(element)
    return sorted({a.name.strip() for a in mol.atoms if el is None or a.element.symbol == el})


def unavailable(key: str, mol: Optional[Molecule], frames: Optional[list]) -> Optional[str]:
    """Why an analysis cannot run on this structure (None if it can).

    The terminal viewer's Analysis-menu rules and messages, plus one more: void volume
    needs a cell (the terminal viewer reports zeros without one).
    """
    n_frames = len(frames) if frames else 0
    # A trajectory without cell data carries a huge stand-in box; it is not a cell.
    no_cell = mol is None or mol.pbc is None or is_fallback_cell(mol.pbc)
    if key in ("energy_plot", "isotherm_plot", "db_search"):
        return None
    if key in ("msd", "rmsd") and n_frames < 3:
        return f"Need a trajectory with 3+ frames for {key.upper()}"
    if key == "gyration" and n_frames < 2:
        return "Need a trajectory for Rg"
    if key == "density3d":
        if n_frames < 2:
            return "Need a trajectory with PBC for 3D density"
        if no_cell:
            return "Need PBC for 3D density map"
    if key == "pxrd" and (mol is None or not mol.atoms or no_cell):
        return "Need a crystal structure with PBC for PXRD"
    if mol is None:
        return "No atoms loaded"
    if key in ("info", "input_generator", "check", "symmetry"):
        return None
    if not mol.atoms:
        return "No atoms loaded yet" if key == "rdf" else "No atoms loaded"
    if no_cell:
        need = {"surface_area": "Need PBC for surface area", "pore_size": "Need PBC for pore size",
                "void_vol": "Need PBC for void volume"}
        if key in need:
            return need[key]
    return None


# ----------------------------------------------------------------------
# Single-structure reports
# ----------------------------------------------------------------------

def system_info_text(mol: Molecule, filepath: str = "") -> str:
    info = system_info(mol, filepath)
    lines = [f"Filename:  {info.get('filename', '')}", f"Atoms:     {info.get('n_atoms', 0)}"]
    if "volume" in info:
        c = info.get("cell", {})
        lines.append(f"Cell:      {c.get('a', 0):.4f}  {c.get('b', 0):.4f}  {c.get('c', 0):.4f}")
        lines.append(
            f"Angles:    {c.get('alpha', 0):.2f}  {c.get('beta', 0):.2f}  {c.get('gamma', 0):.2f}")
        lines.append(f"Volume:    {info.get('volume', 0):.2f} A^3")
        lines.append(f"Density:   {info.get('density', 0):.4f} g/cm^3")
        lines.append("")
        for row in info.get("basis_matrix", []):
            lines.append(f"  {row[0]:>14.8f} {row[1]:>14.8f} {row[2]:>14.8f}")
    lines.append("")
    lines.append("Total:     " + "  ".join(f"{k}{v}" for k, v in formula_unit(mol).items()))
    lines.append("Formula:   " + "  ".join(f"{k}{v}" for k, v in formula_unit_reduced(mol).items()))
    return "\n".join(lines)


def void_volume_report(geo: Dict, he: Dict, n_samples: int = 100000,
                       probe: float = 1.14) -> str:
    """The Void Volume report from the geometric and He-probe results."""
    lines = [f"Void Volume Analysis ({n_samples:,} MC samples)", ""]
    for label, r in (("Geometric (no probe):", geo), (f"He probe (r = {probe:g} A):", he)):
        lines += [label,
                  f"  Void volume:   {r.get('void_volume', 0):.2f} A^3",
                  f"  Total volume:  {r.get('total_volume', 0):.2f} A^3",
                  f"  Void fraction: {r.get('void_fraction', 0) * 100:.1f}%", ""]
    return "\n".join(lines[:-1])


def void_volume_text(mol: Molecule, n_samples: int = 100000, probe: float = 1.14) -> str:
    """Geometric and He-probe void volume, computed and formatted."""
    return void_volume_report(void_volume(mol, n_samples, 0.0), void_volume(mol, n_samples, probe),
                              n_samples, probe)


def surface_area_text(mol: Molecule, n_samples: int = 100000, probe: float = 1.4) -> str:
    r = surface_area(mol, n_samples, probe)
    return "\n".join([
        "Solvent-Accessible Surface Area",
        f"  Probe radius: {probe:g} A" + (" (N2-sized)" if probe == 1.4 else ""),
        f"  Surface area: {r.get('surface_area', 0):.1f} A^2",
        f"  Specific area: {r.get('area_per_volume', 0):.1f} m^2/g",
    ])


def check_text(mol: Optional[Molecule], frames: Optional[list] = None) -> str:
    """The Check Structure report (see ``checks``)."""
    return structure_check_text(mol, frames)


def symmetry_text(mol: Optional[Molecule]) -> str:
    """The Find Symmetry report: space groups ranked by the atomic drift they need."""
    if mol is None or not mol.atoms:
        return "No atoms loaded"
    return symmetry_report(mol)


def charge_report(mol: Molecule) -> str:
    """Summary after QEq: totals, extremes and per-element averages."""
    charges = np.array([a.charge for a in mol.atoms])
    lines = [
        f"QEq charges computed for {len(mol.atoms)} atoms",
        "",
        f"Total charge:  {charges.sum():.6f} e",
        f"Min charge:    {charges.min():.6f} e",
        f"Max charge:    {charges.max():.6f} e",
        f"Mean |charge|: {np.abs(charges).mean():.6f} e",
        "",
        "Per-element averages:",
    ]
    by_el: Dict[str, List[float]] = {}
    for a in mol.atoms:
        by_el.setdefault(a.element.symbol, []).append(a.charge)
    for el in sorted(by_el):
        arr = by_el[el]
        lines.append(f"  {el:<3} ({len(arr):>4}):  avg = {np.mean(arr):>8.5f}  "
                     f"range = [{min(arr):.5f}, {max(arr):.5f}]")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------

def run_pore_size(mol: Molecule, n_samples: int = 50000, n_bins: int = 50,
                  max_pore: float = 20.0) -> AnalysisResult:
    if mol.pbc is None:
        return AnalysisResult(summary="No PBC for pore size calculation")
    bins, hist = pore_size_distribution(mol, n_samples, n_bins, max_pore)
    peak = bins[hist.argmax()] if hist.max() > 0 else 0
    return AnalysisResult(bins * 2, hist, "Pore Size Distribution", "pore diameter (A)",
                          f"Dominant pore diameter: {peak * 2:.2f} A",
                          "pore_diameter_A,fraction", y_max=_grow(hist))


def _empty_window(frames: Optional[list], start: int,
                  end: Optional[int]) -> Optional[AnalysisResult]:
    """A result explaining an empty Start/End window (e.g. Start after End), else None."""
    if frames and len(frames) > 1 and not frame_slice(frames, start, end):
        last = len(frames) if end is None else end
        return AnalysisResult(summary=f"No frames between {start} and {last} "
                                      f"(the trajectory has {len(frames)})")
    return None


def run_rdf(mol: Molecule, el1: str, el2: str, name1: str = ALL_ATOMS, name2: str = ALL_ATOMS,
            r_max: float = 13.0, n_bins: int = 100, frames: Optional[list] = None,
            start: int = 1, end: Optional[int] = None, use_traj: bool = True,
            progress: Optional[Callable[[float], None]] = None) -> AnalysisResult:
    n1, n2 = _opt(name1), _opt(name2)
    label1, label2 = n1 or el1, n2 or el2
    traj = use_traj and frames and len(frames) > 1
    if traj:
        empty = _empty_window(frames, start, end)
        if empty:
            return empty
        sub = frame_slice(frames, start, end)
        r, g = compute_rdf_trajectory(sub, el1, el2, n_bins, r_max, progress, name1=n1, name2=n2)
        first = max(start, 1)
        title = f"g(r) {label1}-{label2} (frames {first}-{first + len(sub) - 1})"
    else:
        r, g = compute_rdf(mol, el1, el2, n_bins, r_max, progress, name1=n1, name2=n2)
        title = f"g(r) {label1}-{label2}"
    if len(r) == 0:
        return AnalysisResult(summary=f"No {label1}-{label2} pairs found")
    return AnalysisResult(r, g, title, "r (A)",
                          f"max g(r) = {g.max():.2f} at r = {r[g.argmax()]:.2f} A", "r(A),g(r)")


def run_coordination(mol: Molecule, center: str, neighbor: str, r_max: float = 5.0,
                     frames: Optional[list] = None, start: int = 1, end: Optional[int] = None,
                     use_traj: bool = False) -> AnalysisResult:
    if use_traj and frames and len(frames) > 1:
        empty = _empty_window(frames, start, end)
        if empty:
            return empty
        r, cn, avg = compute_coordination_trajectory(frame_slice(frames, start, end), center,
                                                     neighbor, r_max)
    else:
        r, cn, avg = compute_coordination(mol, center, neighbor, r_max)
    if len(r) == 0:
        return AnalysisResult(summary=f"No {center} or {neighbor} atoms found")
    return AnalysisResult(r, cn, f"N({neighbor}) around {center}", "r (A)",
                          f"Avg coordination at {r_max} A: {avg:.2f} {neighbor} per {center}",
                          "r_A,coordination_number", y_max=_grow(cn))


def run_hbonds(mol: Molecule, cutoff: float = 3.5, angle: float = 120.0,
               frames: Optional[list] = None, start: int = 1,
               end: Optional[int] = None) -> AnalysisResult:
    if frames and len(frames) > 1:
        empty = _empty_window(frames, start, end)
        if empty:
            return empty
        idx, counts = count_hbonds_trajectory(frame_slice(frames, start, end), cutoff, angle)
        text = ""
    else:
        hb = detect_hbonds(mol, cutoff, angle)
        idx, counts = np.array([0.0]), np.array([float(len(hb))])
        text = "\n".join(["donor      H  acceptor   D...A (A)   angle (deg)"] + [
            f"{d + 1:>5} {h + 1:>6} {a + 1:>9}   {dist:9.3f}   {ang:11.1f}"
            for d, h, a, dist, ang in hb])
    if len(idx) == 0:
        return AnalysisResult(summary="No frames")
    return AnalysisResult(idx, counts, "H-bonds vs frame", "frame",
                          f"Avg: {counts.mean():.1f} H-bonds/frame  |  max: {counts.max():.0f}"
                          f"  |  min: {counts.min():.0f}",
                          "frame,hbond_count", text=text, y_max=_grow(counts))


def run_msd(frames: list, element: str, name: str = ALL_ATOMS, dt: float = 1.0, start: int = 1,
            end: Optional[int] = None) -> AnalysisResult:
    nm = _opt(name)
    empty = _empty_window(frames, start, end)
    if empty:
        return empty
    lags, msd = compute_msd(frame_slice(frames, start, end), element=element, name=nm)
    if len(lags) == 0:
        return AnalysisResult(summary=f"No {element} atoms found")
    summary = "Done"
    D = None
    if len(lags) > 10:
        D, slope = compute_diffusion_constant(lags, msd, dt=dt)
        summary = f"D = {D:.4e} cm^2/s  (slope = {slope:.4f} A^2/ps)"
    return AnalysisResult(lags * dt, msd, f"MSD {nm or element}", "time (ps)", summary,
                          "lag_frames,msd_A2", y_max=_grow(msd), extra={"D": D},
                          csv_data=(lags, msd),
                          text=mc_trajectory_warning(frames, "a diffusion constant"))


def run_rmsd(frames: list, element: str = ALL_ATOMS, name: str = ALL_ATOMS, dt: float = 1.0,
             start: int = 1, end: Optional[int] = None) -> AnalysisResult:
    el, nm = _opt(element), _opt(name)
    empty = _empty_window(frames, start, end)
    if empty:
        return empty
    idx, vals = compute_rmsd(frame_slice(frames, start, end), element=el, name=nm)
    if len(idx) == 0:
        return AnalysisResult(summary="No matching atoms found")
    avg = float(vals[len(vals) // 2:].mean())
    return AnalysisResult(idx * dt, vals, f"RMSD ({el or 'all'})", "time (ps)",
                          f"RMSD avg (2nd half) = {avg:.3f} A  |  max = {vals.max():.3f} A",
                          "frame,rmsd_A", y_max=_grow(vals), csv_data=(idx, vals),
                          text=mc_trajectory_warning(frames, "an RMSD against time"))


def run_gyration(frames: list, element: str = ALL_ATOMS, dt: float = 1.0, start: int = 1,
                 end: Optional[int] = None) -> AnalysisResult:
    el = _opt(element)
    empty = _empty_window(frames, start, end)
    if empty:
        return empty
    idx, rg = compute_gyration_trajectory(frame_slice(frames, start, end), el)
    if len(idx) == 0:
        return AnalysisResult(summary="No data")
    return AnalysisResult(idx * dt, rg, f"Rg ({el or 'all'})", "time (ps)",
                          f"Avg Rg = {rg.mean():.3f} A  |  std = {rg.std():.3f} A", "frame,rg_A",
                          y_min=float(rg.min()) * 0.9 if rg.min() > 0 else 0.0, y_max=_grow(rg),
                          csv_data=(idx, rg))


def run_density_profile(mol: Molecule, element: str, name: str = ALL_ATOMS, axis: int = 2,
                        frames: Optional[list] = None, start: int = 1,
                        end: Optional[int] = None, use_traj: bool = True,
                        n_bins: int = 100) -> AnalysisResult:
    nm = _opt(name)
    axis_label = "abc"[axis]
    traj = use_traj and frames and len(frames) > 1
    if traj:
        empty = _empty_window(frames, start, end)
        if empty:
            return empty
        sub = frame_slice(frames, start, end)
        pos, dens = compute_density_profile_trajectory(sub, axis, n_bins, element, nm)
        title = f"Density {element} ({axis_label}-axis, {len(sub)} frames)"
    else:
        pos, dens = compute_density_profile(mol, axis, n_bins, element, nm)
        title = f"Density {element} ({axis_label}-axis)"
    if len(pos) == 0:
        return AnalysisResult(summary=f"No {element} atoms found")
    return AnalysisResult(pos, dens, title, f"position along {axis_label} (A)",
                          f"max density = {dens.max():.2f} atoms/A at {pos[dens.argmax()]:.2f} A",
                          "position_A,density", y_max=_grow(dens))


def run_pxrd(mol: Molecule, wavelength: float = 1.5406, two_theta_max: float = 50.0,
             peak_width: float = 0.10,
             progress: Optional[Callable[[float], None]] = None) -> AnalysisResult:
    tt, intensity = compute_pxrd(mol, wavelength=wavelength, two_theta_max=two_theta_max,
                                 peak_width=peak_width, progress_callback=progress)
    if len(tt) == 0:
        return AnalysisResult(summary="No reflections found")
    wl_name = "Cu Ka" if abs(wavelength - 1.5406) < 0.01 else f"{wavelength:.4f} A"
    y = np.asarray(intensity)  # peaks: local maxima of the curve, not points above 5%
    n_peaks = int(((y[1:-1] > 5) & (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])).sum())
    summary = f"{n_peaks} peaks above 5% intensity"
    generic = pxrd_untabulated_elements(mol)
    if generic:
        summary += (f"  |  no tabulated form factor for {', '.join(generic)}: "
                    f"intensities approximate")
    return AnalysisResult(tt, intensity, f"PXRD ({wl_name})", "2-theta (deg)",
                          summary, "two_theta_deg,intensity",
                          y_max=105.0)


def read_experimental_pxrd(path: str) -> tuple:
    """Two-column (2-theta, intensity) pattern, scaled to a maximum of 100.

    Columns may be separated by commas, semicolons, tabs or spaces. '#' comments, blank
    lines and text header lines are skipped; extra columns are ignored.
    """
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = [f for f in line.replace(",", " ").replace(";", " ").split() if f]
            try:
                values = [float(f) for f in fields[:2]]
            except ValueError:
                continue  # a header line
            if len(values) == 2:
                rows.append(values)
    if not rows:
        raise ValueError("need 2 columns (2theta, intensity)")
    data = np.array(rows)
    tt, inten = data[:, 0], data[:, 1]
    if inten.max() > 0:
        inten = inten / inten.max() * 100
    return tt, inten


def run_density_3d(frames: list, element: str, n_bins: int = 15,
                   progress: Optional[Callable[[float], None]] = None) -> AnalysisResult:
    positions, values, _ = compute_density_3d(frames, element, n_bins=n_bins,
                                              progress_callback=progress)
    if len(positions) == 0:
        return AnalysisResult(summary=f"No {element} atoms found in trajectory")
    return AnalysisResult(summary=f"3D density: {len(positions)} bins for {element}",
                          extra={"positions": positions, "values": values, "element": element})


# ----------------------------------------------------------------------
# Files: MPMC energy.dat and isotherm folders
# ----------------------------------------------------------------------

def find_energy_file(structure_path: str) -> Optional[str]:
    """energy.dat or energy.csv next to the loaded structure, if present."""
    base = Path(structure_path).parent if structure_path else Path(".")
    for cand in (base / "energy.dat", base / "energy.csv"):
        if cand.exists():
            return str(cand)
    return None


def energy_columns(data: Dict) -> tuple:
    """(plottable columns, default column): everything but 'step'; N, then energy, preferred."""
    cols = [k for k in data if k != "step"]
    default = cols[0] if cols else "N"
    for prefer in ("N", "energy"):
        if prefer in cols:
            default = prefer
            break
    return cols, default


def energy_n_steps(data: Dict) -> int:
    for k in ("step", "N", "energy"):
        if k in data:
            return len(data[k])
    return len(next(iter(data.values()))) if data else 0


def energy_slice(data: Dict, start: int = 1, end: Optional[int] = None) -> slice:
    """1-based inclusive Start/End clamped to the data, as a slice."""
    n = energy_n_steps(data)
    if n == 0:
        return slice(0, 0)
    s = max(0, min(start - 1, n - 1))
    e = n if end is None else end
    return slice(s, max(s + 1, min(e, n)))


def run_energy(data: Dict, column: str, start: int = 1,
               end: Optional[int] = None) -> AnalysisResult:
    if column not in data:
        return AnalysisResult(summary=f"No column {column!r}")
    sl = energy_slice(data, start, end)
    steps = data.get("step", np.arange(len(data[column])))[sl]
    values = data[column][sl]
    if len(values) == 0:
        return AnalysisResult(summary="Empty step range")
    finite = values[np.isfinite(values)]
    if not len(finite):
        return AnalysisResult(summary=f"{column}: no finite values in steps "
                                      f"{sl.start + 1}-{sl.stop}")
    lo, hi = float(finite.min()), float(finite.max())
    y_min = lo * (0.9 if lo > 0 else 1.1)
    y_max = hi * 1.1 if hi > 0 else hi * 0.9
    if y_min == y_max:
        y_max = y_min + 1
    title = column
    if sl.start != 0 or sl.stop != energy_n_steps(data):
        title = f"{column} (steps {sl.start + 1}-{sl.stop})"
    tail = values[-len(values) // 4:] if len(values) > 4 else values
    avg = float(np.nanmean(tail)) if np.isfinite(tail).any() else float("nan")
    return AnalysisResult(steps, values, title, "step",
                          f"{column} [{sl.start + 1}-{sl.stop}]: "
                          f"last-quarter avg = {avg:.4f}, final = {values[-1]:.4f}",
                          ",".join(data.keys()), y_min=y_min, y_max=y_max,
                          extra={"slice": sl})


def write_energy_csv(data: Dict, path: str, sl: slice) -> None:
    """All columns over the chosen steps (the energy plot's CSV export)."""
    cols = list(data.keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(sl.start, sl.stop):
            f.write(",".join(f"{data[c][i]:.6f}" for c in cols) + "\n")


def run_isotherm(directory: str) -> AnalysisResult:
    results = parse_isotherm_results(directory)
    if not results:
        return AnalysisResult(summary="No completed pressure points found")
    p = np.array([r[0] for r in results])
    u = np.array([r[1] for r in results])
    return AnalysisResult(p, u, "Adsorption Isotherm", "pressure (atm)",
                          f"{len(p)} pressure points from {directory}",
                          "pressure_atm,avg_molecules", y_max=_grow(u),
                          text=isotherm_log_table(directory))


def write_csv(path: str, header: str, *arrays: Sequence[float]) -> None:
    """Parallel arrays as CSV with six decimals (the viewers' CSV export)."""
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in zip(*arrays):
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
