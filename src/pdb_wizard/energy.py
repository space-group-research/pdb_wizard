"""MPMC energy.dat file reader and plotter."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_energy_dat(filepath: str) -> dict[str, np.ndarray]:
    """Read an MPMC energy.dat file.

    Parses the header line (starting with #) for column names.
    Returns dict mapping column name -> numpy array.
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    lines = path.read_text().splitlines()
    if not lines:
        return {}

    # Parse header for column names
    header_names: list[str] = []
    data_lines: list[list[float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            # Parse header: "#step #energy #coulombic ..." or "# step energy ..."
            tokens = stripped.lstrip("#!").split()
            if tokens and not header_names:  # later comment lines are not the header
                header_names = [t.lstrip("#") for t in tokens]
            continue
        tokens = stripped.split()
        try:
            float(tokens[0])
            data_lines.append([float(t) for t in tokens])
        except (ValueError, IndexError):
            continue

    if not data_lines:
        return {}

    # MPMC flushes the file in blocks, so a running job's last line can be cut short.
    data_lines = [row for row in data_lines if len(row) == len(data_lines[0])]

    arr = np.array(data_lines)
    n_cols = arr.shape[1]

    # If we got header names, use them; otherwise use defaults
    if len(header_names) >= n_cols:
        col_names = header_names[:n_cols]
    elif header_names:
        col_names = header_names + [f"col{i}" for i in range(len(header_names), n_cols)]
    else:
        # Fallback: common MPMC column order
        defaults = ["step", "energy", "coulombic", "rd", "polar", "vdw",
                     "kinetic", "kin_temp", "N", "spin_ratio", "volume", "core_temp"]
        col_names = defaults[:min(n_cols, len(defaults))]
        col_names += [f"col{i}" for i in range(len(col_names), n_cols)]

    result: dict[str, np.ndarray] = {}
    for i, name in enumerate(col_names):
        result[name] = arr[:, i]

    return result
