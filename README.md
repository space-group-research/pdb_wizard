# PDB Wizard

Structure handling for porous-materials simulations: read, clean up, extend and inspect
crystal structures, assign force fields and charges, pack sorbate molecules, and write
inputs for MPMC, CP2K, OpenMM and LAMMPS. It runs as an interactive 3D terminal viewer,
a classic text menu, or a batch command-line tool.

Developed in the [Space group](https://drbrian.space) at NC State University.

## Install

```bash
pip install "pdb-wizard[viewer] @ git+https://github.com/space-group-research/pdb_wizard"
```

`[viewer]` adds [textual](https://textual.textualize.io) for the terminal viewer. Without it,
everything except the viewer works (the batch tools and `--classic` menu run on Python 3.7+).

A single-file version, `pdb_wizard.py`, needs only NumPy. Build it with `python build.py`
(written to `dist/`); the group ships it in
[space-group-environment](https://github.com/space-group-research/space-group-environment).

## Use

```bash
pdb_wizard --gui structure.cif      # desktop window (Qt; needs the gui extra, PySide6)
pdb_wizard structure.cif            # 3D terminal viewer (textual)
pdb_wizard --classic structure.pdb  # the original text menu
pdb_wizard structure.pqr --info     # batch: formula, cell, density
pdb_wizard in.cif --convert -o out.pqr   # batch: convert (.pqr = MPMC format)
pdb_wizard --help                   # all batch options (--rdf, --msd, --surface, --psd, ...)
```

Reads PDB (standard, pdb2pqr and MPMC dialects), CIF, XYZ, POSCAR, LAMMPS data, SDF, MOL2,
Gaussian, Z-matrix, DCD and multi-frame PDB/XYZ trajectories.

### Desktop window

![PDB Wizard's desktop window: HKUST-1 with packed DC97 water](docs/gui.png)

![An analysis window: the Cu–O radial distribution in HKUST-1](docs/gui-analysis.png)

`pdb_wizard --gui` has everything the terminal viewer has, in windows and menus:

- **View**: ball and stick, licorice, space filling or protein ribbon; colour by element,
  charge, residue, chain or index; atom numbers, hidden water, coordination polyhedra,
  perspective or orthographic, dark or light background, lighting and size sliders. Drag to
  rotate, right-drag to pan, wheel to zoom; the terminal viewer's keys work too (`v`, `C`,
  `#`, `w`, `P`, `O`, `m`, `t`, `c`, `r`, `,`/`.`).
- **Measure**: in measure mode two clicks pin a distance; shift-click two or three atoms
  for a distance or angle.
- **Tabs and trajectories**: each file opens in a tab; multi-frame PDB/XYZ and DCD files
  get a frame slider with play, speed and "open frame in new tab".
- **Edit** (with undo): wrap, sort, supercells and reducing them back (choosing which
  sorbates to keep), delete lone atoms, set X–H distances, update or remove the cell,
  substitute elements, edit coordinates, QEq or file charges, force fields, and sorbate
  packing.
- **Analysis**: system info, void volume, surface area, pore sizes, RDF, coordination
  numbers, H-bonds, MSD and diffusion constants, RMSD, radius of gyration, density
  profiles, 3D density maps, powder XRD (with an experimental overlay), MPMC energy.dat
  and isotherm plots, each with CSV and PNG export.
- **Tools**: the input generator (MPMC isotherms, CP2K, OpenMM), structure database
  search (COD, RCSB PDB, PubChem, Materials Project) and a geometry panel listing bonds,
  angles, dihedrals, contacts, molecules and lone atoms.
- **Export**: PNG and rotation GIF.

Both front ends run the same analysis and input-generator code, so they report the same
numbers (the test suite checks this). The window is drawn with Qt's painter rather than
OpenGL, so it also works over SSH X forwarding. It uses the desktop's own style on KDE
(Breeze) and a dark Fusion theme elsewhere. *Save as* picks the format from the extension
(`.pqr` = MPMC). It needs PySide6, the package's `gui` extra.

### Packing sorbates for MPMC

The *Sorbates* tab in the desktop window, *Edit > Pack Sorbates* (or *Save as MPMC PDB*)
in the terminal viewer, or option 10 in the classic menu packs N copies of a
sorbate (or fills the cell) at random positions and orientations. Overlaps are checked with
the minimum image in the actual (monoclinic or triclinic) cell, against the framework and
against each other. Built-in models include TIP3P, TIP4P, TIP5P, SPC/E, DC97 and POL3 water,
H2, CO2, CH4, N2, O2 and noble gases (LJ, polarizable and PHAHST); any MPMC `.pqr` molecule
can be used as a template. MPMC files that already contain sorbates keep them movable when
re-saved, and force fields or charges applied afterwards only touch the framework.

Random packing is a starting point, not an equilibrated state: relax it in MPMC (NVT) before
starting μVT from it.

## Tests

```bash
pip install -e ".[viewer,test]" pytest-asyncio pytest-xdist hypothesis python-minifier
pytest -n auto
```

Optional suites:

| Option | Runs |
|---|---|
| `--mpmc /path/to/mpmc` | MPMC itself on files pdb_wizard writes (energies, GCMC start-up) |
| `--baseline REF` | differential regression tests against a git ref or a built `pdb_wizard.py` (default `main`; skipped when identical) |
| `--corpus DIR` | the regression tests over every structure under `DIR` (e.g. MPMC's `tutorials_and_examples`) |

The test run builds `dist/pdb_wizard.py` first, so the single-file version is tested too. The
desktop-window tests run headless (`QT_QPA_PLATFORM=offscreen`) and skip without PySide6 and
pytest-qt. `tests/test_gui_parity.py` runs every analysis in both the desktop window and the
terminal viewer and requires identical results, and fails if a terminal-viewer operation has
no desktop counterpart.

## License

GPL-3.0-or-later (see [LICENSE](LICENSE)). The terminal viewer, renderer and Z-matrix reader
are adapted from [moltui](https://github.com/kszenes/moltui) by Kalman Szenes (MIT); see
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
