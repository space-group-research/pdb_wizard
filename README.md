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
pdb_wizard structure.cif            # 3D terminal viewer (textual)
pdb_wizard --classic structure.pdb  # the original text menu
pdb_wizard structure.pqr --info     # batch: formula, cell, density
pdb_wizard in.cif --convert -o out.pqr   # batch: convert (.pqr = MPMC format)
pdb_wizard --help                   # all batch options (--rdf, --msd, --surface, --psd, ...)
```

Reads PDB (standard, pdb2pqr and MPMC dialects), CIF, XYZ, POSCAR, LAMMPS data, SDF, MOL2,
Gaussian, Z-matrix, DCD and multi-frame PDB/XYZ trajectories.

### Packing sorbates for MPMC

*Save as MPMC PDB* in the viewer, or option 10 in the classic menu, packs N copies of a
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

The test run builds `dist/pdb_wizard.py` first, so the single-file version is tested too.

## License

GPL-3.0-or-later (see [LICENSE](LICENSE)). The terminal viewer, renderer and Z-matrix reader
are adapted from [moltui](https://github.com/kszenes/moltui) by Kalman Szenes (MIT); see
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
