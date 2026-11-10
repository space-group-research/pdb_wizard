"""Shared pytest setup.

Several suites test the amalgamated single file (what users download), so it is
built from src/ once, before collection, whenever it is missing or older than the
sources. Options for the optional suites:

  --mpmc PATH       run MPMC itself on the files pdb_wizard writes
  --baseline REF    regression suite: compare against this git ref (or a built
                    pdb_wizard.py path); skipped when it builds to the same file
  --corpus DIR      regression suite: extra structures (*.pqr/*.pdb/*.cif/*.xyz)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "pdb_wizard.py"


def pytest_addoption(parser):
    parser.addoption("--mpmc", default=None,
                     help="path to an mpmc binary; enables the tests that run MPMC itself")
    parser.addoption("--baseline", default="main",
                     help="regression baseline: a git ref, or the path of a built pdb_wizard.py")
    parser.addoption("--corpus", default=None,
                     help="directory of extra structures for the regression suite")


def _dist_is_stale():
    if not DIST.exists():
        return True
    built = DIST.stat().st_mtime
    sources = list((ROOT / "src" / "pdb_wizard").rglob("*.py")) + [ROOT / "build.py"]
    return any(p.stat().st_mtime > built for p in sources)


def pytest_configure(config):
    # Build once in the main process; pytest-xdist workers (which have
    # config.workerinput) start after this and reuse the file.
    if not hasattr(config, "workerinput") and _dist_is_stale():
        subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=ROOT, check=True,
                       capture_output=True)
