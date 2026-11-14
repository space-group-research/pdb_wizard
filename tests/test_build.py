"""Tests for the build system — ensures amalgamated and minified files are valid."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BUILD_DIR = Path(__file__).parent.parent
DIST_DIR = BUILD_DIR / "dist"
SRC_DIR = BUILD_DIR / "src" / "pdb_wizard"


def _module_order() -> list[str]:
    """Read MODULE_ORDER from build.py without importing/executing it."""
    tree = ast.parse((BUILD_DIR / "build.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "MODULE_ORDER":
                    return list(ast.literal_eval(node.value))
    raise AssertionError("MODULE_ORDER not found in build.py")


def _referenced_submodules() -> set[str]:
    """Every pdb_wizard submodule imported anywhere in src (incl. lazy,
    function-local imports)."""
    refs: set[str] = set()
    for path in SRC_DIR.rglob("*.py"):
        if "templates" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "pdb_wizard."
            ):
                refs.add(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("pdb_wizard."):
                        refs.add(alias.name.split(".")[1])
    return refs


def _minified_path() -> Path:
    """dist/pdb_wizard.min.py. build.py does not write it when the minifier's output would
    not run on the oldest supported Python (it does that when run on Python 3.12+); the
    build must then say so, and the tests about the minified file have nothing to check."""
    path = DIST_DIR / "pdb_wizard.min.py"
    if not path.exists():
        result = subprocess.run([sys.executable, str(BUILD_DIR / "build.py")],
                                capture_output=True, text=True, cwd=str(BUILD_DIR))
        assert "Minified build skipped" in result.stdout, (
            "dist/pdb_wizard.min.py not found and the build does not say why")
        pytest.skip("this Python's minifier output would not run on the oldest supported one")
    return path


class TestBuild:
    def test_build_runs(self):
        """Build script completes without error."""
        result = subprocess.run(
            [sys.executable, str(BUILD_DIR / "build.py")],
            capture_output=True, text=True, cwd=str(BUILD_DIR),
        )
        assert result.returncode == 0, result.stderr

    def test_full_file_compiles(self):
        """dist/pdb_wizard.py compiles without syntax errors."""
        path = DIST_DIR / "pdb_wizard.py"
        assert path.exists(), "dist/pdb_wizard.py not found — run build.py first"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_minified_file_compiles(self):
        """dist/pdb_wizard.min.py compiles without syntax errors."""
        path = _minified_path()
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_minified_has_shebang(self):
        path = _minified_path()
        first_line = path.read_text().split("\n")[0]
        assert first_line.startswith("#!/usr/bin/env python")

    def test_minified_is_smaller(self):
        full = DIST_DIR / "pdb_wizard.py"
        mini = _minified_path()
        assert full.exists()
        assert mini.stat().st_size < full.stat().st_size

    def test_no_internal_imports_in_amalgamated(self):
        """No 'from pdb_wizard.' imports should remain in the amalgamated file."""
        path = DIST_DIR / "pdb_wizard.py"
        assert path.exists()
        source = path.read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("from pdb_wizard.") or stripped.startswith("import pdb_wizard."):
                pytest.fail(f"Internal import found: {stripped}")

    def test_single_main_block(self):
        path = DIST_DIR / "pdb_wizard.py"
        assert path.exists()
        source = path.read_text()
        count = source.count("if __name__")
        assert count == 1, f"Expected 1 __name__ block, found {count}"


class TestAmalgamationCompleteness:
    """Every pdb_wizard submodule referenced by another module must be in
    build.py's MODULE_ORDER, or its symbols vanish from the dist and any code
    path that uses them (e.g. the input generator's lazy
    `from pdb_wizard.sim_inputs import ...`) hits NameError at runtime."""

    def test_all_referenced_submodules_are_amalgamated(self):
        order = set(_module_order())
        referenced = _referenced_submodules()
        missing = referenced - order
        assert not missing, (
            f"Submodules imported in src but absent from build.py MODULE_ORDER "
            f"(their symbols won't exist in the dist): {sorted(missing)}"
        )

    def test_dist_input_generators_are_callable(self):
        """Regression: sim_inputs was missing from MODULE_ORDER, so the dist's
        CP2K/OpenMM generators NameError'd. Load the dist and run them."""
        path = DIST_DIR / "pdb_wizard.py"
        assert path.exists(), "run build.py first"
        spec = importlib.util.spec_from_file_location("pdbw_dist_test", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["pdbw_dist_test"] = m  # needed for dataclass field resolution
        try:
            spec.loader.exec_module(m)
            mol = m.Molecule(
                atoms=[m.Atom(0, 0, 0, "C"), m.Atom(1.1, 0, 0, "O")],
                pbc=m.PBC(10, 10, 10, 90, 90, 90),
            )
            tmp = Path(self._tmp())
            r1 = m.generate_cp2k_input(mol, str(tmp), "cell_opt")
            r2 = m.generate_openmm_script(mol, str(tmp), "NVT")
            inp = next(Path(v).read_text() for v in r1.values() if str(v).endswith(".inp"))
            run = next(Path(v).read_text() for v in r2.values() if str(v).endswith(".py"))
            assert "CELL_OPT" in inp
            assert "Simulation" in run
        finally:
            sys.modules.pop("pdbw_dist_test", None)

    @staticmethod
    def _tmp() -> str:
        import tempfile

        return tempfile.mkdtemp()


class TestTextualFallback:
    """The amalgamated dist imports textual at top level (it defines widget
    subclasses). When textual is missing it must still load and fall back to the
    classic text menu, not crash with a NameError on a widget base class."""

    def _run_with_textual_blocked(self, tmp_path, args):
        path = DIST_DIR / "pdb_wizard.py"
        assert path.exists(), "run build.py first"
        # Fake `textual` that raises ImportError, shadowing any real install.
        blocker = tmp_path / "blocker"
        blocker.mkdir()
        (blocker / "textual.py").write_text('raise ImportError("blocked")\n')
        env = {"PYTHONPATH": str(blocker), "PATH": "/usr/bin:/bin"}
        return subprocess.run(
            [sys.executable, str(path), *args],
            env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=60,
        )

    def test_dist_loads_without_textual(self, tmp_path):
        """Importing the file with textual missing must not raise NameError."""
        result = self._run_with_textual_blocked(tmp_path, ["--help"])
        assert "NameError" not in (result.stdout + result.stderr)
        assert "Traceback" not in result.stderr

    def test_dist_falls_back_to_classic_menu(self, tmp_path):
        """With textual missing, launching on a file falls back to the classic
        menu (.cif is rejected immediately, proving run_classic was reached)."""
        result = self._run_with_textual_blocked(tmp_path, ["fake.cif"])
        combined = result.stdout + result.stderr
        assert "falling back to the classic text menu" in combined
        assert "NameError" not in combined
        assert "Could not read fake.cif" in combined  # run_classic reached
