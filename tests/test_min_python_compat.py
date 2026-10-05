"""Guard rails for the project's minimum-Python support target.

pdb_wizard runs on HPC/cluster boxes with old interpreters, so `pyproject.toml`
declares `requires-python = ">=3.7"`. The interactive TUI viewer needs
textual >=3.0 (Python >=3.8.1), so on 3.7 only the batch CLI + `--classic`
text menu are supported -- but the library, the CLI, and the amalgamated
single-file dist must still import and run on 3.7.

Several runtime traps are invisible to `from __future__ import annotations`
(which only defers *annotation* expressions, never these) and to vermin:

  1. PEP 604 unions (`X | Y`) / PEP 585 generics (`list[X]`) in a **base-class
     subscript** -- evaluated at class-definition time, so they crash on
     <3.10 / <3.9. Use typing.Optional / typing.List there.
  2. Positional-only parameters (`def f(a, /)`, PEP 570) are 3.8+ syntax and
     would make the whole amalgamated dist fail to parse on 3.7.
  3. `typing.Literal` is 3.8+; sim_inputs.py must fall back to
     `typing_extensions.Literal` so it imports on 3.7.
  4. `asyncio.to_thread` is 3.9+; viewer.py installs a backport at import.
  5. build.py has no `from __future__ import annotations`, so its own function
     signatures must not use PEP 585/604 or the build crashes on 3.7/3.8.

These checks are static / execution-based, so they pass on any interpreter and
fail loudly if someone reintroduces a version-gated construct.
"""

from __future__ import annotations

import ast
import asyncio
import re
import types
from pathlib import Path

import pytest

import pdb_wizard

SRC_DIR = Path(pdb_wizard.__file__).parent
REPO_ROOT = SRC_DIR.parent.parent
BUILD_PY = REPO_ROOT / "build.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"

PEP585_BUILTINS = {"list", "dict", "set", "tuple", "frozenset", "type"}


def _src_modules() -> list[Path]:
    # Skip the templates package: those .py files are text templates with
    # {{placeholder}} markers, not importable modules.
    return [p for p in SRC_DIR.rglob("*.py") if "templates" not in p.parts]


def _runtime_generic_offenders(expr: ast.expr) -> list[str]:
    """PEP 604 unions / PEP 585 generics inside an expression -- constructs
    evaluated at runtime that break on <3.10 / <3.9."""
    problems: list[str] = []
    for node in ast.walk(expr):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            problems.append("PEP 604 union (`X | Y`)")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in PEP585_BUILTINS:
                problems.append(f"PEP 585 generic (`{node.value.id}[...]`)")
    return problems


class TestNoRuntimeUnionsInBaseClasses:
    def test_base_class_subscripts_are_version_safe(self):
        """No class may subscript a base with `X | Y` or a lowercase builtin
        generic -- those are evaluated when the class is defined and crash on
        Python 3.7/3.8/3.9."""
        failures: list[str] = []
        for path in _src_modules():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):  # catches nested classes too
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    for problem in _runtime_generic_offenders(base):
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{base.lineno} "
                            f"class {node.name} base uses {problem}"
                        )
        assert not failures, (
            "Runtime-evaluated PEP 604/585 syntax in base classes breaks old "
            "Python:\n  " + "\n  ".join(failures)
        )


class TestNoPositionalOnlyParams:
    def test_no_positional_only_syntax(self):
        """Positional-only params (`def f(a, /)`, PEP 570) are 3.8+ syntax; a
        single one anywhere makes the amalgamated dist unparsable on 3.7."""
        failures: list[str] = []
        for path in _src_modules():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # posonlyargs only exists on the 3.8+ AST; absent on 3.7.
                    if getattr(node.args, "posonlyargs", None):
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                            f"{node.name}() uses positional-only params"
                        )
        assert not failures, (
            "Positional-only parameters are 3.8+ syntax:\n  "
            + "\n  ".join(failures)
        )


class TestLiteralBackport:
    def test_sim_inputs_falls_back_to_typing_extensions(self):
        """typing.Literal is 3.8+; sim_inputs must import it with a
        typing_extensions fallback so it works on 3.7."""
        text = (SRC_DIR / "sim_inputs.py").read_text()
        assert "from typing_extensions import Literal" in text, (
            "sim_inputs.py uses typing.Literal but has no typing_extensions "
            "fallback -- it will fail to import on Python 3.7"
        )


class TestToThreadBackport:
    def test_polyfill_installs_and_runs_when_missing(self):
        """viewer.py must define a working asyncio.to_thread backport that
        activates when the interpreter lacks it (3.7/3.8)."""
        # Extract the `if not hasattr(asyncio, "to_thread"):` block by
        # indentation (ast.get_source_segment is 3.8+, so we stay 3.7-safe).
        lines = (SRC_DIR / "viewer.py").read_text().splitlines()
        start = next(
            (i for i, ln in enumerate(lines)
             if ln.startswith("if not hasattr(asyncio,") and "to_thread" in ln),
            None,
        )
        assert start is not None, "to_thread backport block not found in viewer.py"
        block_lines = [lines[start]]
        for ln in lines[start + 1:]:
            if ln.strip() == "" or ln.startswith((" ", "\t")):
                block_lines.append(ln)
            else:
                break
        block = "\n".join(block_lines)

        # Execute the real polyfill source against a stand-in asyncio module
        # lacking to_thread, proving the fallback path works end to end.
        fake = types.ModuleType("asyncio")
        fake.get_running_loop = asyncio.get_running_loop
        ns = {"asyncio": fake}
        exec(block, ns)  # noqa: S102 - executing our own vetted source
        assert hasattr(fake, "to_thread"), "backport did not install to_thread"

        async def _main():
            return await fake.to_thread(lambda a, b: a + b, 40, 2)

        assert asyncio.run(_main()) == 42

    def test_to_thread_available_after_importing_viewer(self):
        # viewer needs textual, which is unsupported on 3.7; skip there.
        pytest.importorskip("textual")
        import asyncio as _a

        import pdb_wizard.viewer  # noqa: F401  (import installs the backport)
        assert hasattr(_a, "to_thread")


class TestPackagingMetadata:
    def test_requires_python_allows_37(self):
        text = PYPROJECT.read_text()
        m = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        assert m, "requires-python not found in pyproject.toml"
        assert "3.7" in m.group(1), (
            f"requires-python is {m.group(1)!r}; expected the 3.7 floor"
        )

    def test_typing_extensions_dependency_for_37(self):
        text = PYPROJECT.read_text()
        assert re.search(r"typing_extensions.*python_version\s*<\s*'3\.8'", text), (
            "typing_extensions must be a dependency on Python <3.8 (Literal backport)"
        )

    def test_numpy_has_37_marker(self):
        text = PYPROJECT.read_text()
        assert re.search(r"numpy>=1\.21.*python_version\s*<\s*'3\.8'", text), (
            "numpy needs a <3.8 marker pinning 1.21 (last release with 3.7 wheels)"
        )

    def test_build_py_defers_annotations(self):
        """build.py uses PEP 585 syntax in its own signatures but can only
        stringize them via `from __future__ import annotations`; without it the
        build crashes on 3.7/3.8."""
        head = BUILD_PY.read_text()[:600]
        assert "from __future__ import annotations" in head, (
            "build.py must import `from __future__ import annotations` so its "
            "own PEP 585 annotations don't crash the build on old Python"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
