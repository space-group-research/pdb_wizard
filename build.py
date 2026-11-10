#!/usr/bin/env python3
"""Build amalgamated pdb_wizard.py single-file distribution.

Reads all modules from src/pdb_wizard/ in dependency order, strips internal
imports, deduplicates external imports, and writes a single dist/pdb_wizard.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src" / "pdb_wizard"
DIST_DIR = Path(__file__).parent / "dist"

MODULE_ORDER = [
    "constants",
    "atom",
    "pbc",
    "forcefields",
    "molecule",
    "io",
    "dcd",
    "trajectory",
    "geometry",
    "operations",
    "renderer",
    "charges",
    "rdf",
    "msd",
    "rmsd",
    "coordination",
    "hbonds",
    "gyration",
    "density3d",
    "templates",
    "sim_inputs",
    "isotherm",
    "reduce_cell",
    "energy",
    "density_profile",
    "pxrd",
    "analyses",
    "inputgen",
    "formats",
    "database",
    "sorbates",
    "panels",
    "viewer",
    "menu",
    "gui_tools",
    "gui",
    "__main__",
]

HEADER = '''\
#!/usr/bin/env python3
# PDB Wizard v0.5.0 — Amalgamated Single-File Distribution
# Based on pdb_wizard by Adam Hogan (GPLv3) and moltui by Kalman Szenes (MIT)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
from __future__ import annotations

import os as _os
import subprocess as _subprocess
import sys as _sys

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

def _ensure_installed(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    print(f"\\'{package}\\' is not installed.")
    try:
        answer = input(f"Install it now with pip? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in ("", "y", "yes"):
        _subprocess.check_call([_sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    return False

if not _ensure_installed("numpy"):
    _sys.exit(1)
'''

INTERNAL_IMPORT_RE = re.compile(
    r"^\s*from\s+pdb_wizard(?:\.\w+)?\s+import\s+.*$"
    r"|^\s*from\s+\.\w+\s+import\s+.*$"
    r"|^\s*import\s+pdb_wizard(?:\.\w+)?\s*$"
)

FUTURE_RE = re.compile(r"^\s*from\s+__future__\s+import\s+annotations\s*$")

IMPORT_RE = re.compile(r"^\s*(import\s+\S+|from\s+\S+\s+import\s+.*)$")

# Textual imports that should be wrapped in try/except
TEXTUAL_IMPORT_RE = re.compile(r"^\s*(?:from\s+textual|from\s+rich)\b")


def extract_imported_names(stmt: str) -> list[str]:
    """Return the names bound by an import statement.

    Handles `from X import a, b as c, (d, e)` (the only form used for textual
    imports). Used to stub these names when textual is unavailable so that the
    class definitions subclassing them still load.
    """
    s = stmt.replace("(", " ").replace(")", " ").replace("\\", " ")
    if " import " not in s:
        return []
    _, _, tail = s.partition(" import ")
    names: list[str] = []
    for part in tail.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(" as ")[-1].strip() if " as " in part else part
        name = name.split(".")[0].strip()
        if name.isidentifier():
            names.append(name)
    return names


# Stub emitted when textual/rich are missing so the amalgamated file still
# imports (and main() can fall back to the classic menu). Every textual base
# class is replaced by a permissive dummy that can be subclassed (swallowing
# class keyword args like can_focus=True), called, and subscripted
# (ModalScreen[...]). These stubs are never used by the classic text menu.
TEXTUAL_STUB = '''\
    class _TextualStub(type):
        def __new__(mcls, name, bases, ns, **kw):
            return super().__new__(mcls, name, bases, ns)
        def __init__(cls, name, bases, ns, **kw):
            super().__init__(name, bases, ns)
        def __getattr__(cls, _n):
            return _TextualStub(_n, (), {})
        def __call__(cls, *a, **k):
            return cls
        def __getitem__(cls, _i):
            return cls
    def _textual_stub(_n):
        return _TextualStub(_n, (), {})
'''


def is_internal_import(line: str) -> bool:
    return bool(INTERNAL_IMPORT_RE.match(line))


def is_future_import(line: str) -> bool:
    return bool(FUTURE_RE.match(line))


def is_external_import(line: str) -> bool:
    return bool(IMPORT_RE.match(line)) and not is_internal_import(line)


def extract_imports_and_body(source: str) -> tuple[list[str], str]:
    """Split a module into its external imports and remaining body.

    Scans ALL lines, strips internal and __future__ imports (including
    multi-line imports), and collects external imports.
    """
    lines = source.splitlines(keepends=True)
    imports: list[str] = []
    body_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        is_indented = line != line.lstrip()

        if is_future_import(stripped):
            i += 1
            continue

        # For indented imports: strip internal ones, keep external ones in-place
        if is_indented:
            if is_internal_import(stripped):
                # Collect full multi-line import
                full = stripped
                while ("(" in full and ")" not in full) or full.endswith("\\"):
                    i += 1
                    if i < len(lines):
                        full += "\n" + lines[i].rstrip()
                    else:
                        break
                i += 1
                continue
            body_lines.append(line)
            i += 1
            continue

        # Top-level imports
        if is_internal_import(stripped) or is_external_import(stripped):
            is_internal = is_internal_import(stripped)
            full_import = stripped
            while ("(" in full_import and ")" not in full_import) or full_import.endswith("\\"):
                i += 1
                if i < len(lines):
                    full_import += "\n" + lines[i].rstrip()
                else:
                    break
            if is_internal:
                i += 1
                continue
            else:
                imports.append(full_import.replace("\n", " ").replace("\\", ""))
                i += 1
                continue

        body_lines.append(line)
        i += 1

    body = "".join(body_lines)
    # Strip any if __name__ == "__main__" blocks from modules
    # (build.py adds its own at the end)
    import re as _re
    body = _re.sub(
        r'\n*if __name__\s*==\s*["\']__main__["\']\s*:\s*\n(?:[ \t]+[^\n]*\n?)*',
        '\n', body,
    )
    return imports, body


def deduplicate_imports(imports: list[str]) -> tuple[list[str], list[str]]:
    """Split imports into standard and textual, deduplicated."""
    seen: set[str] = set()
    standard: list[str] = []
    textual: list[str] = []

    for imp in imports:
        key = imp.strip()
        if key in seen:
            continue
        seen.add(key)
        if TEXTUAL_IMPORT_RE.match(key):
            textual.append(key)
        else:
            standard.append(key)

    return standard, textual


def build() -> None:
    DIST_DIR.mkdir(exist_ok=True)

    all_imports: list[str] = []
    all_bodies: list[str] = []

    for module_name in MODULE_ORDER:
        filepath = SRC_DIR / f"{module_name}.py"
        if not filepath.exists():
            # Try as a package (e.g. templates/__init__.py)
            pkg_init = SRC_DIR / module_name / "__init__.py"
            if pkg_init.exists():
                filepath = pkg_init
            else:
                print(f"WARNING: {filepath} not found, skipping", file=sys.stderr)
                continue

        source = filepath.read_text()
        imports, body = extract_imports_and_body(source)
        all_imports.extend(imports)
        all_bodies.append(f"\n# {'=' * 70}\n# Module: {module_name}\n# {'=' * 70}\n")
        all_bodies.append(body)

        # Templates package: inline shipped template files into a dict so the
        # amalgamated dist works without the templates/ folder next to it.
        if module_name == "templates":
            tpl_dir = SRC_DIR / "templates"
            tpl_files = sorted(p for p in tpl_dir.iterdir()
                               if p.is_file() and p.name != "__init__.py")
            all_bodies.append(
                "\n# Bundled template payload (inlined by build.py)\n"
                "_BUNDLED_TEMPLATES.update({\n"
            )
            for tpl in tpl_files:
                all_bodies.append(
                    f"    {tpl.name!r}: {tpl.read_text()!r},\n"
                )
            all_bodies.append("})\n")

    standard_imports, textual_imports = deduplicate_imports(all_imports)

    out_lines = [HEADER, "\n"]

    # Standard imports
    for imp in standard_imports:
        out_lines.append(imp + "\n")

    # Textual imports wrapped in try/except. When textual/rich are missing the
    # except branch stubs out every imported name so that the textual-dependent
    # class definitions still load — letting main() fall back to the classic
    # text menu instead of crashing the whole file at import time.
    if textual_imports:
        out_lines.append("\n_HAS_TEXTUAL = False\ntry:\n")
        for imp in textual_imports:
            out_lines.append(f"    {imp}\n")
        out_lines.append("    _HAS_TEXTUAL = True\nexcept ImportError:\n")
        out_lines.append(TEXTUAL_STUB)
        seen: set[str] = set()
        for imp in textual_imports:
            for nm in extract_imported_names(imp):
                if nm not in seen:
                    seen.add(nm)
                    out_lines.append(f"    {nm} = _textual_stub({nm!r})\n")

    # Module bodies
    for body in all_bodies:
        out_lines.append(body)

    # Entry point
    out_lines.append("\n\nif __name__ == '__main__':\n    main()\n")

    full_source = "".join(out_lines)

    # Full version
    out_path = DIST_DIR / "pdb_wizard.py"
    out_path.write_text(full_source)
    out_path.chmod(0o755)
    print(f"Built {out_path} ({len(full_source)} bytes)")

    # Minified version
    try:
        import python_minifier
        minified = python_minifier.minify(
            full_source,
            remove_annotations=True,
            remove_pass=True,
            remove_literal_statements=True,
            combine_imports=True,
            hoist_literals=True,
            rename_locals=True,
            rename_globals=False,  # Textual uses class names for message routing
            convert_posargs_to_args=True,
            preserve_shebang=True,
            remove_object_base=True,
            remove_explicit_return_none=True,
        )
        # python-minifier can produce invalid syntax when collapsing
        # f-strings with unicode onto one line — verify it compiles
        compile(minified, "<minified>", "exec")
        min_path = DIST_DIR / "pdb_wizard.min.py"
        min_path.write_text(minified)
        min_path.chmod(0o755)
        ratio = len(minified) / len(full_source) * 100
        print(f"Built {min_path} ({len(minified)} bytes, {ratio:.0f}% of original)")
    except ImportError:
        print("python-minifier not installed, skipping minified build")


if __name__ == "__main__":
    build()
