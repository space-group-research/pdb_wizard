"""Regression: when `textual` is not installed, the interactive launch must
fall back to the classic text menu instead of exiting with an error.

Two layers:
  - In-process: drive __main__.main() with _ensure_installed("textual") -> False
    and a stubbed run_classic, asserting the fallback (and the no-file error).
  - End-to-end subprocess: make `textual` genuinely unimportable and confirm
    `python -m pdb_wizard <file>` reaches the classic menu rather than dying
    with the old "Use --classic" message.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import pdb_wizard.__main__ as entry
import pdb_wizard.menu as menu

# ---------------------------------------------------------------------------
# In-process fallback logic
# ---------------------------------------------------------------------------

def _fake_ensure(monkeypatch, *, textual_ok: bool):
    """Patch _ensure_installed: numpy always ok, textual per `textual_ok`."""
    def fake(package, import_name=None):
        if package == "textual":
            return textual_ok
        return True
    monkeypatch.setattr(entry, "_ensure_installed", fake)


def test_falls_back_to_classic_when_textual_missing(monkeypatch):
    """Missing textual + a file => run_classic(file), no SystemExit."""
    _fake_ensure(monkeypatch, textual_ok=False)
    called = {}
    monkeypatch.setattr(menu, "run_classic", lambda fp: called.setdefault("fp", fp))
    monkeypatch.setattr(sys, "argv", ["pdb_wizard", "structure.pdb"])

    entry.main()  # must NOT raise / sys.exit

    assert called.get("fp") == "structure.pdb"


def test_missing_textual_without_file_errors_helpfully(monkeypatch, capsys):
    """Missing textual + no file => argparse error mentioning the classic menu."""
    _fake_ensure(monkeypatch, textual_ok=False)
    monkeypatch.setattr(menu, "run_classic",
                        lambda fp: pytest.fail("run_classic should not run without a file"))
    monkeypatch.setattr(sys, "argv", ["pdb_wizard"])

    with pytest.raises(SystemExit):
        entry.main()

    err = capsys.readouterr().err
    assert "textual" in err.lower()
    assert "classic" in err.lower()


def test_does_not_fall_back_when_textual_available(monkeypatch):
    """Textual present => classic menu is NOT used (the TUI path is taken)."""
    _fake_ensure(monkeypatch, textual_ok=True)
    monkeypatch.setattr(menu, "run_classic",
                        lambda fp: pytest.fail("run_classic must not run with textual"))
    # Stop before actually launching the app: make the viewer import explode so
    # we can confirm the TUI branch (not the classic branch) was selected.
    import pdb_wizard.viewer as viewer
    sentinel = RuntimeError("reached TUI branch")

    class _Boom:
        def __init__(self, *a, **k):
            raise sentinel

    monkeypatch.setattr(viewer, "PdbWizardApp", _Boom)
    monkeypatch.setattr(sys, "argv", ["pdb_wizard"])  # no file -> empty buffer TUI

    with pytest.raises(RuntimeError, match="reached TUI branch"):
        entry.main()


# ---------------------------------------------------------------------------
# End-to-end: textual genuinely unimportable
# ---------------------------------------------------------------------------

def test_subprocess_fallback_with_textual_blocked(tmp_path):
    """`python -m pdb_wizard file` with textual unimportable must fall back to
    the classic menu (not exit with the old failure message)."""
    # A fake `textual` package that raises ImportError on import, placed first
    # on PYTHONPATH so it shadows any real install.
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "textual.py").write_text('raise ImportError("textual blocked for test")\n')

    src = str(Path(__file__).resolve().parent.parent / "src")
    env = {
        "PYTHONPATH": f"{blocker}{':'}{src}",
        "PATH": "/usr/bin:/bin",
    }
    # a missing .cif makes run_classic exit at once (sys.exit before any input
    # loop), giving a deterministic, non-hanging terminator that still proves
    # we reached the classic menu.
    result = subprocess.run(
        [sys.executable, "-m", "pdb_wizard", "fake.cif"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )

    combined = result.stdout + result.stderr
    # Fell back rather than dying on the old path.
    assert "falling back to the classic text menu" in combined
    assert "Use --classic for the text menu without textual" not in combined
    # And actually entered run_classic (its could-not-read message).
    assert "Could not read fake.cif" in combined
