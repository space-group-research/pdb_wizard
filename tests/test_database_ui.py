"""UI tests for the unified database fetch popup.

Covers the new features beyond what `test_database.py` checks:
  - Source selector populated from BACKENDS
  - Mode selector adapts to the source (formula not offered on RCSB, etc.)
  - Placeholder updates when source/mode changes
  - "Fetch by ID" path skips search entirely
  - Selected-row preview reflects highlighted row
  - "All" mode dispatches to multiple backends concurrently
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import DataTable, Input, Label, Select

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import DatabaseSearchScreen, PdbWizardApp


def _mol():
    atoms = [Atom(float(i), 0, 0, "C") for i in range(3)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _mock_response(payload):
    body = json.dumps(payload).encode() if isinstance(payload, (dict, list)) else payload
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


async def _push_dbsearch(app, pilot):
    app.push_screen(DatabaseSearchScreen())
    await pilot.pause()
    return app.screen


# ---------------------------------------------------------------------------
# Operations-panel wiring: "Fetch from Database" must land on the unified popup
# ---------------------------------------------------------------------------

class TestOperationsPanelEntry:
    def test_only_unified_db_entry_in_panel(self):
        """The left panel must expose 'Fetch from Database' (unified) and
        no longer have separate 'Fetch from COD' / 'Fetch from RCSB PDB'
        entries — those are subsumed by the unified popup."""
        from pdb_wizard.panels import OPERATIONS
        cmds = [op[0] for op in OPERATIONS]
        labels = [op[1] for op in OPERATIONS]

        assert "db_search" in cmds, "panel missing unified db_search command"
        assert "fetch_cod" not in cmds, (
            "fetch_cod still listed — should be removed in favor of db_search"
        )
        assert "fetch_rcsb" not in cmds, (
            "fetch_rcsb still listed — should be removed in favor of db_search"
        )
        # Sanity: the user-visible label says 'Database'
        idx = cmds.index("db_search")
        assert "Database" in labels[idx], (
            f"db_search label should mention 'Database'; got {labels[idx]!r}"
        )

    @pytest.mark.asyncio
    async def test_panel_db_search_command_opens_unified_modal(self):
        """Sending the 'db_search' RunCommand from the OperationsPanel must
        push DatabaseSearchScreen onto the stack."""
        from pdb_wizard.panels import OperationsPanel
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            assert not isinstance(app.screen, DatabaseSearchScreen)
            app.post_message(OperationsPanel.RunCommand("db_search"))
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, DatabaseSearchScreen), (
                f"Expected DatabaseSearchScreen on top of stack; got "
                f"{type(app.screen).__name__}"
            )


# ---------------------------------------------------------------------------
# Source + Mode selectors
# ---------------------------------------------------------------------------

class TestSourceModeSelectors:
    @pytest.mark.asyncio
    async def test_source_select_lists_every_backend_plus_all(self):
        """Source dropdown must include 'all' + every BACKENDS key."""
        from pdb_wizard.database import BACKENDS
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            opts = [v for _, v in screen.query_one("#db-source", Select)._options
                    if isinstance(v, str)]
            assert "all" in opts
            for code in BACKENDS:
                assert code in opts, f"BACKENDS['{code}'] missing from source select"

    @pytest.mark.asyncio
    async def test_mode_options_change_when_source_changes(self):
        """Switching source from 'all' (text+formula) to 'rcsb' (text+id) must
        rebuild the mode dropdown so 'formula' isn't offered for RCSB."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)

            mode_select = screen.query_one("#db-mode", Select)
            initial_modes = {v for _, v in mode_select._options if isinstance(v, str)}
            assert "text" in initial_modes  # 'all' offers text+formula

            # Switch to RCSB — supports text + id, NOT formula
            screen.query_one("#db-source", Select).value = "rcsb"
            await pilot.pause()
            await pilot.pause()
            new_modes = {v for _, v in mode_select._options if isinstance(v, str)}
            assert "text" in new_modes
            assert "id" in new_modes
            assert "formula" not in new_modes, (
                f"RCSB mode list should not offer formula; got {new_modes}"
            )

    @pytest.mark.asyncio
    async def test_mp_only_offers_formula_and_id(self):
        """Materials Project doesn't have a free-text endpoint here."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-source", Select).value = "mp"
            await pilot.pause()
            modes = {v for _, v in screen.query_one("#db-mode", Select)._options
                     if isinstance(v, str)}
            assert modes == {"formula", "id"}, f"MP modes = {modes}"


# ---------------------------------------------------------------------------
# Placeholder updates with source + mode
# ---------------------------------------------------------------------------

class TestPlaceholderUpdates:
    @pytest.mark.asyncio
    async def test_placeholder_changes_when_source_changes(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            inp = screen.query_one("#db-query", Input)
            initial_placeholder = inp.placeholder

            # Switch to PubChem with text mode → placeholder mentions "aspirin"
            screen.query_one("#db-source", Select).value = "pubchem"
            await pilot.pause()
            assert inp.placeholder != initial_placeholder
            assert "aspirin" in inp.placeholder.lower(), (
                f"PubChem placeholder should hint a compound name; got {inp.placeholder!r}"
            )

    @pytest.mark.asyncio
    async def test_placeholder_changes_when_mode_changes(self):
        """Switching from text → id mode on COD should change the placeholder
        from 'cuprite'-style to '1542931'-style."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-source", Select).value = "cod"
            await pilot.pause()
            inp = screen.query_one("#db-query", Input)
            text_placeholder = inp.placeholder

            screen.query_one("#db-mode", Select).value = "id"
            await pilot.pause()
            assert inp.placeholder != text_placeholder, (
                "Placeholder should change between modes"
            )
            # The id placeholder should look numeric-ish, not be a word
            assert any(c.isdigit() for c in inp.placeholder), (
                f"id placeholder should be numeric-ish; got {inp.placeholder!r}"
            )


# ---------------------------------------------------------------------------
# Direct-ID fetch
# ---------------------------------------------------------------------------

class TestDirectIdFetch:
    @pytest.mark.asyncio
    async def test_fetch_by_id_skips_search_and_calls_fetch_directly(self, tmp_path):
        """In id mode, clicking 'Fetch by ID' must hit the fetch URL with the
        exact query string — no search round-trip first."""
        captured_urls: list[str] = []

        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return _mock_response(b"data_dummy\n")

        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-source", Select).value = "cod"
            await pilot.pause()
            screen.query_one("#db-query", Input).value = "1542931"
            await pilot.pause()

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                # Override the cache dir so we don't litter the real fs
                app._db_cache_dir_path = str(tmp_path)
                # Click "Fetch by ID"
                await pilot.click("#db-fetch-id")
                # Wait for fetch task
                for _ in range(40):
                    import asyncio as _aio
                    await _aio.sleep(0.05)
                    if screen._compute_task and screen._compute_task.done():
                        break

        # Exactly one URL hit, and it's the COD fetch URL with our ID
        assert len(captured_urls) == 1, f"expected 1 URL hit, got {captured_urls}"
        assert "/cod/1542931.cif" in captured_urls[0]

    @pytest.mark.asyncio
    async def test_fetch_by_id_with_all_source_notifies_user(self):
        """Direct-ID fetch only makes sense for a single backend. Picking
        'all' + Fetch-by-ID should notify the user, not crash."""
        notifications = []
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            orig = app.notify

            def capture(msg, *args, **kwargs):
                notifications.append(str(msg))
                return orig(msg, *args, **kwargs)
            app.notify = capture

            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-query", Input).value = "1234"
            await pilot.click("#db-fetch-id")
            await pilot.pause()

            assert any(
                "specific" in n.lower() or "not 'All'" in n.lower() or "all" in n.lower()
                for n in notifications
            ), f"expected a 'pick a specific db' notification; got {notifications}"


# ---------------------------------------------------------------------------
# 'All' source aggregates from multiple backends
# ---------------------------------------------------------------------------

class TestAllSourceAggregation:
    @pytest.mark.asyncio
    async def test_all_text_search_dispatches_to_text_capable_backends(self):
        """In 'all' + 'text' mode, search must hit COD + RCSB + PubChem
        (text-supporting), but NOT MP (which has no text mode)."""
        from pdb_wizard.database import BACKENDS
        text_backends = {c for c, s in BACKENDS.items() if "text" in s["modes"]}
        assert "mp" not in text_backends  # confirm fixture assumption

        captured_calls: list[str] = []

        def fake_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            captured_calls.append(url)
            return _mock_response([])  # empty results

        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-query", Input).value = "test"

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                await pilot.click("#db-search")
                for _ in range(40):
                    import asyncio as _aio
                    await _aio.sleep(0.05)
                    if screen._compute_task and screen._compute_task.done():
                        break

        # Each text-capable backend should have been hit at least once
        joined = "\n".join(captured_calls)
        assert "crystallography.net" in joined, "COD should have been queried"
        assert "search.rcsb.org" in joined or "rcsb.org" in joined, "RCSB should have been queried"
        assert "pubchem" in joined, "PubChem should have been queried"
        # MP has no text mode — should NOT be hit
        assert "materialsproject" not in joined, (
            "MP must not be hit for text search — it only supports formula"
        )

    @pytest.mark.asyncio
    async def test_search_status_breaks_down_per_source_in_all_mode(self):
        """The status line in 'all' mode must show per-source result counts."""
        # COD returns 2 hits; RCSB and PubChem return 0
        cod_hits = [
            {"file": "1", "formula": "C", "title": "a"},
            {"file": "2", "formula": "O", "title": "b"},
        ]

        def fake_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "cod" in url and "result.php" in url:
                return _mock_response(cod_hits)
            return _mock_response({})  # empty for others

        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-query", Input).value = "anything"

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                await pilot.click("#db-search")
                for _ in range(40):
                    import asyncio as _aio
                    await _aio.sleep(0.05)
                    if screen._compute_task and screen._compute_task.done():
                        break
                    await pilot.pause()

            status = str(screen.query_one("#db-status", Label).render())
            # Should contain "2 results" (total) and per-source breakdown
            assert "2 results" in status, status
            assert "COD=2" in status, (
                f"per-source breakdown should mention COD=2; got {status!r}"
            )


# ---------------------------------------------------------------------------
# Selected-row preview
# ---------------------------------------------------------------------------

class TestRowPreview:
    @pytest.mark.asyncio
    async def test_preview_shows_selected_row_details(self):
        """Cursoring through results should populate #db-preview with the
        formula/title of the highlighted row."""
        cod_hits = [
            {"file": "111", "formula": "C9H8O4", "title": "aspirin"},
            {"file": "222", "formula": "Cu O", "title": "cuprite"},
        ]

        def fake_urlopen(req, timeout=None):
            return _mock_response(cod_hits)

        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            screen = await _push_dbsearch(app, pilot)
            screen.query_one("#db-source", Select).value = "cod"
            screen.query_one("#db-query", Input).value = "test"
            await pilot.pause()

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                await pilot.click("#db-search")
                for _ in range(40):
                    import asyncio as _aio
                    await _aio.sleep(0.05)
                    if screen._compute_task and screen._compute_task.done():
                        break

            # After search, table has 2 rows. Move cursor to row 0.
            table = screen.query_one("#db-results", DataTable)
            table.move_cursor(row=0)
            await pilot.pause()
            preview = str(screen.query_one("#db-preview", Label).render())
            assert "aspirin" in preview or "C9H8O4" in preview, (
                f"row-0 preview should mention aspirin/C9H8O4; got {preview!r}"
            )

            table.move_cursor(row=1)
            await pilot.pause()
            preview = str(screen.query_one("#db-preview", Label).render())
            assert "cuprite" in preview or "Cu O" in preview, (
                f"row-1 preview should mention cuprite/Cu O; got {preview!r}"
            )
