"""Tests for the database fetch flow.

These tests stub out urllib so no real network call happens. They verify:
  - The BACKENDS table is the single source of truth (lowercase keys)
    used by both search and fetch — no case-coupling bugs.
  - Each backend's URL is what we claim it is.
  - search results carry source codes that round-trip to fetch dispatch.
  - HTTP helpers send a User-Agent and respect timeout.
  - Errors surface (not silently swallowed for fetch).
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from pdb_wizard import database
from pdb_wizard.database import (
    BACKENDS,
    fetch_cod,
    fetch_pubchem_sdf,
    fetch_rcsb,
    search_cod,
    search_pubchem,
    search_rcsb,
)


def _mock_response(payload: bytes | str | dict | list) -> MagicMock:
    if isinstance(payload, (dict, list)):
        body = json.dumps(payload).encode()
    elif isinstance(payload, str):
        body = payload.encode()
    else:
        body = payload
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# ---------------------------------------------------------------------------
# BACKENDS table contract
# ---------------------------------------------------------------------------

class TestBackendsTable:
    def test_keys_are_lowercase_and_match_search_source_field(self):
        """Every BACKENDS key must match the 'source' field its search returns,
        so the dispatch in _do_fetch can never drift out of sync."""
        # Stub each search to return a single fake hit; check the source field
        cases = {
            "cod": ("search_cod", [{"file": "1", "formula": "C", "title": "t"}]),
            "rcsb": ("search_rcsb", {"result_set": [{"identifier": "1ABC"}]}),
            "pubchem": ("search_pubchem", {"PropertyTable": {"Properties": [
                {"CID": 1, "MolecularFormula": "C", "IUPACName": "x"},
            ]}}),
        }
        for code, (_func_name, payload) in cases.items():
            with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
                results = BACKENDS[code]["search"]("water")
            assert len(results) == 1, f"{code} search returned no hits"
            assert results[0]["source"] == code, (
                f"{code}: search emitted source={results[0]['source']!r}, "
                f"but BACKENDS key is {code!r} — fetch dispatch will fail"
            )

    def test_every_backend_has_label_search_fetch(self):
        for code, spec in BACKENDS.items():
            assert "label" in spec, f"{code} missing label"
            assert callable(spec["search"]), f"{code} search not callable"
            assert callable(spec["fetch"]), f"{code} fetch not callable"
            assert spec["label"], f"{code} label is empty"

    def test_lowercase_canonical_keys_only(self):
        for code in BACKENDS:
            assert code == code.lower(), f"BACKENDS key {code!r} is not lowercase"


# ---------------------------------------------------------------------------
# Search shape: each backend produces the documented dict layout
# ---------------------------------------------------------------------------

class TestSearchShape:
    def test_search_cod_emits_canonical_dict(self):
        payload = [{"file": "4001234", "formula": "C O 2", "title": "carbon dioxide"}]
        with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
            results = search_cod(text="CO2")
        assert results == [{
            "source": "cod",
            "id": "4001234",
            "formula": "C O 2",
            "title": "carbon dioxide",
        }]

    def test_search_rcsb_emits_canonical_dict(self):
        payload = {"result_set": [{"identifier": "1VII"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
            results = search_rcsb("villin")
        assert results == [{
            "source": "rcsb",
            "id": "1VII",
            "formula": "",
            "title": "1VII",
        }]

    def test_search_pubchem_emits_canonical_dict(self):
        payload = {"PropertyTable": {"Properties": [
            {"CID": 2244, "MolecularFormula": "C9H8O4", "IUPACName": "aspirin"},
        ]}}
        with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
            results = search_pubchem("aspirin")
        assert results == [{
            "source": "pubchem",
            "id": "2244",
            "formula": "C9H8O4",
            "title": "aspirin",
        }]

    def test_search_returns_empty_on_network_error(self):
        """The current contract: search swallows network errors and returns [].
        A change to that behavior should be intentional — this test is the
        guard against silent regressions in either direction."""
        err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            assert search_cod(text="anything") == []
            assert search_rcsb("anything") == []
            assert search_pubchem("anything") == []


# ---------------------------------------------------------------------------
# Fetch URL routing — verifies the URL we hit is the one we claim
# ---------------------------------------------------------------------------

class TestFetchURLs:
    def test_fetch_cod_hits_correct_url(self, tmp_path):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["timeout"] = timeout
            return _mock_response(b"data_dummy\n")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            path = fetch_cod("4001234", str(tmp_path))

        assert captured["url"] == "https://www.crystallography.net/cod/4001234.cif"
        assert "User-agent" in captured["headers"] or "User-Agent" in captured["headers"]
        assert captured["timeout"] is not None
        assert open(path).read() == "data_dummy\n"

    def test_fetch_rcsb_hits_correct_url_and_uppercases_id(self, tmp_path):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_response(b"HEADER    DUMMY\nEND\n")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            path = fetch_rcsb("1vii", str(tmp_path))

        assert captured["url"] == "https://files.rcsb.org/download/1VII.pdb"
        assert path.endswith("1VII.pdb")  # uppercased

    def test_fetch_pubchem_uses_cid_path_for_numeric_id(self, tmp_path):
        """Search returns CID numbers; fetch must use the cid/ endpoint, not name/."""
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_response(b"$$$$\n")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            fetch_pubchem_sdf("2244", str(tmp_path))

        assert "/compound/cid/2244/SDF" in captured["url"]

    def test_fetch_pubchem_uses_name_path_for_text(self, tmp_path):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_response(b"$$$$\n")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            fetch_pubchem_sdf("aspirin", str(tmp_path))

        assert "/compound/name/aspirin/SDF" in captured["url"]


# ---------------------------------------------------------------------------
# HTTP helper: User-Agent + timeout + error surfacing on fetch
# ---------------------------------------------------------------------------

class TestHttpHelpers:
    def test_http_get_sets_user_agent(self):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return _mock_response(b"ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            database._http_get("https://example/")

        # Header keys are normalized capitalized in urllib's Request.
        assert any(
            "pdb-wizard" in v.lower() for v in captured["headers"].values()
        ), f"User-Agent not set; headers were {captured['headers']}"

    def test_http_get_passes_timeout(self):
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["timeout"] = timeout
            return _mock_response(b"ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            database._http_get("https://example/", timeout=7.5)

        assert captured["timeout"] == 7.5

    def test_fetch_propagates_http_errors_to_caller(self, tmp_path):
        """fetch_* must raise on network failure so the UI can show an error.
        Silent failure (returning empty file) would let a broken download
        masquerade as a successful fetch."""
        err = urllib.error.HTTPError(
            "http://example/", 404, "Not Found", {}, io.BytesIO(b"")
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(urllib.error.HTTPError):
                fetch_cod("9999999", str(tmp_path))

            with pytest.raises(urllib.error.HTTPError):
                fetch_rcsb("ZZZZ", str(tmp_path))


# ---------------------------------------------------------------------------
# Round-trip: search source → BACKENDS dispatch → fetch URL
# ---------------------------------------------------------------------------

class TestSearchToFetchDispatch:
    @pytest.mark.parametrize("code,search_payload,expected_url_part", [
        ("cod",
         [{"file": "1234567", "formula": "Cu O", "title": "cuprite"}],
         "/cod/1234567.cif"),
        ("rcsb",
         {"result_set": [{"identifier": "1ABC"}]},
         "/download/1ABC.pdb"),
        ("pubchem",
         {"PropertyTable": {"Properties": [
             {"CID": 7777, "MolecularFormula": "X", "IUPACName": "stub"},
         ]}},
         "/compound/cid/7777/SDF"),
    ])
    def test_search_then_fetch_routes_to_correct_url(
        self, code, search_payload, expected_url_part, tmp_path,
    ):
        # 1. Search returns a hit with source=code
        with patch("urllib.request.urlopen", return_value=_mock_response(search_payload)):
            results = BACKENDS[code]["search"]("query")
        assert results, f"{code}: search returned nothing"
        hit = results[0]
        assert hit["source"] == code

        # 2. Fetch the same hit → must hit the expected URL
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_response(b"stub")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            BACKENDS[hit["source"]]["fetch"](hit["id"], str(tmp_path))

        assert expected_url_part in captured["url"], (
            f"{code} fetch went to {captured['url']!r}, expected {expected_url_part!r}"
        )


# ---------------------------------------------------------------------------
# Per-backend modes (text / formula / id) and placeholders
# ---------------------------------------------------------------------------

class TestBackendModes:
    def test_every_backend_declares_at_least_one_mode(self):
        for code, spec in BACKENDS.items():
            assert "modes" in spec, f"{code} missing 'modes'"
            assert spec["modes"], f"{code} has empty modes tuple"
            assert isinstance(spec["modes"], tuple)

    def test_every_mode_has_a_placeholder(self):
        for code, spec in BACKENDS.items():
            ph = spec.get("placeholder", {})
            for mode in spec["modes"]:
                assert mode in ph, f"{code}: mode {mode!r} missing placeholder"
                assert ph[mode], f"{code}: empty placeholder for mode {mode!r}"

    def test_mode_text_present_or_documented_alternative(self):
        """Every backend should support at least one search mode that is
        either 'text' or 'formula' (i.e. user can search without knowing IDs)."""
        for code, spec in BACKENDS.items():
            modes = set(spec["modes"])
            assert modes & {"text", "formula"}, (
                f"{code} has no search-from-scratch mode (only {modes})"
            )

    def test_cod_formula_search_uses_formula_param(self):
        """COD's mode='formula' must call search_cod with formula= kwarg, not text="""
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url if hasattr(req, "full_url") else str(req)
            return _mock_response([{"file": "1", "formula": "C", "title": "x"}])

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            BACKENDS["cod"]["search"]("Cu O", "formula")
        assert "formula=" in captured["url"], (
            f"COD formula mode should hit formula= URL param, got {captured['url']!r}"
        )

        # And mode="text" hits the text= param
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            BACKENDS["cod"]["search"]("villin", "text")
        assert "text=" in captured["url"]

    def test_search_signature_accepts_mode_kw_for_every_backend(self):
        """Every backend's search must accept a 2nd positional 'mode' arg
        without crashing, even if it doesn't use it."""
        with patch("urllib.request.urlopen", return_value=_mock_response([])):
            for code in BACKENDS:
                # Call with an arbitrary mode — most backends ignore it; the
                # contract is that they don't raise.
                BACKENDS[code]["search"]("anything", "text")
