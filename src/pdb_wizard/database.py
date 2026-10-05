"""Structure database fetching: COD, RCSB PDB, PubChem, Materials Project."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

# Some servers (notably PubChem) reject Python's default User-Agent.
_USER_AGENT = "pdb-wizard/1.0 (+https://github.com)"
_DEFAULT_TIMEOUT = 30.0


def _http_get(url: str, *, timeout: float = _DEFAULT_TIMEOUT,
              extra_headers: dict | None = None) -> bytes:
    """GET a URL with timeout and User-Agent. Returns the response bytes.
    Raises urllib.error.URLError / HTTPError on failure."""
    headers = {"User-Agent": _USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_download(url: str, out_path: str, *,
                   timeout: float = _DEFAULT_TIMEOUT,
                   extra_headers: dict | None = None) -> str:
    """Download a URL to disk with timeout and User-Agent. Returns out_path."""
    payload = _http_get(url, timeout=timeout, extra_headers=extra_headers)
    with open(out_path, "wb") as f:
        f.write(payload)
    return out_path


def fetch_cod(cod_id: str, output_dir: str = ".") -> str:
    """Fetch a CIF file from the Crystallography Open Database by ID.

    Args:
        cod_id: COD entry ID (e.g., '4512072')
        output_dir: Directory to save the file.

    Returns:
        Path to the downloaded CIF file.
    """
    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
    out_path = str(Path(output_dir) / f"COD_{cod_id}.cif")
    return _http_download(url, out_path)


def fetch_rcsb(pdb_id: str, output_dir: str = ".") -> str:
    """Fetch a PDB file from the RCSB Protein Data Bank.

    Args:
        pdb_id: 4-letter PDB code (e.g., '1VII')
        output_dir: Directory to save the file.

    Returns:
        Path to the downloaded PDB file.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    out_path = str(Path(output_dir) / f"{pdb_id.upper()}.pdb")
    return _http_download(url, out_path)


def search_cod(formula: str = "", text: str = "") -> list[dict]:
    """Search COD for structures matching a formula or text query.

    Returns list of dicts with 'id', 'formula', 'title', 'source' keys.
    """
    params = []
    if formula:
        params.append("formula=" + urllib.parse.quote(formula))
    if text:
        params.append("text=" + urllib.parse.quote(text))
    if not params:
        return []

    query = "&".join(params)
    url = "https://www.crystallography.net/cod/result.php?" + query + "&format=json"
    try:
        data = json.loads(_http_get(url, timeout=10))
        results = []
        for entry in data[:50]:
            results.append({
                "source": "cod",
                "id": str(entry.get("file", "")),
                "formula": entry.get("formula", ""),
                "title": entry.get("title", "")[:60],
            })
        return results
    except Exception:
        return []


def search_rcsb(query: str) -> list[dict]:
    """Search RCSB PDB for structures matching a text query.

    Returns list of dicts with 'id', 'formula', 'title', 'source' keys.
    """
    if not query.strip():
        return []
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 25}},
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        for hit in data.get("result_set", []):
            pdb_id = hit.get("identifier", "")
            results.append({
                "source": "rcsb",
                "id": pdb_id,
                "formula": "",
                "title": pdb_id,
            })
        return results
    except Exception:
        return []


def fetch_pubchem_sdf(name: str, output_dir: str = ".") -> str:
    """Fetch a 3D SDF file from PubChem by compound name.

    Args:
        name: Compound name (e.g., 'aspirin', 'benzene') or numeric CID.
        output_dir: Directory to save the file.

    Returns:
        Path to the downloaded SDF file.
    """
    # PubChem search returns CIDs; allow either name or CID here.
    is_cid = name.isdigit()
    encoded = urllib.parse.quote(name)
    if is_cid:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + encoded + "/SDF?record_type=3d"
        )
        out_path = str(Path(output_dir) / f"CID_{name}.sdf")
    else:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            + encoded + "/SDF?record_type=3d"
        )
        out_path = str(Path(output_dir) / (name.replace(" ", "_") + ".sdf"))
    return _http_download(url, out_path)


def search_pubchem(query: str) -> list[dict]:
    """Search PubChem for compounds by name.

    Returns list of dicts with 'id' (CID), 'formula', 'title', 'source' keys.
    """
    if not query.strip():
        return []
    encoded = urllib.parse.quote(query)
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + encoded + "/property/MolecularFormula,IUPACName/JSON")
    try:
        data = json.loads(_http_get(url, timeout=10))
        results = []
        for prop in data.get("PropertyTable", {}).get("Properties", [])[:25]:
            cid = str(prop.get("CID", ""))
            results.append({
                "source": "pubchem",
                "id": cid,
                "formula": prop.get("MolecularFormula", ""),
                "title": prop.get("IUPACName", "")[:60],
            })
        return results
    except Exception:
        return []


def search_materials_project(formula: str, api_key: str = "") -> list[dict]:
    """Search Materials Project for structures by formula.

    Requires MP_API_KEY environment variable or api_key parameter.
    Returns list of dicts with 'id', 'formula', 'title', 'source' keys.
    """
    import os
    key = api_key or os.environ.get("MP_API_KEY", "")
    if not key or not formula.strip():
        return []
    encoded = urllib.parse.quote(formula.strip())
    url = ("https://api.materialsproject.org/materials/summary/"
           "?formula=" + encoded + "&_limit=25&_fields=material_id,formula_pretty,symmetry")
    try:
        data = json.loads(_http_get(url, timeout=15, extra_headers={"X-API-KEY": key}))
        results = []
        for entry in data.get("data", []):
            mid = entry.get("material_id", "")
            formula_p = entry.get("formula_pretty", "")
            sym = entry.get("symmetry", {})
            sg = sym.get("symbol", "") if isinstance(sym, dict) else ""
            results.append({
                "source": "mp",
                "id": mid,
                "formula": formula_p,
                "title": sg,
            })
        return results
    except Exception:
        return []


def fetch_materials_project_cif(
    material_id: str, api_key: str = "", output_dir: str = ".",
) -> str:
    """Fetch a CIF from Materials Project by material ID (e.g. 'mp-149')."""
    import os
    key = api_key or os.environ.get("MP_API_KEY", "")
    if not key:
        raise ValueError("Set MP_API_KEY env var (free at materialsproject.org)")
    url = ("https://api.materialsproject.org/materials/summary/"
           "?material_ids=" + material_id + "&_fields=structure&_limit=1")
    data = json.loads(_http_get(url, timeout=15, extra_headers={"X-API-KEY": key}))
    entries = data.get("data", [])
    if not entries:
        raise ValueError(f"Material {material_id} not found")
    structure = entries[0].get("structure", {})
    out_path = str(Path(output_dir) / (material_id.replace("/", "_") + ".cif"))
    lattice = structure.get("lattice", {})
    sites = structure.get("sites", [])
    with open(out_path, "w") as f:
        f.write("data_mp\n")
        for p, v in [("a", "a"), ("b", "b"), ("c", "c")]:
            f.write(f"_cell_length_{p}    {lattice.get(v, 1)}\n")
        for p, v in [("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")]:
            f.write(f"_cell_angle_{p} {lattice.get(v, 90)}\n")
        f.write("_symmetry_space_group_name_H-M   'P 1'\n\n")
        f.write("loop_\n_atom_site_label\n_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
        for i, site in enumerate(sites):
            species = site.get("species", [{}])
            el = species[0].get("element", "X") if species else "X"
            abc = site.get("abc", [0, 0, 0])
            f.write(f"{el}{i+1} {el} {abc[0]:.6f} {abc[1]:.6f} {abc[2]:.6f}\n")
    return out_path


# ---------------------------------------------------------------------------
# Backend registry — single source of truth for code, label, search, fetch.
# Lowercase keys flow through both the UI Select and the result dict's
# 'source' field so dispatch can never drift between search and fetch.
#
# Each backend declares its supported search modes:
#   - "text":    free-text query (works on every backend)
#   - "formula": chemical formula (e.g. "C9H8O4", "Cu O")
#   - "id":      direct ID — bypasses search entirely, calls fetch with the
#                user's exact string
# UI is responsible for offering only the modes a backend supports.
# ---------------------------------------------------------------------------

def _search_cod_dispatch(query: str, mode: str = "text") -> list[dict]:
    if mode == "formula":
        return search_cod(formula=query)
    return search_cod(text=query)


BACKENDS: dict[str, dict] = {
    "cod": {
        "label": "COD",
        "modes": ("text", "formula", "id"),
        "placeholder": {
            "text": "anything (eg 'cuprite')",
            "formula": "Cu O",
            "id": "1542931",
        },
        "search": _search_cod_dispatch,
        "fetch": fetch_cod,
    },
    "rcsb": {
        "label": "RCSB PDB",
        "modes": ("text", "id"),
        "placeholder": {
            "text": "villin headpiece",
            "id": "1VII",
        },
        "search": lambda q, mode="text": search_rcsb(q),
        "fetch": fetch_rcsb,
    },
    "pubchem": {
        "label": "PubChem",
        "modes": ("text", "id"),
        "placeholder": {
            "text": "aspirin",
            "id": "2244 (CID)",
        },
        "search": lambda q, mode="text": search_pubchem(q),
        "fetch": fetch_pubchem_sdf,
    },
    "mp": {
        "label": "Materials Project",
        "modes": ("formula", "id"),
        "placeholder": {
            "formula": "Si O2",
            "id": "mp-149",
        },
        "search": lambda q, mode="formula": search_materials_project(q),
        # fetch_materials_project_cif takes (material_id, api_key, output_dir)
        # Bind api_key='' so callers pass (id, output_dir) like the others.
        "fetch": lambda mid, output_dir=".": fetch_materials_project_cif(mid, "", output_dir),
    },
}
