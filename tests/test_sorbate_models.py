"""Built-in sorbate models against their sources.

Each model matches the MPMC repo's models/ file, except where that file is known to be
wrong (the group's audit, knowledge/codes/mpmc_models.md, and the papers):
- co2_phastp: LJ sites at 1.114 Å (Mullen et al., JCTC 2013, Table 2), not 1.208;
- o2_trappe: O mass 15.9994, not 7.9997.
n2_polar used to take its N2N sites from the non-polarizable model.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.sorbates import SORBATE_MODELS

REPO_MODELS = Path("/home/user/mpmc/models")
KNOWN_REPO_ERRORS = {"co2_phastp", "o2_trappe"}


def site(model, name):
    return next(s for s in SORBATE_MODELS[model].sites if s.atomtype == name)


def test_co2_phast_star_lj_sites_follow_the_paper():
    cons = [s for s in SORBATE_MODELS["co2_phastp"].sites if s.atomtype == "CON"]
    assert sorted(s.x for s in cons) == [-1.114, 1.114]
    c = site("co2_phastp", "COG")
    assert (c.epsilon, c.sigma, c.charge, c.polarizability) == (19.61757, 3.03366, 0.77134, 1.2281)


def test_o2_trappe_weighs_one_o2():
    assert sum(s.mass for s in SORBATE_MODELS["o2_trappe"].sites) == pytest.approx(31.9988)


def test_n2_polar_uses_its_own_lj_sites():
    n = [s for s in SORBATE_MODELS["n2_polar"].sites if s.atomtype == "N2N"]
    assert {(abs(s.x), s.epsilon, s.sigma) for s in n} == {(0.7445, 16.142, 3.16141)}


def read_repo(path):
    rows = []
    for line in path.read_text().splitlines():
        t = line.split()
        if t and t[0] in ("ATOM", "HETATM"):
            v = [float(x) for x in t[6:]] + [0.0] * 13
            rows.append((t[2], np.array(v[:3]), v[3], v[4], v[5], v[6], v[7]))
    return rows


@pytest.mark.skipif(not REPO_MODELS.is_dir(), reason="MPMC models/ checkout not found")
@pytest.mark.parametrize("key", sorted(SORBATE_MODELS))
def test_models_match_the_mpmc_repo(key):
    path = REPO_MODELS / key
    if not path.is_file() or key in KNOWN_REPO_ERRORS:
        pytest.skip("no repo file, or the repo's file is known to be wrong")
    repo = read_repo(path)
    ours = SORBATE_MODELS[key].sites
    assert sorted(r[0] for r in repo) == sorted(s.atomtype for s in ours)
    by_name = {r[0]: r for r in repo}
    for s in ours:
        _, _, mass, q, alpha, eps, sig = by_name[s.atomtype]
        assert (s.mass, s.charge, s.polarizability, s.epsilon, s.sigma) == pytest.approx(
            (mass, q, alpha, eps, sig), rel=1e-4, abs=1e-6), (key, s.atomtype)

    def distances(points):
        return sorted(float(np.linalg.norm(a - b)) for a, b in itertools.combinations(points, 2))

    np.testing.assert_allclose(distances([np.array([s.x, s.y, s.z]) for s in ours]),
                               distances([r[1] for r in repo]), atol=2e-3)


# ---- the MPMC input form offers every built-in model ----------------------------------------


def test_mpmc_form_offers_exactly_the_builtin_models():
    from pdb_wizard.sorbates import SORBATE_MODELS
    from pdb_wizard.templates import ENGINES, engine_defaults
    var = next(v for v in ENGINES["mpmc_isotherm"].user_vars if v.name == "sorbate")
    assert var.choices == tuple(SORBATE_MODELS)
    assert engine_defaults("mpmc_isotherm")["sorbate"] == "h2_bss" in var.choices


def test_every_offered_sorbate_generates_inputs(tmp_path):
    """No dead entries: each choice in the form writes an isotherm folder."""
    from pathlib import Path

    from pdb_wizard.io import read_file
    from pdb_wizard.isotherm import generate_isotherm
    from pdb_wizard.templates import ENGINES
    mol = read_file(str(Path(__file__).resolve().parent.parent / "HKUST-1.pdb"))
    var = next(v for v in ENGINES["mpmc_isotherm"].user_vars if v.name == "sorbate")
    for name in var.choices:
        dirs = generate_isotherm(mol, sorbate_model=name, temperature=77.0, pressures=[1.0],
                                 output_dir=str(tmp_path / name))
        assert len(dirs) == 1 and (Path(dirs[0]) / "mpmc.inp").exists(), name
