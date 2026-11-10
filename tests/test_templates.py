"""Tests for the template substitution engine.

Covers:
  - load_template raises on missing files
  - find_variables enumerates unique {{{var}}} placeholders
  - render fills placeholders, raises on missing (strict), tolerates extras
  - Engine registry consistency: every user_var name actually appears in its
    template, OR is documented as a generator-supplied derived value
  - Every shipped template loads and parses without leftover placeholders
    after a smoke-render with realistic defaults
"""

from __future__ import annotations

import pytest

from pdb_wizard.templates import (
    ENGINES,
    TEMPLATES_DIR,
    engine_defaults,
    find_variables,
    load_template,
    render,
)

# Vars supplied by the generator code (not the UI) for each engine. These
# placeholders MUST exist in the template but won't be in `engine.user_vars`.
_GENERATOR_VARS = {
    "mpmc_isotherm": {
        "cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",
        "job_name", "pressure", "pqr_input", "model_keywords",
    },
    "cp2k_cell_opt": {"cell_vec_a", "cell_vec_b", "cell_vec_c", "coords_file", "kind_blocks"},
    "cp2k_dft_md": {"cell_vec_a", "cell_vec_b", "cell_vec_c", "coords_file", "kind_blocks"},
    "openmm_npt": {
        "forcefield_xmls", "pdb_filename", "script_filename",
    },
    "openmm_nvt": {
        "forcefield_xmls", "pdb_filename", "script_filename", "nonbonded_setup",
    },
}

# User vars that are consumed by the generator (e.g. forcefield → forcefield_xmls)
# rather than directly substituted, so they won't appear in the template.
_UI_ONLY_VARS = {
    "mpmc_isotherm": {"sorbate", "pressures"},  # loop driver + side file
    "openmm_npt": {"forcefield"},
    "openmm_nvt": {"forcefield", "nonbonded_cutoff_nm"},
}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TestRender:
    def test_simple_substitution(self):
        out = render("hello {{{name}}}", {"name": "world"})
        assert out == "hello world"

    def test_multiple_occurrences_of_same_var(self):
        out = render("{{{x}}} + {{{x}}} = 2{{{x}}}", {"x": "5"})
        assert out == "5 + 5 = 25"

    def test_numeric_value_stringified(self):
        out = render("T={{{t}}}", {"t": 77.5})
        assert out == "T=77.5"

    def test_extra_vars_ignored(self):
        out = render("a={{{a}}}", {"a": 1, "ignored": 99})
        assert out == "a=1"

    def test_missing_var_raises_in_strict_mode(self):
        with pytest.raises(KeyError, match="unfilled"):
            render("a={{{a}}} b={{{b}}}", {"a": 1})

    def test_missing_var_left_in_place_when_not_strict(self):
        out = render("a={{{a}}} b={{{b}}}", {"a": 1}, strict=False)
        assert out == "a=1 b={{{b}}}"

    def test_unrelated_braces_untouched(self):
        # MPMC and CP2K both use single curly braces or double quotes — those
        # must not be touched. Only the triple-brace token is special.
        src = "f({a, b}) {{not}} a placeholder"
        out = render(src, {})
        assert out == src

    def test_no_placeholders_template_returned_verbatim(self):
        src = "no vars here\nat all\n"
        assert render(src, {"unused": "x"}) == src


# ---------------------------------------------------------------------------
# find_variables
# ---------------------------------------------------------------------------

class TestFindVariables:
    def test_returns_unique_ordered_names(self):
        vs = find_variables("{{{a}}} {{{b}}} {{{a}}} {{{c}}}")
        assert vs == ["a", "b", "c"]

    def test_empty_template(self):
        assert find_variables("") == []

    def test_double_braces_not_matched(self):
        # Only triple braces count
        assert find_variables("{{x}} and {x}") == []

    def test_underscored_names_allowed(self):
        assert find_variables("{{{first_arg}}} {{{second_arg_2}}}") == [
            "first_arg", "second_arg_2",
        ]


# ---------------------------------------------------------------------------
# load_template
# ---------------------------------------------------------------------------

class TestLoadTemplate:
    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError, match="template not found"):
            load_template("no_such_template_42.inp")

    @pytest.mark.parametrize("engine_key", list(ENGINES.keys()))
    def test_every_shipped_engine_template_loads(self, engine_key):
        spec = ENGINES[engine_key]
        text = load_template(spec.template_file)
        assert text, f"{spec.template_file} is empty"
        # And it actually contains placeholders
        assert "{{{" in text, f"{spec.template_file} has no placeholders"


# ---------------------------------------------------------------------------
# Engine registry consistency
# ---------------------------------------------------------------------------

class TestEngineRegistry:
    @pytest.mark.parametrize("engine_key", list(ENGINES.keys()))
    def test_user_vars_are_either_in_template_or_marked_ui_only(self, engine_key):
        """Every VarSpec.name must either appear in the template (so the
        renderer fills it) or be documented as a UI-only var consumed by the
        generator code."""
        spec = ENGINES[engine_key]
        tpl_vars = set(find_variables(load_template(spec.template_file)))
        user_var_names = {v.name for v in spec.user_vars}
        ui_only = _UI_ONLY_VARS.get(engine_key, set())
        for name in user_var_names:
            assert name in tpl_vars or name in ui_only, (
                f"engine {engine_key}: user var {name!r} not in template "
                f"and not declared UI-only — orphan"
            )

    @pytest.mark.parametrize("engine_key", list(ENGINES.keys()))
    def test_every_template_placeholder_is_known(self, engine_key):
        """Every {{{var}}} in a template must come from somewhere: either a
        user var or a documented generator-supplied derived value. Catches
        stale templates referencing a var nobody fills."""
        spec = ENGINES[engine_key]
        tpl_vars = set(find_variables(load_template(spec.template_file)))
        user_var_names = {v.name for v in spec.user_vars}
        gen_vars = _GENERATOR_VARS.get(engine_key, set())
        unknown = tpl_vars - user_var_names - gen_vars
        assert not unknown, (
            f"engine {engine_key}: template has placeholders nobody fills: "
            f"{sorted(unknown)}"
        )

    def test_engine_defaults_matches_user_vars(self):
        for key, spec in ENGINES.items():
            defaults = engine_defaults(key)
            for v in spec.user_vars:
                assert v.name in defaults, (
                    f"engine_defaults({key!r}) missing {v.name!r}"
                )
            assert set(defaults.keys()) == {v.name for v in spec.user_vars}


# ---------------------------------------------------------------------------
# End-to-end smoke render with realistic dummy values
# ---------------------------------------------------------------------------

def _dummy_kind_blocks() -> str:
    return (
        "    &KIND C\n"
        "      BASIS_SET DZVP-MOLOPT-SR-GTH\n"
        "      POTENTIAL GTH-PBE-q4\n"
        "    &END KIND\n"
    )


def _dummy_render_vars(engine_key: str) -> dict:
    """Build a kwargs dict that satisfies *all* placeholders in `engine_key`'s
    template so render() in strict mode succeeds."""
    base = dict(engine_defaults(engine_key))
    gen = {
        "cell_a": "20.0", "cell_b": "20.0", "cell_c": "20.0",
        "cell_alpha": "90.0", "cell_beta": "90.0", "cell_gamma": "90.0",
        "cell_vec_a": "20.0 0.0 0.0", "cell_vec_b": "0.0 20.0 0.0", "cell_vec_c": "0.0 0.0 20.0",
        "job_name": "test_job",
        "pressure": "1.0",
        "pqr_input": "input.pqr",
        "model_keywords": "pbc_cutoff 10.0",
        "coords_file": "system.xyz",
        "kind_blocks": _dummy_kind_blocks(),
        "forcefield_xmls": "'amber14-all.xml'",
        "pdb_filename": "system.pdb",
        "script_filename": "run.py",
        "nonbonded_setup": "    nonbondedMethod=PME,\n    nonbondedCutoff=1.0 * nanometer,\n",
    }
    base.update(gen)
    # csv_floats user vars need to be a scalar in the template (loop driver),
    # but the template itself doesn't see them — they're UI-only. Anything goes
    # since render ignores extras.
    return base


@pytest.mark.parametrize("engine_key", list(ENGINES.keys()))
def test_smoke_render_strict_succeeds(engine_key):
    """A full strict render with realistic values must not leave placeholders."""
    spec = ENGINES[engine_key]
    tpl = load_template(spec.template_file)
    rendered = render(tpl, _dummy_render_vars(engine_key), strict=True)
    assert "{{{" not in rendered, (
        f"{engine_key}: leftover placeholders after render:\n{rendered}"
    )


# ---------------------------------------------------------------------------
# Edge: hand-edited template with new placeholder
# ---------------------------------------------------------------------------

def test_render_picks_up_user_added_placeholder(tmp_path):
    """If a user edits a template and adds a NEW placeholder, render
    should fail loudly in strict mode — not silently emit the literal token."""
    custom = tmp_path / "custom.inp"
    custom.write_text("user_added: {{{new_var}}}\n")
    text = custom.read_text()
    with pytest.raises(KeyError):
        render(text, {}, strict=True)
    # And non-strict leaves the token alone
    assert render(text, {}, strict=False) == "user_added: {{{new_var}}}\n"


def test_templates_dir_resolves_to_an_existing_directory():
    """If the package is installed editably, TEMPLATES_DIR must point at the
    on-disk folder so users can edit shipped templates in place."""
    assert TEMPLATES_DIR.is_dir(), f"TEMPLATES_DIR {TEMPLATES_DIR} not a dir"
    # And at least our five shipped templates are there
    shipped = {p.name for p in TEMPLATES_DIR.iterdir() if p.is_file()}
    for spec in ENGINES.values():
        assert spec.template_file in shipped, (
            f"{spec.template_file} missing from {TEMPLATES_DIR}"
        )
