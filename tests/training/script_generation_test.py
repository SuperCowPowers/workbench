"""Tests for template filling — specifically that hyperparameters cross as JSON.

Hyperparameters are caller data (knob names, categorical options, free-text notes) that
lands inside generated Python source. Encoding it as a ``repr``-escaped JSON string is what
keeps that data from being read as code, so these tests pin the escaping and the round trip
rather than any particular template's content.
"""

import ast
import json
import tempfile

import pytest

# Workbench Imports
from workbench.model_scripts.script_generation import fill_template

BASE_PARAMS = {
    "model_type": "regressor",
    "target_column": "solubility",
    "feature_list": ["alpha", "beta"],
    "id_column": "id",
    "model_metrics_path": "s3://bucket/metrics",
}


def _generate(hyperparameters):
    """Fill a minimal template and return (source, executed TEMPLATE_PARAMS)."""
    template = tempfile.NamedTemporaryFile("w", suffix=".template", delete=False)
    template.write('import json\nTEMPLATE_PARAMS = {\n    "hyperparameters": json.loads("{{hyperparameters}}"),\n}\n')
    template.close()

    script = fill_template(
        template.name, {**BASE_PARAMS, "hyperparameters": hyperparameters}, "gen.py", output_dir=tempfile.mkdtemp()
    )
    source = open(script).read()
    ast.parse(source)  # the generated script must be valid Python
    namespace = {}
    exec(compile(source, script, "exec"), namespace)  # noqa: S102 — the point is that it runs
    return source, namespace["TEMPLATE_PARAMS"]["hyperparameters"]


def test_hyperparameters_round_trip_unchanged():
    """What the caller passed is what the training script sees."""
    hyperparameters = {
        "uq_version": "v1",
        "max_lr": 0.001,
        "layers": "512-128",
        "hpo": {"n_trials": 60, "search_space": "basic+optimizer"},
    }
    _, loaded = _generate(hyperparameters)
    assert loaded == hyperparameters


def test_quotes_and_newlines_cannot_break_out_of_the_literal():
    """The value is data, not source. A caller string closing the quote and appending code
    must stay inside the literal — otherwise hyperparameters are an injection vector."""
    payload = '", "injected": __import__("os").system("touch /tmp/pwned"), "x": "'
    _, loaded = _generate({"note": payload, "multiline": "one\ntwo\\three"})

    assert loaded["note"] == payload  # survives verbatim...
    assert loaded["multiline"] == "one\ntwo\\three"
    assert set(loaded) == {"note", "multiline"}  # ...and creates no new keys


def test_the_generated_source_has_no_python_literal_dict():
    """Regression guard: the dict must not be str()'d into the source. A Python literal
    would carry True/None/single quotes and would be re-parsed as code."""
    source, _ = _generate({"flag": True, "missing": None, "ratio": 0.5})

    assert "json.loads(" in source
    assert "'flag': True" not in source  # the Python repr spelling
    assert '"flag": true' in source  # the JSON spelling


def test_unicode_and_nested_structures_survive():
    """Nested dicts/lists are how the hpo block and search spaces travel."""
    hyperparameters = {
        "note": "α β γ",
        "hpo": {"search_space": {"ffn_hidden_dim": {"dist": "choice", "options": [900, 1200, "512-128"]}}},
    }
    _, loaded = _generate(hyperparameters)
    assert loaded == hyperparameters
    assert loaded["hpo"]["search_space"]["ffn_hidden_dim"]["options"] == [900, 1200, "512-128"]


@pytest.mark.parametrize("empty", [None, {}])
def test_absent_hyperparameters_become_an_empty_dict(empty):
    """A model with no hyperparameters still needs a valid script."""
    _, loaded = _generate(empty)
    assert loaded == {}


def test_shipped_templates_generate_valid_python():
    """The real templates, not a stand-in: each must parse after filling."""
    from workbench.api import ModelFramework, ModelType
    from workbench.model_scripts.script_generation import generate_model_script

    frameworks = [
        (ModelFramework.XGBOOST, ModelType.REGRESSOR),
        (ModelFramework.CHEMPROP, ModelType.REGRESSOR),
        (ModelFramework.PYTORCH, ModelType.REGRESSOR),
    ]
    for framework, model_type in frameworks:
        params = {
            **BASE_PARAMS,
            "model_type": model_type,
            "model_framework": framework,
            "model_class": None,
            "compressed_features": [],
            "hyperparameters": {"uq_version": "v1", "hpo": {"n_trials": 5}, "note": 'has "quotes"'},
        }
        script = generate_model_script(params)
        source = open(script).read()
        tree = ast.parse(source)  # the generated script must be valid Python

        # Pull the json.loads argument straight out of the AST: if the quotes in `note` had
        # escaped the literal, this would not parse as a single string constant.
        payloads = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        assert len(payloads) == 1, framework
        assert json.loads(payloads[0]) == params["hyperparameters"], framework
