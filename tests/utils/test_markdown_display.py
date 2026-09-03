"""Readable rendering of hyperparameter values in the model UI.

Model hyperparameters come back as float32, so a task weight set to 0.3 arrives as
0.29999998211860657. A multi-task model has one per task, which filled the details panel
with several hundred characters of float noise.
"""

import math

from workbench.utils.markdown_utils import display_list, display_names, display_value

FLOAT32_POINT_THREE = 0.29999998211860657


def test_float32_noise_reads_as_the_number_it_is():
    # The producer no longer emits this, but stored hyperparameters from earlier models do.
    assert display_value(FLOAT32_POINT_THREE) == "0.3"


def test_small_magnitudes_survive():
    # Significant figures, not decimal places — %.2f would flatten these to 0.00.
    assert display_value(0.0001) == "0.0001"
    assert display_value(1e-8) == "1e-08"


def test_non_floats_pass_through():
    assert display_value(700) == "700"
    assert display_value("mae") == "mae"
    assert display_value(None) == "None"
    # bool is an int subclass; it must not be formatted as a number
    assert display_value(True) == "True"


def test_non_finite_is_left_alone():
    assert display_value(float("nan")) == "nan"
    assert display_value(math.inf) == "inf"


def test_repeated_run_collapses():
    weights = [0.6628257036209106, 1.2044342756271362] + [FLOAT32_POINT_THREE] * 22
    assert display_list(weights) == "0.663, 1.2, 0.3 (x22)"


def test_short_runs_stay_written_out():
    # Collapsing two entries into "(x2)" costs characters rather than saving them.
    assert display_list([0.3, 0.3]) == "0.3, 0.3"
    assert display_list([1.0, 0.3, 0.3, 0.3]) == "1, 0.3 (x3)"


def test_runs_are_positional_not_global():
    # A value recurring in separate runs stays in the order the tasks are declared.
    assert display_list([0.3] * 3 + [1.0] + [0.3] * 3) == "0.3 (x3), 1, 0.3 (x3)"


def test_mixed_types_and_empty():
    assert display_list(["a", "b"]) == "a, b"
    assert display_list([]) == ""


def test_weights_from_the_producer_are_already_clean():
    """The float32 divide that produced the noise is gone; display only has to shorten."""
    import pandas as pd

    from workbench.utils.multi_task import compute_inverse_count_task_weights

    df = pd.DataFrame({f"t{i}": [1.0] * n + [None] * (2500 - n) for i, n in enumerate([2335, 1285, 1493, 1412])})
    weights = compute_inverse_count_task_weights(df, list(df.columns))
    assert sum(weights) / len(weights) == 1.0
    assert 0.3 * (sum(weights) / len(weights)) == 0.3


# --- display_names: long name lists in the model details panel ----------------------

TARGETS = (
    [f"cyp{i}_pic50_direct_inhibition" for i in ("3a4", "2c9", "2d6", "1a2")]
    + [f"cyp{i}_log2fc" for i in ("3a4", "2c9", "2d6", "1a2")]
    + [f"cyp{i}_pic50_chembl" for i in ("3a4", "2c9", "2d6", "1a2", "2c19")]
)


def test_long_list_leads_with_the_count_and_says_what_it_hid():
    out = display_names(TARGETS)
    assert out.startswith("(13) cyp3a4_pic50_direct_inhibition")
    assert out.endswith(" more")
    assert len(out) < 140


def test_short_list_is_not_marked_as_truncated():
    # The old formatter appended "..." unconditionally, so a two-feature model read as elided.
    assert display_names(["smiles", "mol_weight"]) == "(2) smiles, mol_weight"
    assert "..." not in display_names(["solubility"])


def test_a_bare_string_counts_as_one_name():
    assert display_names("solubility") == "(1) solubility"


def test_empty_and_missing_are_distinguishable():
    assert display_names([]) == "(0)"
    assert display_names(None) == "-"


def test_a_single_over_long_name_is_kept_whole():
    # Better to overflow once than to render a name that does not exist.
    name = "x" * 140
    assert display_names([name]) == f"(1) {name}"


def test_truncation_reports_the_real_remainder():
    out = display_names(TARGETS, max_chars=60)
    shown = out.split(") ", 1)[1].split(", ... +")[0].split(", ")
    assert f"+{len(TARGETS) - len(shown)} more" in out
