"""Tests for the S3 model-artifact readers in model_utils (live Workbench/AWS).

Run against the sandbox:
    WORKBENCH_CONFIG=~/.workbench/scp_sandbox.json .venv/bin/python -m pytest tests/utils/test_model_artifacts.py

Marked `medium` — each test downloads and extracts a tarball from S3.
"""

import pandas as pd
import pytest

# Workbench Imports
from workbench.api import Model
from workbench.utils.model_utils import (
    extracted_artifact,
    get_hpo_results,
    load_category_mappings_from_s3,
    load_hyperparameters_from_s3,
)

HPO_MODEL = "aqsol-chemprop-hpo"  # published by examples/models/chemprop_hpo.py
PLAIN_MODEL = "aqsol-chemprop"  # same data/framework, no hpo block

BOGUS_URI = "s3://sandbox-sageworks-artifacts/no/such/artifact.tar.gz"


@pytest.mark.medium
def test_get_hpo_results_shape():
    """A searched model returns its best_config plus the rerank/trials frames."""
    results = get_hpo_results(Model(HPO_MODEL))
    assert results is not None, f"{HPO_MODEL} should carry HPO artifacts"

    assert results["metric"] in ("cv_mae", "holdout_mae")
    assert isinstance(results["best_config"], dict) and results["best_config"]
    # best_value/baseline_value share the re-rank basis; search_best_value is phase 1's.
    assert isinstance(results["best_value"], float)
    assert isinstance(results["search_best_value"], float)
    assert isinstance(results["rerank_fresh_split"], bool)

    rerank = results["rerank"]
    assert isinstance(rerank, pd.DataFrame)
    # The baseline always rides along as a candidate — that's the downside guard.
    assert "baseline" in rerank["candidate"].tolist()
    assert results["metric"] in rerank.columns

    assert isinstance(results["trials"], pd.DataFrame)
    assert len(results["trials"]) > 0


@pytest.mark.medium
def test_get_hpo_results_none_for_unsearched_model():
    """No search artifacts means None, so this doubles as the is-it-HPO check."""
    assert get_hpo_results(Model(PLAIN_MODEL)) is None


@pytest.mark.medium
def test_published_config_matches_a_rerank_candidate():
    """The published config is whichever candidate won the re-rank, not the search's pick."""
    results = get_hpo_results(Model(HPO_MODEL))
    winner = min(
        (row for _, row in results["rerank"].iterrows() if pd.notna(row[results["metric"]])),
        key=lambda row: row[results["metric"]],
    )
    assert winner[results["metric"]] == pytest.approx(results["best_value"])


@pytest.mark.medium
def test_load_hyperparameters_from_s3():
    """Hyperparameters come back from the model bundle for a real model."""
    hyperparameters = load_hyperparameters_from_s3(Model(HPO_MODEL).model_data_url())
    assert isinstance(hyperparameters, dict)
    assert "n_folds" in hyperparameters
    # The hpo block is dropped when the winner is published.
    assert "hpo" not in hyperparameters


@pytest.mark.medium
def test_absent_file_in_present_artifact_returns_none():
    """A chemprop bundle has no category_mappings.json — absent file, not an error."""
    assert load_category_mappings_from_s3(Model(HPO_MODEL).model_data_url()) is None


@pytest.mark.medium
def test_unreachable_artifact_returns_none():
    """An artifact that isn't there yields None rather than raising."""
    assert load_hyperparameters_from_s3(BOGUS_URI) is None
    with extracted_artifact(BOGUS_URI) as artifact_dir:
        assert artifact_dir is None
