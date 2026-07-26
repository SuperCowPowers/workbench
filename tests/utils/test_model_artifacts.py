"""Tests for the S3 model-artifact readers in model_utils (live Workbench/AWS).

Run against the sandbox:
    WORKBENCH_CONFIG=~/.workbench/scp_sandbox.json .venv/bin/python -m pytest tests/utils/test_model_artifacts.py

Marked `medium` — each test downloads and extracts a tarball from S3.
"""

import json
import os
import tarfile

import awswrangler as wr
import pandas as pd
import pytest

# Workbench Imports
from workbench.api import Model
from workbench.utils.model_utils import (
    _load_json_from_artifact,
    extracted_artifact,
    get_hpo_results,
    load_hyperparameters_from_s3,
)

# XGBoost rather than chemprop: a search is minutes on a CPU box, so regenerating the
# fixtures is cheap. Both come from test_artifacts/create_aqsol_artifacts.py.
HPO_MODEL = "aqsol-xgb-hpo"
PLAIN_MODEL = "aqsol-regression"  # same data/framework, no hpo block

BOGUS_URI = "s3://sandbox-sageworks-artifacts/no/such/artifact.tar.gz"
SCRATCH_PREFIX = "s3://sandbox-sageworks-artifacts/tests/model_artifacts"


@pytest.fixture
def artifact_uri(tmp_path):
    """A two-file tarball in S3: one JSON present, everything else absent."""
    (tmp_path / "present.json").write_text(json.dumps({"answer": 42}))
    tar_path = tmp_path / "artifact.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tmp_path / "present.json", arcname="present.json")

    uri = f"{SCRATCH_PREFIX}/{os.getpid()}/artifact.tar.gz"
    wr.s3.upload(local_file=str(tar_path), path=uri)
    yield uri
    wr.s3.delete_objects(uri)


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
def test_absent_file_in_present_artifact_returns_none(artifact_uri):
    """A readable tarball that lacks the requested file yields None, not an error."""
    assert _load_json_from_artifact(artifact_uri, "present.json") == {"answer": 42}
    assert _load_json_from_artifact(artifact_uri, "nope.json") is None


@pytest.mark.medium
def test_unreachable_artifact_returns_none():
    """An artifact that isn't there yields None rather than raising."""
    assert load_hyperparameters_from_s3(BOGUS_URI) is None
    with extracted_artifact(BOGUS_URI) as artifact_dir:
        assert artifact_dir is None
