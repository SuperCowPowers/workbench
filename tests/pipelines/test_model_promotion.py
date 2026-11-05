"""Tests for the default promotion arbiter (workbench:models/model_promotion.py)"""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from workbench.api import ModelType
from workbench.batch.models import model_promotion
from workbench.batch.models.model_promotion import beats, main

ENDPOINT = "my-endpoint-reg-1"


class FakeEndpointDeploy:
    """What to_endpoint() hands back: the inference calls the arbiter makes on a new endpoint."""

    def __init__(self):
        self.test_inference_calls = 0
        self.cross_fold_calls = 0

    def test_inference(self):
        self.test_inference_calls += 1

    def cross_fold_inference(self):
        self.cross_fold_calls += 1


class FakeModel:
    """A stand-in for Model: metrics, deprecated meta, and a record of copy/delete/deploy."""

    def __init__(self, name, rmse=None, mae=None, f1=None, deprecated=False, exists=True):
        self.name = name
        self.rmse = rmse
        self.mae = mae
        self.f1 = f1
        self.deprecated = deprecated
        self._exists = exists
        self.model_type = ModelType.CLASSIFIER if f1 is not None else ModelType.REGRESSOR
        self.deleted = False
        self.copied_to = None
        self.frozen = None
        self.deployed_to = None
        self.deploy = FakeEndpointDeploy()

    def exists(self):
        return self._exists

    def workbench_meta(self):
        return {"deprecated": True} if self.deprecated else {}

    def get_inference_metrics(self, inference_run="full_cross_fold"):
        if self.f1 is not None:
            return pd.DataFrame([{"class": "all", "f1": self.f1}])
        if self.rmse is None:
            return None
        return pd.DataFrame([{"rmse": self.rmse, "mae": self.mae}])

    def copy(self, dated_name, owner=None):
        self.copied_to = dated_name
        self.frozen = FakeModel(dated_name, rmse=self.rmse, mae=self.mae, f1=self.f1)
        return self.frozen

    def to_endpoint(self, endpoint_name, **kwargs):
        self.deployed_to = endpoint_name
        return self.deploy

    def delete(self):
        self.deleted = True

    def created(self):
        return pd.Timestamp("2026-01-01", tz="UTC")


class Arbiter:
    """Runs main() against a fake registry, capturing what it promoted and reported."""

    def __init__(self, challengers, no_promote=None, incumbent=None, endpoint_exists=True):
        self.models = {m.name: m for m in challengers}
        self.incumbent = incumbent
        if incumbent is not None:
            self.models[incumbent.name] = incumbent
        self.endpoint_exists = endpoint_exists
        self.no_promote = no_promote or []
        self.report_calls = []
        self.frozen = None

    def _model(self, name):
        # An unknown name is a model that does not exist, as Model() would report
        return self.models.get(name) or FakeModel(name, exists=False)

    def _endpoint(self, name):
        end = MagicMock()
        end.exists.return_value = self.endpoint_exists
        end.get_input.return_value = self.incumbent.name if self.incumbent else None
        return end

    def _contest_report(self, champion, challengers, endpoint_name, **kwargs):
        self.report_calls.append(
            {
                "champion": champion.name if champion is not None else None,
                "challengers": [c.name for c in challengers],
                "no_promote": kwargs.get("no_promote"),
            }
        )
        return pd.DataFrame([{"model": "row"}])

    def run(self):
        meta = {
            "endpoint_name": ENDPOINT,
            "serverless": True,
            "challengers": [m.name for m in self.models.values() if m is not self.incumbent],
            "no_promote": self.no_promote,
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            with patch.object(model_promotion, "Model", side_effect=self._model):
                with patch.object(model_promotion, "Endpoint", side_effect=self._endpoint):
                    with patch.object(model_promotion, "contest_report", side_effect=self._contest_report):
                        with patch.object(model_promotion, "Reports", MagicMock()):
                            main()
        return self

    @property
    def promoted(self):
        """The challenger that was frozen onto the endpoint, or None if nothing was promoted."""
        promoted = [m for m in self.models.values() if m.copied_to is not None]
        assert len(promoted) <= 1, f"More than one model promoted: {[m.name for m in promoted]}"
        return promoted[0] if promoted else None

    @property
    def reported(self):
        """The single contest report published by this run."""
        assert len(self.report_calls) == 1, f"Expected one contest report, got {len(self.report_calls)}"
        return self.report_calls[0]


class TestNoPromote:
    """A :no_promote challenger competes and is reported, but never takes the endpoint."""

    def test_a_no_promote_challenger_never_wins(self):
        # The flagged model has the best rmse, so only the flag can keep it off the endpoint
        run = Arbiter(
            challengers=[
                FakeModel("best-but-flagged", rmse=0.50, mae=0.40),
                FakeModel("runner-up", rmse=0.60, mae=0.50),
            ],
            no_promote=["best-but-flagged"],
        ).run()

        assert run.promoted.name == "runner-up"
        assert run.promoted.copied_to.startswith("runner-up-")
        assert run.models["best-but-flagged"].copied_to is None

    def test_a_no_promote_challenger_is_still_ranked_and_reported(self):
        run = Arbiter(
            challengers=[
                FakeModel("best-but-flagged", rmse=0.50, mae=0.40),
                FakeModel("runner-up", rmse=0.60, mae=0.50),
            ],
            no_promote=["best-but-flagged"],
        ).run()

        # Ranked best-first, flagged model included, and the flag reaches the report
        assert run.reported["challengers"] == ["best-but-flagged", "runner-up"]
        assert run.reported["no_promote"] == ["best-but-flagged"]

    def test_every_challenger_flagged_promotes_nothing_but_still_reports(self):
        run = Arbiter(
            challengers=[FakeModel("flagged-a", rmse=0.50, mae=0.40), FakeModel("flagged-b", rmse=0.60, mae=0.50)],
            no_promote=["flagged-a", "flagged-b"],
        ).run()

        assert run.promoted is None
        assert run.reported["challengers"] == ["flagged-a", "flagged-b"]

    def test_an_unflagged_cohort_promotes_the_best(self):
        run = Arbiter(
            challengers=[FakeModel("winner", rmse=0.50, mae=0.40), FakeModel("runner-up", rmse=0.60, mae=0.50)]
        ).run()

        assert run.promoted.name == "winner"
        assert run.reported["no_promote"] == []


class TestDeprecated:
    """A deprecated challenger is treated like a no_promote one: ranked, reported, never promoted."""

    def test_a_deprecated_challenger_never_wins(self):
        run = Arbiter(
            challengers=[
                FakeModel("best-but-deprecated", rmse=0.50, mae=0.40, deprecated=True),
                FakeModel("runner-up", rmse=0.60, mae=0.50),
            ]
        ).run()

        assert run.promoted.name == "runner-up"

    def test_a_deprecated_challenger_is_still_reported(self):
        run = Arbiter(
            challengers=[
                FakeModel("best-but-deprecated", rmse=0.50, mae=0.40, deprecated=True),
                FakeModel("runner-up", rmse=0.60, mae=0.50),
            ]
        ).run()

        assert run.reported["challengers"] == ["best-but-deprecated", "runner-up"]

    def test_a_deprecated_incumbent_forfeits_the_endpoint(self):
        # The incumbent is better on every metric, but deprecated, so it still loses
        run = Arbiter(
            challengers=[FakeModel("challenger", rmse=0.80, mae=0.70)],
            incumbent=FakeModel("old-champ", rmse=0.10, mae=0.05, deprecated=True),
        ).run()

        assert run.promoted.name == "challenger"
        assert run.models["old-champ"].deleted is True
        # A promotion reports the frozen copy as the new champion
        assert run.reported["champion"] == run.promoted.copied_to


class TestIncumbentComparison:
    """The best promotable challenger has to beat the model currently serving the endpoint."""

    def test_a_worse_challenger_does_not_take_the_endpoint(self):
        run = Arbiter(
            challengers=[FakeModel("challenger", rmse=0.90, mae=0.80)],
            incumbent=FakeModel("champ", rmse=0.50, mae=0.40),
        ).run()

        assert run.promoted is None
        assert run.reported["champion"] == "champ"

    def test_a_better_challenger_takes_the_endpoint_and_retires_the_incumbent(self):
        run = Arbiter(
            challengers=[FakeModel("challenger", rmse=0.30, mae=0.20)],
            incumbent=FakeModel("champ", rmse=0.50, mae=0.40),
        ).run()

        assert run.promoted.name == "challenger"
        assert run.models["champ"].deleted is True

    def test_a_challenger_better_on_only_one_metric_does_not_win(self):
        # beats() requires both rmse and mae to improve
        run = Arbiter(
            challengers=[FakeModel("challenger", rmse=0.30, mae=0.90)],
            incumbent=FakeModel("champ", rmse=0.50, mae=0.40),
        ).run()

        assert run.promoted is None

    def test_no_incumbent_promotes_the_best_promotable(self):
        run = Arbiter(
            challengers=[FakeModel("winner", rmse=0.50, mae=0.40)],
            endpoint_exists=False,
        ).run()

        assert run.promoted.name == "winner"
        assert run.reported["champion"] == run.promoted.copied_to


class TestScoring:
    """Challengers without metrics are skipped; a cohort with none promotes nothing."""

    def test_a_challenger_without_metrics_is_skipped(self):
        run = Arbiter(
            challengers=[FakeModel("no-metrics"), FakeModel("has-metrics", rmse=0.60, mae=0.50)],
        ).run()

        assert run.promoted.name == "has-metrics"
        assert run.reported["challengers"] == ["has-metrics"]

    def test_a_cohort_with_no_metrics_promotes_nothing_and_reports_nothing(self):
        run = Arbiter(challengers=[FakeModel("no-metrics"), FakeModel("also-none")]).run()

        assert run.promoted is None
        assert run.report_calls == []

    def test_classifiers_rank_on_highest_f1(self):
        run = Arbiter(
            challengers=[FakeModel("low-f1", f1=0.60), FakeModel("high-f1", f1=0.90)],
        ).run()

        assert run.promoted.name == "high-f1"


class TestPromotionDeploy:
    """The frozen copy is what gets deployed, and it populates its own metrics."""

    def test_the_frozen_copy_is_deployed_and_runs_inference(self):
        run = Arbiter(challengers=[FakeModel("challenger", rmse=0.50, mae=0.40)]).run()

        # The frozen copy takes the endpoint, not the challenger itself
        frozen = run.promoted.frozen
        assert frozen.name == run.promoted.copied_to
        assert frozen.deployed_to == ENDPOINT
        assert run.promoted.deployed_to is None
        assert frozen.deploy.test_inference_calls == 1
        assert frozen.deploy.cross_fold_calls == 1

    def test_a_dt_suffix_is_dropped_from_the_dated_name(self):
        run = Arbiter(challengers=[FakeModel("challenger-dt", rmse=0.50, mae=0.40)]).run()

        assert run.promoted.copied_to.startswith("challenger-")
        assert "-dt-" not in run.promoted.copied_to


class TestBeats:
    """The comparison itself: both metrics must improve (classifier: f1 alone)."""

    @pytest.mark.parametrize(
        "challenger, incumbent, expected",
        [
            ({"rmse": 0.4, "mae": 0.3}, {"rmse": 0.5, "mae": 0.4}, True),
            ({"rmse": 0.6, "mae": 0.5}, {"rmse": 0.5, "mae": 0.4}, False),
            ({"rmse": 0.4, "mae": 0.5}, {"rmse": 0.5, "mae": 0.4}, False),
            ({"rmse": 0.5, "mae": 0.4}, {"rmse": 0.5, "mae": 0.4}, False),
        ],
    )
    def test_regressor_needs_both_metrics_better(self, challenger, incumbent, expected):
        assert beats(challenger, incumbent, is_classifier=False) is expected

    @pytest.mark.parametrize(
        "challenger, incumbent, expected",
        [({"f1": 0.9}, {"f1": 0.8}, True), ({"f1": 0.7}, {"f1": 0.8}, False), ({"f1": 0.8}, {"f1": 0.8}, False)],
    )
    def test_classifier_compares_f1(self, challenger, incumbent, expected):
        assert beats(challenger, incumbent, is_classifier=True) is expected

    def test_no_incumbent_is_always_beaten(self):
        assert beats({"rmse": 9.9, "mae": 9.9}, None, is_classifier=False) is True
