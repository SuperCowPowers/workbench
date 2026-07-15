"""Default promotion arbiter (a "workbench:models/model_promotion.py" pipeline node).

Picks the best challenger for a champion endpoint and, if it beats the incumbent,
freezes it to a dated copy and deploys it onto the endpoint. The challengers (the
``-dt`` models) and the champion ``endpoint_name`` come from PipelineMeta.

This is a deliberately simple reference: no thresholds, notifications, or config --
clients override it with a ``plugin:`` script for custom promotion policy.
"""

import logging
from datetime import datetime

from workbench.api import Model, Endpoint, ModelType
from workbench.core.pipelines.pipeline_meta import PipelineMeta

log = logging.getLogger("workbench")


def primary_metrics(model: Model) -> dict | None:
    """The metrics we rank/compare on, by model type (None if unavailable).

    Regressor -> {rmse, mae} from the primary target row.
    Classifier -> {f1} from the 'all' summary row.
    """
    df = model.get_inference_metrics("full_cross_fold")
    if df is None or df.empty:
        return None
    if model.model_type == ModelType.CLASSIFIER:
        all_row = df[df.eq("all").any(axis=1)]  # the row labeled 'all'
        return {"f1": float(all_row.iloc[0]["f1"])} if not all_row.empty else None
    row = df.iloc[0]
    return {"rmse": float(row["rmse"]), "mae": float(row["mae"])}


def beats(challenger: dict, incumbent: dict | None, is_classifier: bool) -> bool:
    """Does the challenger beat the incumbent? No incumbent -> always True."""
    if incumbent is None:
        return True
    if is_classifier:
        return challenger["f1"] > incumbent["f1"]
    return challenger["rmse"] < incumbent["rmse"] and challenger["mae"] < incumbent["mae"]


def main():
    pm = PipelineMeta()
    endpoint_name = pm.endpoint_name

    # Score each challenger; drop any without full_cross_fold metrics
    scored = []
    for name in pm.challengers:
        model = Model(name)
        metrics = primary_metrics(model)
        if metrics is None:
            log.warning(f"No full_cross_fold metrics for {name}, skipping")
            continue
        scored.append((model, metrics))
    if not scored:
        log.error("No challengers with metrics; nothing to promote")
        return

    # Rank challengers (classifier: highest f1; regressor: lowest rmse) and take the best
    is_classifier = scored[0][0].model_type == ModelType.CLASSIFIER
    scored.sort(key=(lambda s: -s[1]["f1"]) if is_classifier else (lambda s: s[1]["rmse"]))
    winner, winner_metrics = scored[0]
    log.important(f"Best challenger for {endpoint_name}: {winner.name} {winner_metrics}")

    # Compare against the model currently serving the champion endpoint
    end = Endpoint(endpoint_name)
    dethroned = end.get_input() if end.exists() else None
    incumbent_metrics = None
    if dethroned and dethroned != "unknown" and Model(dethroned).exists():
        incumbent_metrics = primary_metrics(Model(dethroned))

    if not beats(winner_metrics, incumbent_metrics, is_classifier):
        log.important(f"Champion stays: {winner.name} does not beat incumbent {incumbent_metrics}")
        return

    # Promote: freeze a dated copy of the winner, deploy it onto the champion endpoint,
    # then retire the dethroned model (current-only retention).
    dated_name = f"{winner.name.removesuffix('-dt')}-{datetime.now().strftime('%y%m%d')}"
    log.important(f"Promoting {winner.name} -> {dated_name} on endpoint {endpoint_name}")
    frozen = winner.copy(dated_name, owner="Pro")
    end = frozen.to_endpoint(endpoint_name)

    # Populate test and full_cross_fold metrics on the new endpoint
    end.test_inference()
    end.cross_fold_inference()

    if dethroned and dethroned not in ("unknown", dated_name):
        log.important(f"Retiring dethroned model {dethroned}")
        Model(dethroned).delete()


if __name__ == "__main__":
    main()
