"""Default promotion arbiter (a "workbench:models/model_promotion.py" pipeline node).

Picks the best challenger for a champion endpoint and, if it beats the incumbent,
freezes it to a dated copy and deploys it onto the endpoint. The challengers (the
``-dt`` models) and the champion ``endpoint_name`` come from PipelineMeta, as does the
``no_promote`` subset: those challengers are ranked and reported but never take the
endpoint, so the best *promotable* challenger is the one that faces the incumbent. A
model with ``deprecated`` set in its workbench_meta is treated the same way, and a
deprecated incumbent forfeits the endpoint.
Each promotion decision publishes the contest results to Reports() at /contests/<endpoint>.

This is a deliberately simple reference: no thresholds, notifications, or config --
clients override it with a ``plugin:`` script for custom promotion policy.
"""

import logging
from datetime import datetime

from workbench.api import Model, Endpoint, ModelType, Reports
from workbench.core.pipelines.pipeline_meta import PipelineMeta
from workbench.utils.model_comparison import contest_report

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


def publish_contest_report(endpoint_name: str, champion: Model | None, challengers: list, no_promote: list):
    """Publish the contest results to Reports() at /contests/<endpoint>. Never raises:
    the promotion decision has already been made, so a report failure is logged and swallowed."""
    try:
        report = contest_report(champion, challengers, endpoint_name, no_promote=no_promote)
        if report is not None:
            Reports().upsert(f"/contests/{endpoint_name}", report)
            log.important(f"Published contest report /contests/{endpoint_name}")
    except Exception as e:
        log.error(f"Failed to publish contest report for {endpoint_name}: {e}")


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

    # Rank challengers (classifier: highest f1; regressor: lowest rmse)
    is_classifier = scored[0][0].model_type == ModelType.CLASSIFIER
    scored.sort(key=(lambda s: -s[1]["f1"]) if is_classifier else (lambda s: s[1]["rmse"]))
    challengers = [model for model, _ in scored]
    no_promote = set(pm.no_promote)

    # The model currently serving the champion endpoint (a deprecated one forfeits it)
    end = Endpoint(endpoint_name)
    dethroned = end.get_input() if end.exists() else None
    incumbent = Model(dethroned) if dethroned and dethroned != "unknown" and Model(dethroned).exists() else None
    if incumbent and incumbent.workbench_meta().get("deprecated"):
        log.important(f"Incumbent {dethroned} is deprecated; it forfeits {endpoint_name}")
        incumbent = None
    incumbent_metrics = primary_metrics(incumbent) if incumbent else None

    # The best *promotable* challenger takes on the incumbent; no_promote and deprecated
    # models are ranked and reported, but can never take the endpoint.
    eligible = [s for s in scored if s[0].name not in no_promote and not s[0].workbench_meta().get("deprecated")]
    if not eligible:
        log.important(f"Every challenger for {endpoint_name} is no_promote or deprecated; champion stays")
        publish_contest_report(endpoint_name, incumbent, challengers, pm.no_promote)
        return
    winner, winner_metrics = eligible[0]
    if winner is not scored[0][0]:
        log.important(f"Top-ranked {scored[0][0].name} is not promotable; best promotable is {winner.name}")
    log.important(f"Best challenger for {endpoint_name}: {winner.name} {winner_metrics}")

    if not beats(winner_metrics, incumbent_metrics, is_classifier):
        log.important(f"Champion stays: {winner.name} does not beat incumbent {incumbent_metrics}")
        publish_contest_report(endpoint_name, incumbent, challengers, pm.no_promote)
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

    publish_contest_report(endpoint_name, frozen, challengers, pm.no_promote)


if __name__ == "__main__":
    main()
