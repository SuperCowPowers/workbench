"""This Script publishes the Local Artifacts to AWS and checks they reproduce

Run create_local_artifacts.py first. Publishing cascades up the lineage, so this
creates whatever AWS does not already have:

DataSources:
    - local_test_data
FeatureSets:
    - local_test_features
Models:
    - local-test-regression      (retrained in AWS from the published FeatureSet)
Endpoints:
    - local-test-regression-end  (serverless)

The point of this script is the check at the end: the AWS model is trained by the
same generated script, on the same rows, with the same row roles, so its
out-of-fold predictions should track the local ones closely. A wide gap means
something diverged between local and AWS.
"""

import logging

import numpy as np

from workbench.api import Endpoint
from workbench.local import LocalModel

# Setup the logger
log = logging.getLogger("workbench")

# Correlation below this between local and AWS out-of-fold predictions means the
# two runs disagree by more than fold-shuffling noise explains
MIN_CORRELATION = 0.95


if __name__ == "__main__":

    model = LocalModel("local-test-regression")
    if model.training_state().get("state") != "completed":
        raise SystemExit("Run create_local_artifacts.py first (local model has not trained)")

    # Package versions the AWS run will use, which is what makes reproduction plausible
    drift = model.version_drift()
    log.important(f"Version drift: {drift or 'none'}")

    # Show the plan, then publish the whole lineage
    log.important("Publish plan:")
    for step in model.publish():
        log.important(f"  {step['action']:>6}  {step['type']:<12} {step['name']}")

    aws_model = model.publish(confirm=True)
    log.important(f"Published model: {aws_model.name} (exists: {aws_model.exists()})")

    # Deploy the endpoint (serverless, so it scales to zero)
    if Endpoint("local-test-regression-end").exists():
        aws_endpoint = Endpoint("local-test-regression-end")
    else:
        aws_endpoint = aws_model.to_endpoint(name="local-test-regression-end")

    # Pull the AWS model's out-of-fold predictions. This reads the oof_predictions.csv
    # the training job wrote, so it also confirms save_output's S3 DataFrame path.
    aws_oof = aws_endpoint.cross_fold_inference()
    if aws_oof is None or aws_oof.empty:
        log.warning("No AWS cross-fold predictions, skipping the comparison")
        raise SystemExit(0)
    log.important(f"AWS cross-fold inference: {len(aws_oof)} rows")

    # Compare out-of-fold predictions: same script, same rows, same roles
    local_oof = model.oof_predictions()

    merged = local_oof.merge(aws_oof, on="id", suffixes=("_local", "_aws"))
    log.important(f"Compared {len(merged)} rows (local {len(local_oof)}, aws {len(aws_oof)})")

    correlation = merged["prediction_local"].corr(merged["prediction_aws"])
    max_diff = (merged["prediction_local"] - merged["prediction_aws"]).abs().max()
    mean_diff = np.abs(merged["prediction_local"] - merged["prediction_aws"]).mean()

    log.important(f"Prediction correlation: {correlation:.4f}")
    log.important(f"Mean absolute difference: {mean_diff:.4f}")
    log.important(f"Max absolute difference: {max_diff:.4f}")

    if correlation < MIN_CORRELATION:
        log.critical(f"Local and AWS predictions diverged (correlation {correlation:.4f} < {MIN_CORRELATION})")
    else:
        log.important("Local and AWS models agree")
