"""This Script creates the Abalone Quantile Regression Artifacts in AWS/SageWorks

Models:
    - abalone-quantile-reg
Endpoints:
    - abalone-qr-end
"""

import logging
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

log = logging.getLogger("sageworks")


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the Abalone Quantile Regression Model
    if recreate or not Model("abalone-quantile-reg").exists():

        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            ModelType.REGRESSOR,
            target_column="class_number_of_rings",
            name="abalone-quantile-reg",
            description="Abalone Quantile Regression",
            tags=["abalone", "quantiles"],
        )

    # Create the Abalone Quantile Regression Endpoint
    if recreate or not Endpoint("abalone-qr-end").exists():
        m = Model("abalone-quantile-reg")
        m.to_endpoint(name="abalone-qr-end", tags=["abalone", "quantiles"])
