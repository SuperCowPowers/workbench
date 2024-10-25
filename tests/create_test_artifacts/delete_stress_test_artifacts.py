"""This Script Deletes the SageWorks Artifacts used for the tests"""

import time
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Generated Model Names
    model_names = [f"test-model-{i}" for i in range(16)]
    feature_names = [f"{model.replace('-', '_')}_features" for model in model_names]
    endpoint_names = [f"{model}-end" for model in model_names]

    # Delete the FeatureSets
    for feature_name in feature_names:
        FeatureSet.delete(feature_name)

    # Delete the Models
    for model_name in model_names:
        Model.delete(model_name)

    # Delete the Endpoints
    for endpoint_name in endpoint_names:
        Endpoint.delete(endpoint_name)

    time.sleep(5)
    print("All stress test artifacts should now be deleted!")
