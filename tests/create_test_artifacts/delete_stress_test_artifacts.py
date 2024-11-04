"""This Script Deletes the SageWorks Artifacts used for stress tests"""

import time
from sageworks.api import DataSource, FeatureSet, Model, Endpoint


if __name__ == "__main__":

    # Generated Model Names
    model_names = [f"stress-model-{i}" for i in range(16)]
    datasource_names = [f"{model.replace('-model-', '_data_')}" for model in model_names]
    feature_names = [f"{model.replace('-model-', '_features_')}" for model in model_names]
    endpoint_names = [f"{model}-end" for model in model_names]

    # Delete the DataSources
    for datasource_name in datasource_names:
        DataSource.managed_delete(datasource_name)

    # Delete the FeatureSets
    for feature_name in feature_names:
        FeatureSet.managed_delete(feature_name)

    # Delete the Models
    for model_name in model_names:
        Model.managed_delete(model_name)

    # Delete the Endpoints
    for endpoint_name in endpoint_names:
        Endpoint.managed_delete(endpoint_name)

    time.sleep(5)
    print("All stress test artifacts should now be deleted!")
