from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelFramework
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("abalone_features")

# Transform FeatureSet into KNN Regression Model
# Note: model_class can be any sckit-learn model ("KMeans", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
tags = ["abalone", "clusters"]
my_model = my_features.to_model(
    model_class="DBSCAN",
    model_import_str="from sklearn.cluster import DBSCAN",
    model_framework=ModelFramework.XGBOOST,
    target_column="class_number_of_rings",
    name="abalone-clusters",
    description="Abalone DBSCAN Clustering",
    tags=tags,
    train_all_data=True,
)
pprint(my_model.details())
my_model = Model("abalone-clusters")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(tags=tags)

# Run auto-inference on the Endpoint
endpoint.auto_inference()
