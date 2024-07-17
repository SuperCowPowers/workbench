from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("abalone_features")

# Transform FeatureSet into KNN Regression Model
# Note: model_class can be any sckit-learn model ("KNeighborsRegressor", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
my_model = my_features.to_model(
    model_class="DBSCAN",
    target_column="class_number_of_rings",
    name="abalone-clusters",
    description="Abalone DBSCAN Clustering",
    tags=["abalone", "clusters"],
    train_all_data=True,
)
pprint(my_model.details())
my_model = Model("abalone-clusters")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(name="abalone-clusters-end", tags=["abalone", "clusters"])

# Run auto-inference on the Endpoint
endpoint.auto_inference(capture=True)
