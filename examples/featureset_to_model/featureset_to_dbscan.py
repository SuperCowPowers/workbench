from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("abalone_features")

# Transform FeatureSet into KNN Regression Model
# Note: scikit_model_class can be any sckit-learn model ("KMeans", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
my_model = my_features.to_model(
    scikit_model_class="DBSCAN",
    model_import_str="from sklearn.cluster import DBSCAN",
    target_column="class_number_of_rings",
    name="abalone-clusters",
    description="Abalone DBSCAN Clustering",
    tags=["abalone", "clusters"],
    train_all_data=True,
)
pprint(my_model.details())
my_model = Model("abalone-clusters")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(name="abalone-clusters", tags=["abalone", "clusters"])

# Run auto-inference on the Endpoint
endpoint.auto_inference(capture=True)
