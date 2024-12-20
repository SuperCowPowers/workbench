from workbench.api.feature_set import FeatureSet
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("wine_features")

# Using a Scikit-Learn Model
# Note: scikit_model_class can be any sckit-learn model ("KMeans", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
my_model = my_features.to_model(
    scikit_model_class="RandomForestClassifier",
    model_import_str="from sklearn.ensemble import RandomForestClassifier",
    target_column="wine_class",
    name="wine-rfc-class",
    description="Wine RandomForest Classification",
    tags=["wine", "rfc"],
)
pprint(my_model.details())
