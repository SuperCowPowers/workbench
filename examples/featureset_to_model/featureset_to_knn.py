from sageworks.api.feature_set import FeatureSet
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("abalone_features")

# Transform FeatureSet into KNN Regression Model
# Note: model_class can be any sckit-learn model ("KNeighborsRegressor", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
my_model = my_features.to_model(
    model_class="KNeighborsRegressor",
    target_column="class_number_of_rings",
    name="abalone-knn-reg",
    description="Abalone KNN Regression",
    tags=["abalone", "knn"],
    train_all_data=True,
)
pprint(my_model.details())
