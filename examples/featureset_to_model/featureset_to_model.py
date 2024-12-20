from workbench.api.feature_set import FeatureSet
from workbench.api.model import ModelType
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("test_features")

# Create a Model from the FeatureSet
# Note: ModelTypes can be CLASSIFIER, REGRESSOR,
#       UNSUPERVISED, or TRANSFORMER
my_model = my_features.to_model(name="test-model", model_type=ModelType.REGRESSOR, target_column="iq_score")
pprint(my_model.details())
