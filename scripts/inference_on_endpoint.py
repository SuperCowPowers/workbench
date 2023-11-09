# Inference on an Endpoint
#
# - Pull data from a Test DataSet
# - Run inference on an Endpoint
# - Capture performance metrics in S3 SageWorks Model Bucket

from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint
import awswrangler as wr

# Test DatsSet
# There's a couple of options here:
# 1. Grab a SageWorks FeatureSet and pull data from it
# 2. Pull the data from S3 directly

# S3_DATA_PATH = "s3://sageworks-data-science-dev/data-sources/abalone_holdout_2023_10_19.csv"
S3_DATA_PATH = None
FEATURE_SET_NAME = "aqsol_features"
FEATURE_SET_NAME = "hlm_phase2_reg_0_230830"
FEATURE_SET_NAME = "stab_phase_2_features"
FEATURE_SET_NAME = "wine_features"
FEATURE_SET_NAME = "solubility_test_features"
FEATURE_SET_NAME = "abalone_feature_set"
# FEATURE_SET_NAME = None

ENDPOINT_NAME = "aqsol-solubility-regression-end"
ENDPOINT_NAME = "hlm-phase2-reg-0-230830-test-endpoint"
ENDPOINT_NAME = "stab-regression-end"
ENDPOINT_NAME = "wine-classification-end"
ENDPOINT_NAME = "solubility-test-regression-end"
ENDPOINT_NAME = "abalone-regression-end"

# These should be filled in
DATA_NAME = "stab_phase_2_features(20)"
DATA_HASH = "12345"
DESCRIPTION = "Test Stability Phase2 Features"
TARGET_COLUMN = "stability"
DATA_NAME = "wine_features (20)"
DATA_HASH = "12345"
DESCRIPTION = "Test Wine Features"
TARGET_COLUMN = "wine_class"

DATA_NAME = "solubility_test_features (20)"
DATA_HASH = "12345"
DESCRIPTION = "Test Solubility Features"
TARGET_COLUMN = "log_s"

DATA_NAME = "abalone_feature_set (20)"
DATA_HASH = "12345"
DESCRIPTION = "Test Abalone Features"
TARGET_COLUMN = "class_number_of_rings"


if S3_DATA_PATH is not None:
    # Read the data from S3
    df = wr.s3.read_csv(S3_DATA_PATH)
else:
    # Grab the FeatureSet
    features = FeatureSet(FEATURE_SET_NAME)
    table = features.training_view.table_name
    df = features.query(f"SELECT * FROM {table} where training = 0")

# Spin up our Endpoint
my_endpoint = Endpoint(ENDPOINT_NAME, force_refresh=True)

# Capture the performance metrics for this Endpoint
my_endpoint.capture_performance_metrics(
    df, TARGET_COLUMN, data_name=DATA_NAME, data_hash=DATA_HASH, description=DESCRIPTION
)
