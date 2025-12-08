import pandas as pd
from workbench.api import Model, Endpoint, DFStore

df_store = DFStore()

# List of all available models
model_list = [
    'caco-2-efflux-reg-chemprop',
    'caco-2-efflux-reg-chemprop-hybrid',
    'caco-2-efflux-reg-pytorch',
    'caco-2-efflux-reg-xgb',
    'caco-2-papp-a-b-reg-chemprop',
    'caco-2-papp-a-b-reg-chemprop-hybrid',
    'caco-2-papp-a-b-reg-pytorch',
    'caco-2-papp-a-b-reg-xgb',
    'hlm-clint-reg-chemprop',
    'hlm-clint-reg-chemprop-hybrid',
    'hlm-clint-reg-pytorch',
    'hlm-clint-reg-xgb',
    'ksol-reg-chemprop',
    'ksol-reg-chemprop-hybrid',
    'ksol-reg-pytorch',
    'ksol-reg-xgb',
    'logd-reg-chemprop',
    'logd-reg-chemprop-hybrid',
    'logd-reg-pytorch',
    'logd-reg-xgb',
    'mbpb-reg-chemprop',
    'mbpb-reg-chemprop-hybrid',
    'mbpb-reg-pytorch',
    'mbpb-reg-xgb',
    'mgmb-reg-chemprop',
    'mgmb-reg-chemprop-hybrid',
    'mgmb-reg-pytorch',
    'mgmb-reg-xgb',
    'mlm-clint-reg-chemprop',
    'mlm-clint-reg-chemprop-hybrid',
    'mlm-clint-reg-pytorch',
    'mlm-clint-reg-xgb',
    'mppb-reg-chemprop',
    'mppb-reg-chemprop-hybrid',
    'mppb-reg-pytorch',
    'mppb-reg-xgb'
]

xgb_models = [name for name in model_list if name.endswith("-xgb")]
pytorch_models = [name for name in model_list if name.endswith("-pytorch")]
chemprop_models = [name for name in model_list if name.endswith("-chemprop")]
chemprop_hybrid_models = [name for name in model_list if name.endswith("-chemprop-hybrid")]

# Grab test data
test_df = pd.read_csv("test_data_blind.csv")

# Hit Feature Endpoint
"""
rdkit_end = Endpoint("smiles-to-taut-md-stereo-v1")
df_features = rdkit_end.inference(test_df)

# Shove this into the DFStore for faster use later
df_store.upsert("/workbench/datasets/open_admet_test_featurized", df_features)
"""

# Grab featurized test data from DFStore
df_features = df_store.get("/workbench/datasets/open_admet_test_featurized")

# Run inference on some set of models
for model_name in xgb_models:
    model = Model(model_name)
    end = Endpoint(model_name)

    # Run inference on the AWS Endpoint
    result_df = end.inference(df_features)
    print(f"Inference results for model {model_name}:")
    print(result_df.head())
