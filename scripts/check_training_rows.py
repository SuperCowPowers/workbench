"""This script will check if the training rows have any overlap with the hold out ids

   Note: This is an example, feel free to change how the hold_out_ids are pulled
"""

from pprint import pprint
import awswrangler as wr
from sageworks.api import ParameterStore, FeatureSet

feature_set_to_check = "log_s_clean"
print(f"Checking the feature set: {feature_set_to_check}")

# Feel free to change this to whatever/however you want to get the hold outs
params = ParameterStore()
s3_path = params.get("/sageworks/nightly/sol_hold_out_path")
print(s3_path)
hold_out_ids = wr.s3.read_csv(s3_path, low_memory=False)["udm_mol_bat_id"].tolist()


# The training rows should not have any of the hold out ids
fs = FeatureSet(feature_set_to_check)
query = f"select udm_mol_bat_id from {fs.view('training').table} where training = 1"
training_ids = fs.query(query)["udm_mol_bat_id"].tolist()

# Check if there's any overlap
overlap = set(hold_out_ids).intersection(training_ids)
if overlap:
    print("Your training data has Overlap with the hold out ids!")
    print(f"Number of Overlapping ids: {len(overlap)}")
    pprint(list(overlap))
