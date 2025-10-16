import awswrangler as wr

# Workbench Imports
from workbench.core.transforms.pandas_transforms import PandasToFeatures

import multiprocessing
import multiprocess

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    multiprocess.set_start_method("spawn", force=True)

    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"
    aqsol_data = wr.s3.read_csv(s3_path)
    aqsol_data.columns = aqsol_data.columns.str.lower()

    to_features = PandasToFeatures("temp_delete_me")
    to_features.set_input(aqsol_data, id_column="id")
    to_features.set_output_tags(["aqsol", "public"])
    to_features.transform()
