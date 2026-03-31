# Workbench Imports
from workbench.api import PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures

import multiprocessing
import multiprocess

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    multiprocess.set_start_method("spawn", force=True)

    aqsol_data = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    aqsol_data.columns = aqsol_data.columns.str.lower()

    to_features = PandasToFeatures("temp_delete_me")
    to_features.set_input(aqsol_data, id_column="id")
    to_features.set_output_tags(["aqsol", "public"])
    to_features.transform()
