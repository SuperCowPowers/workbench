"""ModelData Class for storing and managing the data associated with model"""
import pandas as pd
import json


class ModelData:
    """ModelData Class for storing and managing the data associated with model"""

    def __init__(self, path):
        self._data = pd.read_csv(path, parse_dates=["date_created"])

    def get_model_df(self) -> pd.DataFrame:
        """Return the underlying Pandas Dataframe"""
        return self._data

    def get_model_details(self, row_index):
        """Get the model details for this model (via row_index)"""
        row = self._data.iloc[row_index]
        return row.to_dict()

    def get_model_confusion_matrix(self, row_index):
        """Get the model confusion matrix for this model (via row_index)"""
        row = self._data.iloc[row_index]
        return json.loads(row["c_matrix"])

    def get_model_feature_importance(self, row_index):
        """Get the model feature importance for this model (via row_index)"""
        if not isinstance(row_index, int):
            raise RuntimeError(f"row_index must be an integer not {type(row_index)}: {row_index}")

        row = self._data.iloc[row_index]
        return json.loads(row["feature_importance"])


# Test for our ModelData Class
if __name__ == "__main__":
    file_path = "/Users/briford/work/sageworks/applications/artifact_viewer/pages/data/toy_data.csv"
    model_data = ModelData(file_path)
    print(model_data.get_model_feature_importance(0))
    print(model_data.get_model_confusion_matrix(1))
