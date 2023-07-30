"""DimensionalityReduction: Perform Dimensionality Reduction on a DataFrame"""
import numpy as np
import pandas as pd
import logging
from sklearn.manifold import TSNE

# SageWorks Imports
from sageworks.utils.sageworks_logging import logging_setup

logging_setup()


# Dimensionality Reduction Class
class DimensionalityReduction:
    def __init__(self):
        """DimensionalityReduction:  Perform Dimensionality Reduction on a DataFrame"""
        self.log = logging.getLogger(__name__)
        self.projection_model = None
        self.features = None

    def fit_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Fit and Transform the DataFrame
        Args:
            df: Pandas DataFrame
            features: List of feature column names
        Returns:
            Pandas DataFrame with new columns x and y
        """

        # Sanity checks
        if not all(column in df.columns for column in features):
            self.log.critical("Some features are missing in the DataFrame")
            return df
        if len(features) < 2:
            self.log.critical("At least two features are required")
            return df
        if df.empty:
            self.log.critical("DataFrame is empty")
            return df

        # Project the multidimensional features onto an x,y plane
        self.log.info("Projecting features onto an x,y plane...")

        # Perplexity is a hyperparameter that controls the number of neighbors used to compute the manifold
        # The number of neighbors should be less than the number of samples
        perplexity = min(40, len(df) - 1)
        self.log.info(f"Perplexity: {perplexity}")
        self.projection_model = TSNE(perplexity=perplexity)
        self.features = features
        projection = self.projection_model.fit_transform(df[self.features])

        # Put the projection results back into the given DataFrame
        df["x"] = projection[:, 0]  # Projection X Column
        df["y"] = projection[:, 1]  # Projection Y Column

        # Return the DataFrame with the new columns
        return df

    @staticmethod
    def resolve_coincident_points(df: pd.DataFrame):
        """Resolve coincident points in a DataFrame
        Args:
            df(pd.DataFrame): The DataFrame to resolve coincident points in
        Returns:
            pd.DataFrame: The DataFrame with resolved coincident points
        """
        # Adding Jitter to the projection
        x_scale = (df["x"].max() - df["x"].min()) * 0.05
        y_scale = (df["y"].max() - df["y"].min()) * 0.05
        df["x"] += np.random.normal(-x_scale, +x_scale, len(df))
        df["y"] += np.random.normal(-y_scale, +y_scale, len(df))
        return df


def test():
    """Test for the Dimensionality Reduction Class"""
    # Set some pandas options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Make some fake data
    data = {
        "ID": ["id_0", "id_0", "id_2", "id_3", "id_4", "id_5", "id_6", "id_7", "id_8", "id_9"],
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat3": [0.1, 0.1, 0.2, 1.6, 2.5, 0.1, 0.1, 0.2, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    data_df = pd.DataFrame(data)
    features = ["feat1", "feat2", "feat3"]

    # Create the class and run the dimensionality reduction
    projection = DimensionalityReduction()
    new_df = projection.fit_transform(data_df, features)

    # Check that the x and y columns were added
    assert "x" in new_df.columns
    assert "y" in new_df.columns

    # Output the DataFrame
    print(new_df)


if __name__ == "__main__":
    test()
