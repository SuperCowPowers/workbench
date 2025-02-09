import numpy as np
import pandas as pd
import logging
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Try importing UMAP with a fallback to TSNE
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class Projection2D:
    """Perform Dimensionality Reduction on a DataFrame using TSNE, MDS, PCA, or UMAP."""

    def __init__(self):
        """Initialize the Projection2D class."""
        self.log = logging.getLogger("workbench")
        self.projection_model = None

    def fit_transform(self, input_df: pd.DataFrame, features: list = None, projection: str = "UMAP") -> pd.DataFrame:
        """Fit and transform a DataFrame using the selected dimensionality reduction method.

        This method creates a copy of the input DataFrame, processes the specified features
        for normalization and projection, and returns a new DataFrame with added 'x' and 'y' columns
        containing the projected 2D coordinates.

        Args:
            input_df (pd.DataFrame): The DataFrame containing features to project.
            features (list, optional): List of feature column names. If None, numeric columns are auto-selected.
            projection (str, optional): The projection to use ('UMAP', 'TSNE', 'MDS' or 'PCA'). Default 'UMAP'.

        Returns:
            pd.DataFrame: A new DataFrame (a copy of input_df) with added 'x' and 'y' columns.
        """
        # Create a copy of the input DataFrame
        df = input_df.copy()

        # Auto-identify numeric features if none are provided
        if features is None:
            features = [col for col in df.select_dtypes(include="number").columns if not col.endswith("id")]
            self.log.info(f"Auto-identified numeric features: {features}")

        if len(features) < 2 or df.empty:
            self.log.critical("At least two numeric features are required, and DataFrame must not be empty.")
            return df

        # Process a copy of the feature data for projection
        X = df[features]
        X = X.apply(lambda col: col.fillna(col.mean()))
        X_scaled = StandardScaler().fit_transform(X)

        # Select the projection method (using df for perplexity calculation)
        self.projection_model = self._get_projection_model(projection, df)

        # Apply the projection on the normalized data
        projection_result = self.projection_model.fit_transform(X_scaled)
        df[["x", "y"]] = projection_result

        # Resolve coincident points by adding jitter and return the new DataFrame
        return self.resolve_coincident_points(df)

    def _get_projection_model(self, projection: str, df: pd.DataFrame):
        """Select and return the appropriate projection model.

        Args:
            projection (str): The projection method ('TSNE', 'MDS', 'PCA', or 'UMAP').
            df (pd.DataFrame): The DataFrame being transformed (used for computing perplexity).

        Returns:
            A dimensionality reduction model instance.
        """
        if projection == "TSNE":
            perplexity = min(40, len(df) - 1)
            self.log.info(f"Projection: TSNE with perplexity {perplexity}")
            return TSNE(perplexity=perplexity)

        if projection == "MDS":
            self.log.info("Projection: MDS")
            return MDS(n_components=2, random_state=0)

        if projection == "PCA":
            self.log.info("Projection: PCA")
            return PCA(n_components=2)

        if projection == "UMAP" and UMAP_AVAILABLE:
            self.log.info("Projection: UMAP")
            return umap.UMAP(n_components=2)

        self.log.warning(
            f"Projection method '{projection}' not recognized or UMAP not available. Falling back to TSNE."
        )
        return TSNE(perplexity=min(40, len(df) - 1))

    @staticmethod
    def resolve_coincident_points(df: pd.DataFrame) -> pd.DataFrame:
        """Resolve coincident points in a DataFrame by adding jitter.

        Args:
            df (pd.DataFrame): The DataFrame containing x and y projection coordinates.

        Returns:
            pd.DataFrame: The DataFrame with resolved coincident points.
        """
        jitter_x = (df["x"].max() - df["x"].min()) * 0.005
        jitter_y = (df["y"].max() - df["y"].min()) * 0.005
        df["x"] += np.random.normal(0, jitter_x, len(df))
        df["y"] += np.random.normal(0, jitter_y, len(df))
        return df


if __name__ == "__main__":
    """Exercise the Dimensionality Reduction."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    data = {
        "ID": [f"id_{i}" for i in range(10)],
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat3": [0.1, 0.15, 0.2, 0.9, 2.8, 0.25, 0.35, 0.4, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    input_df = pd.DataFrame(data)

    df = Projection2D().fit_transform(input_df, features=["feat1", "feat2", "feat3"], projection="UMAP")
    print(df)

    # Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
    unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y")
    unit_test.run()
