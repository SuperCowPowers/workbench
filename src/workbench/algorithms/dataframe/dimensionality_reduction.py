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


class DimensionalityReduction:
    """Perform Dimensionality Reduction on a DataFrame using TSNE, MDS, PCA, or UMAP."""

    def __init__(self):
        """Initialize the DimensionalityReduction class."""
        self.log = logging.getLogger("workbench")
        self.projection_model = None

    def fit_transform(self, df: pd.DataFrame, features: list = None, projection: str = "TSNE") -> pd.DataFrame:
        """Fit and transform a DataFrame using the selected dimensionality reduction method.

        Args:
            df (pd.DataFrame): The DataFrame containing features to project.
            features (list, optional): List of feature column names. If None, numeric columns are auto-selected.
            projection (str, optional): The projection model to use ('TSNE', 'MDS', 'PCA', or 'UMAP'). Defaults to 'TSNE'.

        Returns:
            pd.DataFrame: DataFrame with new columns 'x' and 'y' containing the projected 2D coordinates.
        """

        # Auto-identify numeric features if none are provided
        if features is None:
            features = [col for col in df.select_dtypes(include="number").columns if not col.endswith("id")]
            self.log.info(f"Auto-identified numeric features: {features}")

        if len(features) < 2 or df.empty:
            self.log.critical("At least two numeric features are required, and DataFrame must not be empty.")
            return df

        # Fill NaNs with column mean and normalize features
        df.loc[:, features] = df[features].apply(lambda col: col.fillna(col.mean()))
        df.loc[:, features] = StandardScaler().fit_transform(df[features])

        # Select projection method
        self.projection_model = self._get_projection_model(projection, df)

        # Apply projection
        projection_result = self.projection_model.fit_transform(df[features])
        df[["x", "y"]] = projection_result

        # Resolve coincident points by adding jitter
        return self.resolve_coincident_points(df)

    def _get_projection_model(self, projection: str, df: pd.DataFrame):
        """Select and return the appropriate projection model.

        Args:
            projection (str): The projection method ('TSNE', 'MDS', 'PCA', or 'UMAP').
            df (pd.DataFrame): The DataFrame being transformed.

        Returns:
            A fitted dimensionality reduction model instance.
        """
        if projection == "TSNE":
            perplexity = min(40, len(df) - 1)
            self.log.info(f"Using TSNE with perplexity {perplexity}")
            return TSNE(perplexity=perplexity)

        if projection == "MDS":
            self.log.info("Using MDS")
            return MDS(n_components=2, random_state=0)

        if projection == "PCA":
            self.log.info("Using PCA")
            return PCA(n_components=2)

        if projection == "UMAP" and UMAP_AVAILABLE:
            self.log.info("Using UMAP")
            return umap.UMAP(n_components=2)

        self.log.warning(f"Projection method '{projection}' not recognized or UMAP not available. Falling back to TSNE.")
        return TSNE(perplexity=min(40, len(df) - 1))

    @staticmethod
    def resolve_coincident_points(df: pd.DataFrame) -> pd.DataFrame:
        """Resolve coincident points in a DataFrame by adding jitter.

        Args:
            df (pd.DataFrame): The DataFrame containing x and y projection coordinates.

        Returns:
            pd.DataFrame: The DataFrame with resolved coincident points.
        """
        jitter_x = (df["x"].max() - df["x"].min()) * 0.1
        jitter_y = (df["y"].max() - df["y"].min()) * 0.1
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
        "feat3": [0.1, 0.1, 0.2, 1.6, 2.5, 0.1, 0.1, 0.2, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    df = pd.DataFrame(data)

    projection = DimensionalityReduction().fit_transform(df, features=["feat1", "feat2", "feat3"], projection="UMAP")
    print(projection)

    # Run the Unit Test on the Plugin
    unit_test = PluginUnitTest(
        ScatterPlot,
        input_data=df,
    )
    unit_test.run()
