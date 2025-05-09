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

        # Resolve coincident points and return the new DataFrame
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
        """Resolve coincident points using random jitter

        Args:
            df (pd.DataFrame): DataFrame with x and y coordinates.

        Returns:
            pd.DataFrame: DataFrame with resolved coincident points
        """

        # Set jitter size based on rounding precision
        precision = 3
        jitter_amount = 10 ** (-precision) * 2  # 2x the rounding precision

        # Create rounded values for grouping
        rounded = pd.DataFrame(
            {"x_round": df["x"].round(precision), "y_round": df["y"].round(precision), "idx": df.index}
        )

        # Find duplicates
        duplicated = rounded.duplicated(subset=["x_round", "y_round"], keep=False)
        print("Coincident Points found:", duplicated.sum())
        if not duplicated.any():
            return df

        # Get the dtypes of the columns
        x_dtype = df["x"].dtype
        y_dtype = df["y"].dtype

        # Process each group
        for (x_round, y_round), group in rounded[duplicated].groupby(["x_round", "y_round"]):
            indices = group["idx"].values
            if len(indices) <= 1:
                continue

            # Apply random jitter to all points
            for i, idx in enumerate(indices):
                # Generate and apply properly typed offsets
                dx = np.array(jitter_amount * (np.random.random() * 2 - 1), dtype=x_dtype)
                dy = np.array(jitter_amount * (np.random.random() * 2 - 1), dtype=y_dtype)
                df.loc[idx, "x"] += dx
                df.loc[idx, "y"] += dy

        return df


if __name__ == "__main__":
    """Exercise the Dimensionality Reduction."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
    from workbench.api import FeatureSet, Model, Endpoint, DFStore
    from workbench.utils.shap_utils import shap_feature_importance

    data = {
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [2.1, 2.2, 2.4, 6.1, 9.0, 2.5, 2.0, 2.2, 6.6, 7.5],
        "feat3": [0.1, 0.15, 0.2, 0.9, 2.8, 0.25, 0.35, 0.4, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    input_df = pd.DataFrame(data)
    # Concat a bunch of copies to test out the coincidence resolution
    input_df = pd.concat([input_df] * 10, ignore_index=True)
    input_df["ID"] = [f"id_{i}" for i in range(len(input_df))]

    df = Projection2D().fit_transform(input_df, features=["feat1", "feat2", "feat3"], projection="UMAP")
    print(df)

    # Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
    # unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y")
    # unit_test.run()

    # Okay now for a real test with real data
    model = Model("aqsol-ensemble")

    # Pull a FeatureSet and run inference on it
    recreate = False
    if recreate:
        fs = FeatureSet(model.get_input())
        df = fs.pull_dataframe()
        end = Endpoint(model.endpoints()[0])
        df = end.inference(df)

        # Store the inference dataframe
        DFStore().upsert("/workbench/models/aqsol-ensemble/full_inference", df)
    else:
        # Retrieve the cached inference dataframe
        df = DFStore().get("/workbench/models/aqsol-ensemble/full_inference")
        if df is None:
            raise ValueError("No cached inference DataFrame found.")

    # Compute SHAP values and get the top 10 features
    shap_importances = shap_feature_importance(model)[:10]
    shap_features = [feature for feature, _ in shap_importances]
    df = Projection2D().fit_transform(df, features=shap_features, projection="UMAP")

    # Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
    unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y")
    unit_test.run()
