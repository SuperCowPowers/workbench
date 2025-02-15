"""InferenceView Class: A View that does endpoint inference and computes residuals"""

from typing import Union
import logging

# Workbench Imports
from workbench.api import FeatureSet, Model, ModelType, Endpoint
from workbench.core.views.view import View
from workbench.core.views.pandas_to_view import PandasToView

# Set up logging
log = logging.getLogger("workbench")


class InferenceView:
    """InferenceView Class: A View that does endpoint inference and computes residuals

    Common Usage:
        ```python
        # Grab a Model
        model = Model("abalone-regression")

        # Create an InferenceView
        inf_view = InferenceView.create(model)
        my_df = inf_view.pull_dataframe(limit=5)

        # Query the view
        df = inf_view.query(f"SELECT * FROM {inf_view.table} where residuals > 0.5")
        ```
    """

    @classmethod
    def create(
        cls,
        model: Model,
    ) -> Union[View, None]:
        """Create a View that does endpoint inference and computes residuals

        Args:
            model (Model): The Model object to use for the target and features

        Returns:
            Union[View, None]: The created View object (or None if failed)
        """
        # Log view creation
        log.important("Creating Inference View...")

        # Pull in data from the FeatureSet
        fs = FeatureSet(model.get_input())
        df = fs.pull_dataframe()

        # Grab the target from the model
        target = model.target()

        # Run inference on the data
        end = Endpoint(model.endpoints()[0])
        df = end.inference(df)

        # Determine if the target is a classification or regression target
        if model.model_type == ModelType.REGRESSOR:
            df["residuals"] = df[target] - df["prediction"]
            df["residuals_abs"] = df["residuals"].abs()
        elif model.model_type == ModelType.CLASSIFIER:
            class_labels = model.class_labels()
            class_index = {label: i for i, label in enumerate(class_labels)}
            df["residuals"] = df["prediction"].map(class_index) - df[target].map(class_index)
            df["residuals_abs"] = df["residuals"].abs()
        else:
            log.warning(f"Model type {model.model_type} has undefined residuals computation")
            df["residuals"] = 0
            df["residuals_abs"] = 0

        # Save the inference results to an inference view
        view_name = f"inf_{model.uuid.replace('-', '_')}"
        return PandasToView.create(view_name, fs, df=df, id_column=fs.id_column)


if __name__ == "__main__":
    """Exercise the InferenceView functionality"""

    # Grab a classification Model
    my_model = Model("wine-classification")

    # Create an Inference View
    inf_view = InferenceView.create(my_model)

    # Pull the inference dataframe
    my_df = inf_view.pull_dataframe(limit=5)
    print(my_df)

    # Query the view
    df = inf_view.query(f"SELECT * FROM {inf_view.table} where residuals_abs > 0.5")
    print(df)

    # Grab a Regression Model
    my_model = Model("abalone-regression")
    inf_view = InferenceView.create(my_model)
    my_df = inf_view.pull_dataframe(limit=5)
    print(my_df)
    df = inf_view.query(f"SELECT * FROM {inf_view.table} where residuals_abs > 0.5")
    print(df)
