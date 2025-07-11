"""A Markdown Component for details/information about Models"""

from typing import Union

# Dash Imports
from dash import html, callback, dcc, Input, Output, State

# Workbench Imports
from workbench.api import ModelType, ParameterStore
from workbench.cached.cached_model import CachedModel
from workbench.utils.markdown_utils import health_tag_markdown
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.markdown_utils import tags_to_markdown, dict_to_markdown, dict_to_collapsible_html


class ModelDetails(PluginInterface):
    """Model Details Composite Component"""

    # Initialize this Plugin Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL
    params = ParameterStore()

    def __init__(self):
        """Initialize the ModelDetails plugin class"""
        self.component_id = None
        self.current_model = None  # Don't use this in callbacks (not thread safe)

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.component_id = component_id
        self.container = html.Div(
            id=self.component_id,
            children=[
                html.H4(id=f"{self.component_id}-header", children="Model: Loading..."),
                dcc.Markdown(id=f"{self.component_id}-summary"),
                html.H5(children="Inference Metrics", style={"marginTop": "20px"}),
                dcc.Dropdown(id=f"{self.component_id}-dropdown", className="dropdown"),
                dcc.Markdown(id=f"{self.component_id}-metrics"),
                html.H5(children="Cross Fold Metrics", style={"marginTop": "20px"}),
                dcc.Markdown(id=f"{self.component_id}-cross-metrics", dangerously_allow_html=True),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-summary", "children"),
            (f"{self.component_id}-dropdown", "options"),
            (f"{self.component_id}-dropdown", "value"),
            (f"{self.component_id}-metrics", "children"),
            (f"{self.component_id}-cross-metrics", "children"),
        ]
        self.signals = [(f"{self.component_id}-dropdown", "value")]

        # Return the container
        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            model (CachedModel): An instantiated CachedModel object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        self.log.important(f"Updating Plugin with Model: {model.name} and kwargs: {kwargs}")

        # Update the header and the details
        self.current_model = model
        header = f"{self.current_model.name}"
        details = self.model_summary()

        # Populate the inference runs dropdown
        inference_runs, default_run = self.get_inference_runs()
        metrics = self.inference_metrics(default_run)
        cross_metrics = self.cross_metrics()

        # Return the updated property values for the plugin
        return [header, details, inference_runs, default_run, metrics, cross_metrics]

    def register_internal_callbacks(self):
        @callback(
            Output(f"{self.component_id}-metrics", "children", allow_duplicate=True),
            Input(f"{self.component_id}-dropdown", "value"),
            State(f"{self.component_id}-header", "children"),
            prevent_initial_call=True,
        )
        def update_inference_run(inference_run, model_name):
            # Suboptimal: We need to create the model object again
            self.current_model = CachedModel(model_name)

            # Update the model metrics
            metrics = self.inference_metrics(inference_run)
            return metrics

    def model_summary(self):
        """Construct the markdown string for the model summary

        Returns:
            str: A markdown string
        """

        # Get these fields from the model
        show_fields = [
            "health_tags",
            "input",
            "workbench_registered_endpoints",
            "workbench_model_type",
            "workbench_model_target",
            "workbench_model_features",
            "param_meta",
            "workbench_tags",
        ]

        # Construct the markdown string
        summary = self.current_model.summary()
        markdown = ""
        for key in show_fields:

            # Special case for the health tags
            if key == "health_tags":
                markdown += health_tag_markdown(summary.get(key, []))
                continue

            # Special case for the features
            if key == "workbench_model_features":
                value = summary.get(key, [])
                key = "features"
                value = f"({len(value)}) {', '.join(value)[:100]}..."
                markdown += f"**{key}:** {value}  \n"
                continue

            # Special case for Parameter Store Metadata
            if key == "param_meta":
                model_name = summary["name"]
                meta_data = self.params.get(f"/workbench/models/{model_name}/meta", warn=False)
                if meta_data:
                    markdown += dict_to_markdown(meta_data, title="Additional Metadata")
                continue

            # Special case for tags
            if key == "workbench_tags":
                tags = summary.get(key, "")
                markdown += tags_to_markdown(tags)
                continue

            # Get the value
            value = summary.get(key, "-")

            # If the value is a list, convert it to a comma-separated string
            if isinstance(value, list):
                value = ", ".join(value)

            # Chop off the "workbench_" prefix
            key = key.replace("workbench_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        return markdown

    def inference_metrics(self, inference_run: Union[str, None]) -> str:
        """Construct the markdown string for the model metrics

        Args:
            inference_run (str): The inference run to get the metrics for (None gives a 'not found' markdown)
        Returns:
            str: A markdown string
        """
        # Inference Metrics
        if self.current_model is None:
            meta_df = None
        else:
            meta_df = self.current_model.get_inference_metadata(inference_run) if inference_run else None
        if meta_df is None:
            test_data = "Inference Metadata Not Found"
            test_data_hash = " - "
            description = None
        else:
            inference_meta = meta_df.to_dict(orient="records")[0]
            test_data = inference_meta.get("name", " - ")
            test_data_hash = inference_meta.get("data_hash", " - ")
            description = inference_meta.get("description")

        # Add the markdown for the model test metrics
        markdown = "\n"
        markdown += f"**Test Data:** {test_data} ({test_data_hash})  \n"
        if description:
            markdown += f"**Description:** {description}  \n"

        # Grab the Metrics from the model details
        metrics = self.current_model.get_inference_metrics(capture_name=inference_run)
        if metrics is None:
            markdown += "  \nNo Data  \n"
        else:
            markdown += "  \n"
            metrics = metrics.round(3)

            # If the model is a classification model, have the index sorting match the class labels
            if self.current_model.model_type == ModelType.CLASSIFIER:
                # Sort the metrics by the class labels (if they match)
                class_labels = self.current_model.class_labels()
                if set(metrics.index) == set(class_labels):
                    metrics = metrics.reindex(class_labels)
            markdown += metrics.to_markdown()

        # Get additional inference metrics if they exist
        model_name = self.current_model.name
        inference_data = self.params.get(f"/workbench/models/{model_name}/inference/{inference_run}", warn=False)
        if inference_data:
            markdown += "\n\n"
            markdown += dict_to_markdown(inference_data, title="Additional Inference Metrics")
        return markdown

    def cross_metrics(self) -> str:
        # Get cross fold metrics if they exist
        model_name = self.current_model.name
        cross_fold_data = self.params.get(f"/workbench/models/{model_name}/inference/cross_fold", warn=False)
        if not cross_fold_data:
            return "**No Cross Fold Data**"

        # Convert the cross fold data to a markdown string
        html = dict_to_collapsible_html(cross_fold_data)
        return html

    def get_inference_runs(self):
        """Get the inference runs for the model

        Returns:
            list[str]: A list of inference runs
            default_run (str): The default inference run
        """

        # Inference runs
        inference_runs = self.current_model.list_inference_runs()

        # Check if there are any inference runs to select
        if not inference_runs:
            return [], None

        # Set "auto_inference" as the default, if that doesn't exist, set the first
        default_inference_run = "auto_inference" if "auto_inference" in inference_runs else inference_runs[0]

        # Return the options for the dropdown and the selected value
        return inference_runs, default_inference_run


if __name__ == "__main__":
    # This class takes in model details and generates a details Markdown component
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ModelDetails, theme="midnight_blue").run()
