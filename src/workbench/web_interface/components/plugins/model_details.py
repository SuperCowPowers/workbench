"""Model Details: A compound plugin composing ModelSummary + InferenceMetrics."""

from typing import Union

# Dash Imports
from dash import html, callback, dcc, no_update, Input, Output

# Workbench Imports
from workbench.api import ModelType, ParameterStore
from workbench.cached.cached_model import CachedModel
from workbench.utils.markdown_utils import (
    health_tag_markdown,
    tags_to_markdown,
    dict_to_markdown,
    dict_to_collapsible_html,
    df_to_html_table,
)
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ModelSummary(PluginInterface):
    """Model Summary: header + summary markdown for a model."""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL
    params = ParameterStore()

    def __init__(self):
        """Initialize the ModelSummary plugin class"""
        self.component_id = None
        self.model = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the summary component (header + markdown).

        Args:
            component_id (str): The base component ID (shared with parent)

        Returns:
            html.Div: A container with the header and summary markdown
        """
        self.component_id = component_id
        self.container = html.Div(
            id=f"{self.component_id}-summary-section",
            children=[
                html.H4(id=f"{self.component_id}-header", children="Model: Loading..."),
                dcc.Markdown(id=f"{self.component_id}-summary", dangerously_allow_html=True),
            ],
        )
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-summary", "children"),
        ]
        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the header and summary markdown.

        Args:
            model (CachedModel): An instantiated CachedModel object
            **kwargs (dict): Additional keyword arguments (unused by this sub-plugin)

        Returns:
            list: [header, summary_markdown]
        """
        self.model = model
        header = model.name
        summary = self._model_summary()
        return [header, summary]

    def _model_summary(self) -> str:
        """Construct the markdown string for the model summary.

        Returns:
            str: A markdown string
        """
        summary = self.model.summary()
        markdown = ""

        # Health tags
        markdown += health_tag_markdown(summary.get("health_tags", []))

        # Simple fields
        markdown += f"**input:** {summary.get('input', '-')}  \n"
        endpoints = ", ".join(summary.get("workbench_registered_endpoints", []))
        markdown += f"**registered_endpoints:** {endpoints or '-'}  \n"
        markdown += f"**model_type:** {summary.get('workbench_model_type', '-')}  \n"
        markdown += f"**model_target:** {summary.get('workbench_model_target', '-')}  \n"

        # Features (truncated)
        features = summary.get("workbench_model_features", [])
        features_str = f"({len(features)}) {', '.join(features)[:100]}..."
        markdown += f"**features:** {features_str}  \n"

        # Parameter Store metadata
        model_name = summary["name"]
        meta_data = self.params.get(f"/workbench/models/{model_name}/meta", warn=False)
        if meta_data:
            markdown += dict_to_markdown(meta_data, title="Additional Metadata")

        # Tags
        markdown += tags_to_markdown(summary.get("workbench_tags", "")) + "  \n"

        # Hyperparameters
        hyperparams = summary.get("hyperparameters")
        if hyperparams and isinstance(hyperparams, dict):
            markdown += dict_to_collapsible_html(hyperparams, title="Hyperparameters", collapse_all=True)

        return markdown


class InferenceMetrics(PluginInterface):
    """Inference Metrics: dropdown selector + metrics markdown for a model's inference runs."""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL
    params = ParameterStore()

    def __init__(self):
        """Initialize the InferenceMetrics plugin class"""
        self.component_id = None
        self.model = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the inference metrics component (dropdown + markdown).

        Args:
            component_id (str): The base component ID (shared with parent)

        Returns:
            html.Div: A container with the dropdown and metrics markdown
        """
        self.component_id = component_id
        self.container = html.Div(
            id=f"{self.component_id}-metrics-section",
            children=[
                html.H5(children="Inference Metrics", style={"marginTop": "20px"}),
                dcc.Dropdown(id=f"{self.component_id}-dropdown", className="dropdown"),
                dcc.Markdown(id=f"{self.component_id}-metrics", dangerously_allow_html=True),
            ],
        )
        self.properties = [
            (f"{self.component_id}-dropdown", "options"),
            (f"{self.component_id}-dropdown", "value"),
            (f"{self.component_id}-metrics", "children"),
        ]
        self.signals = [(f"{self.component_id}-dropdown", "value")]
        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the dropdown options and metrics markdown.

        Args:
            model (CachedModel): An instantiated CachedModel object
            **kwargs (dict): Additional keyword arguments
                - inference_run: Current inference run selection (to preserve user's choice)
                - previous_model_name: Name of the previously selected model

        Returns:
            list: [dropdown_options, dropdown_value, metrics_markdown]
        """
        self.model = model

        # Populate the inference runs dropdown
        inference_runs, default_run = self._get_inference_runs()

        # Check if the model changed
        previous_model_name = kwargs.get("previous_model_name")
        current_inference_run = kwargs.get("inference_run")
        model_changed = previous_model_name != model.name

        # Only preserve the inference run if the model hasn't changed AND the selection is valid
        if not model_changed and current_inference_run and current_inference_run in inference_runs:
            metrics = self._inference_metrics(current_inference_run)
            return [inference_runs, no_update, metrics]
        else:
            metrics = self._inference_metrics(default_run)
            return [inference_runs, default_run, metrics]

    def register_internal_callbacks(self):
        """Register the dropdown → metrics callback."""

        # allow_duplicate: metrics also set by page-level update_model_details callback
        @callback(
            Output(f"{self.component_id}-metrics", "children", allow_duplicate=True),
            Input(f"{self.component_id}-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_inference_run(inference_run):
            # Uses self.model directly (set by update_properties) instead of
            # re-instantiating CachedModel from scratch on every dropdown change.
            return self._inference_metrics(inference_run)

    def _inference_metrics(self, inference_run: Union[str, None]) -> str:
        """Construct the markdown string for the model metrics.

        Args:
            inference_run (str): The inference run to get the metrics for (None gives a 'not found' markdown)

        Returns:
            str: A markdown string
        """
        if self.model is None:
            meta_df = None
        else:
            meta_df = self.model.get_inference_metadata(inference_run) if inference_run else None
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
        metrics = self.model.get_inference_metrics(capture_name=inference_run)
        if metrics is None:
            markdown += "  \nNo Data  \n"
        else:
            markdown += "  \n"

            # If the model is a classification model, have the index sorting match the class labels
            if self.model.model_type == ModelType.CLASSIFIER:
                class_labels = self.model.class_labels()
                if set(metrics.index) == set(class_labels):
                    metrics = metrics.reindex(class_labels)

            markdown += df_to_html_table(metrics)

        # Get additional inference metrics if they exist
        model_name = self.model.name
        inference_data = self.params.get(f"/workbench/models/{model_name}/inference/{inference_run}", warn=False)
        if inference_data:
            markdown += "\n\n"
            markdown += dict_to_markdown(inference_data, title="Additional Inference Metrics")
        return markdown

    def _get_inference_runs(self):
        """Get the inference runs for the model.

        Returns:
            list[str]: A list of inference runs
            default_run (str): The default inference run
        """
        inference_runs = self.model.list_inference_runs()

        if not inference_runs:
            return [], None

        # Default inference run (full_cross_fold if it exists, then auto_inference, then first)
        if "full_cross_fold" in inference_runs:
            default_inference_run = "full_cross_fold"
        elif "auto_inference" in inference_runs:
            default_inference_run = "auto_inference"
        else:
            default_inference_run = inference_runs[0]

        return inference_runs, default_inference_run


class ModelDetails(PluginInterface):
    """Model Details: A compound plugin composing ModelSummary + InferenceMetrics."""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ModelDetails compound plugin"""
        self.component_id = None
        self.summary = ModelSummary()
        self.inference = InferenceMetrics()
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the compound component by assembling both children.

        Args:
            component_id (str): The ID of the web component

        Returns:
            html.Div: A container with both sub-components
        """
        self.component_id = component_id

        # Create children (both use the same base component_id for leaf IDs)
        summary_component = self.summary.create_component(component_id)
        inference_component = self.inference.create_component(component_id)

        # Assemble into a single container
        self.container = html.Div(
            id=self.component_id,
            children=[summary_component, inference_component],
        )

        # Aggregate properties: summary (2) + inference (3) = 5 total (same order as before)
        self.properties = list(self.summary.properties) + list(self.inference.properties)
        self.signals = list(self.inference.signals)

        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update both sub-plugins with the model.

        Args:
            model (CachedModel): An instantiated CachedModel object
            **kwargs (dict): Additional keyword arguments
                - inference_run: Current inference run selection (to preserve user's choice)
                - previous_model_name: Name of the previously selected model

        Returns:
            list: Combined property values from both sub-plugins
        """
        self.log.important(f"Updating Plugin with Model: {model.name} and kwargs: {kwargs}")
        summary_props = self.summary.update_properties(model, **kwargs)
        inference_props = self.inference.update_properties(model, **kwargs)
        return summary_props + inference_props

    def register_internal_callbacks(self):
        """Register internal callbacks for the inference metrics sub-plugin."""
        self.inference.register_internal_callbacks()


if __name__ == "__main__":
    # This class takes in model details and generates a details Markdown component
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ModelDetails, theme="midnight_blue").run()
