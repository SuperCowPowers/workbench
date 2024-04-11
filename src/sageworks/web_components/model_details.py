"""A Markdown Component for details/information about Models"""

# Dash Imports
from dash import html, callback, no_update, dcc
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.api import Model
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.utils.markdown_utils import health_tag_markdown


class ModelDetails(ComponentInterface):
    """Model Markdown Component"""

    def __init__(self):
        self.prefix_id = ""
        self.model = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.prefix_id = component_id
        container = html.Div(
            id=self.prefix_id,
            children=[
                html.H3(id=f"{self.prefix_id}-header", children="Model: Loading..."),
                dcc.Markdown(id=f"{self.prefix_id}-summary"),
                html.H3(children="Inference Metrics"),
                dcc.Dropdown(id=f"{self.prefix_id}-dropdown", className="dropdown"),
                dcc.Markdown(id=f"{self.prefix_id}-metrics"),
                dcc.Store(id=f"{self.prefix_id}-model-uuid"),  # Store the model uuid
            ],
        )
        return container

    def register_callbacks(self, model_table):
        @callback(
            [
                Output(f"{self.prefix_id}-header", "children"),
                Output(f"{self.prefix_id}-summary", "children"),
                Output(f"{self.prefix_id}-dropdown", "options"),
                Output(f"{self.prefix_id}-dropdown", "value"),
                Output(f"{self.prefix_id}-metrics", "children", allow_duplicate=True),
                Output(f"{self.prefix_id}-model-uuid", "data"),
            ],
            Input(model_table, "derived_viewport_selected_row_ids"),
            State(model_table, "data"),
            prevent_initial_call=True,
        )
        def update_model(selected_rows, table_data):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                return no_update

            # Get the selected row data, grab the uuid, and set the Model object
            selected_row_data = table_data[selected_rows[0]]
            model_uuid = selected_row_data["uuid"]
            self.model = Model(model_uuid, legacy=True)

            # Update the header, the summary, and the details
            header = f"Model: {self.model.uuid}"
            summary = self.model_summary()

            # Populate the inference runs dropdown
            inference_runs, default_run = self.get_inference_runs()

            # Update the metrics for the default inference run
            metrics = self.inference_metrics(default_run)

            # Return the updated components
            return header, summary, inference_runs, default_run, metrics, model_uuid

        @callback(
            Output(f"{self.prefix_id}-metrics", "children", allow_duplicate=True),
            Input(f"{self.prefix_id}-dropdown", "value"),
            State(f"{self.prefix_id}-model-uuid", "data"),
            prevent_initial_call=True,
        )
        def update_inference_run(inference_run, model_uuid):
            # Check for no inference run
            if not inference_run or (self.model.uuid != model_uuid):
                return no_update

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
            "sageworks_registered_endpoints",
            "sageworks_model_type",
            "sageworks_tags",
            "sageworks_model_target",
            "sageworks_model_features",
        ]

        # Construct the markdown string
        summary = self.model.summary()
        markdown = ""
        for key in show_fields:

            # Special case for the health tags
            if key == "health_tags":
                markdown += health_tag_markdown(summary.get(key, []))
                continue

            # Special case for the features
            if key == "sageworks_model_features":
                value = summary.get(key, [])
                key = "features"
                value = f"({len(value)}) {', '.join(value)[:100]}..."
                markdown += f"**{key}:** {value}  \n"
                continue

            # Get the value
            value = summary.get(key, "-")

            # If the value is a list, convert it to a comma-separated string
            if isinstance(value, list):
                value = ", ".join(value)

            # Chop off the "sageworks_" prefix
            key = key.replace("sageworks_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        return markdown

    def inference_metrics(self, inference_run: str):
        """Construct the markdown string for the model metrics

        Args:
            inference_run (str): The inference run to get the metrics for
        Returns:
            str: A markdown string
        """
        # Model Metrics
        meta_df = self.model.inference_metadata(inference_run)
        if meta_df is None:
            test_data = "Inference Metadata Not Found"
            test_data_hash = " N/A "
            test_rows = " - "
            description = " - "
        else:
            inference_meta = meta_df.to_dict(orient="records")[0]
            test_data = inference_meta.get("name", " - ")
            test_data_hash = inference_meta.get("data_hash", " - ")
            test_rows = inference_meta.get("num_rows", " - ")
            description = inference_meta.get("description", " - ")

        # Add the markdown for the model test metrics
        markdown = "\n"
        markdown += f"**Test Data:** {test_data}  \n"
        markdown += f"**Data Hash:** {test_data_hash}  \n"
        markdown += f"**Test Rows:** {test_rows}  \n"
        markdown += f"**Description:** {description}  \n"

        # Grab the Metrics from the model details
        metrics = self.model.performance_metrics(capture_uuid=inference_run)
        if metrics is None:
            markdown += "  \nNo Data  \n"
        else:
            markdown += "  \n"
            metrics = metrics.round(3)
            markdown += metrics.to_markdown(index=False)

        print(markdown)
        return markdown

    def get_inference_runs(self):
        """Get the inference runs for the model

        Returns:
            list[str]: A list of inference runs
            default_run (str): The default inference run
        """

        # Inference runs
        inference_runs = self.model.list_inference_runs()

        # Check if there are any inference runs to select
        if not inference_runs:
            return [], None

        # Set "training_holdout" as the default, if that doesn't exist, set the first
        default_inference_run = "training_holdout" if "training_holdout" in inference_runs else inference_runs[0]

        # Return the options for the dropdown and the selected value
        return inference_runs, default_inference_run


if __name__ == "__main__":
    # This class takes in model details and generates a details Markdown component
    import dash
    import dash_bootstrap_components as dbc
    from sageworks.web_components.table import Table
    from sageworks.api.meta import Meta

    # Create a model table
    models_table = Table().create_component(
        "my_models_table", header_color="rgb(60, 100, 60)", row_select="single", max_height=270
    )

    # Populate the table with data
    models = Meta().models()
    models["uuid"] = models["Model Group"]
    models["id"] = range(len(models))
    column_setup_list = Table().column_setup(models)
    models_table.columns = column_setup_list
    models_table.data = models.to_dict("records")

    # Instantiate the ModelDetails class
    md = ModelDetails()
    details_component = md.create_component("my_model_details")

    # Register the callbacks
    md.register_callbacks("my_models_table")

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
    )

    app.layout = html.Div([models_table, details_component])
    app.run(debug=True)
