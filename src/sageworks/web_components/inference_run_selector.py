"""An Inference Selector Component for models"""

from dash import dcc

# SageWorks Imports
from sageworks.api import Model
from sageworks.web_components.component_interface import ComponentInterface


class InferenceRunSelector(ComponentInterface):
    """Inference Run Selector Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Dropdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Dropdown: A Dropdown component
        """
        # waiting_markdown = html.Div("*Waiting for data...*")
        return dcc.Dropdown(id=component_id)

    def generate_inference_runs(self, model: Model) -> list[str]:
        """Generates the inference runs to be used as options for the Dropdown
        Args:
            model (Model): Sageworks Model object
        Returns:
            list[str]: A list of inference runs
        """

        # Inference runs
        inference_runs = model.list_inference_runs()

        return inference_runs


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    from sageworks.api import Model

    # Instantiate model
    m = Model("abalone-regression")

    # Instantiate the class
    irs = InferenceRunSelector()

    # Generate the component
    dropdown = irs.create_component("dropdown")
    inf_runs = irs.generate_inference_runs(m)

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder="/home/kolmar/sageworks/applications/aws_dashboard/assets",
    )

    app.layout = html.Div([dropdown, html.Div(id="dd-output-container")])
    dropdown.options = inf_runs

    @callback(Output("dd-output-container", "children"), Input("dropdown", "value"))
    def update_output(value):
        return f"You have selected {value}"

    # Run server
    app.run_server(debug=True)
