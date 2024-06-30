from dash import dcc, html
from dash.dependencies import Input, Output
from dash import Dash, callback


class ModelDetailsComponent:
    def __init__(self, id_prefix: str):
        self.id_prefix = id_prefix
        self.layout = html.Div(
            [
                dcc.Markdown(id=f"{id_prefix}-markdown-1"),
                dcc.Markdown(id=f"{id_prefix}-markdown-2"),
                dcc.Dropdown(
                    id=f"{id_prefix}-dropdown",
                    options=[
                        {"label": "Model 1", "value": "model1"},
                        {"label": "Model 2", "value": "model2"},
                        # Add more models here
                    ],
                    value="model1",  # Default value
                ),
            ]
        )

    def register_callbacks(self):
        @callback(
            [
                Output(f"{self.id_prefix}-markdown-1", "children"),
                Output(f"{self.id_prefix}-markdown-2", "children"),
            ],
            [Input(f"{self.id_prefix}-dropdown", "value")],
        )
        def update_markdown(selected_model):
            # Update the markdown text based on the selected model
            if selected_model == "model1":
                return "Markdown text for Model 1", "Additional details for Model 1"
            elif selected_model == "model2":
                return "Markdown text for Model 2", "Additional details for Model 2"
            # Add more conditions for other models
            return "", ""


# Usage

model_details_component = ModelDetailsComponent("model-details")
model_details_component.register_callbacks()
app = Dash(__name__)
app.layout = html.Div([model_details_component.layout])

if __name__ == "__main__":
    app.run(debug=True)
