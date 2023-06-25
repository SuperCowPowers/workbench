"""A Component for model details/information"""
from dash import dcc


def create_markdown(model_info: dict):
    """Create the Markdown for the Model Details"""

    markdown_template = """
        #### {{model_name}}
        - **Created:** {{date_created}}
        - **Training Data:**
            - Nightly ({{date_created}})
            - {{training_data}}
        - **Features Set:** {{feature_set}}
        - **Type:** {{model_type}}
        - **Algorithm:** XGBoost
        - **Endpoints:**
            - {{model_name}}-80
            - {{model_name}}-100

        ##### Model Scores and Metrics

        All model scores and metrics are based on 80/20 train/test splits of the **Training Data** (see above).
        The project will be improving upon these existing reference models as the project progresses.
    """
    # Hack Logic
    if "regress" in model_info["model_name"]:
        model_type = "Regression/Numerical"
    else:
        model_type = "Classification/Categorical"

    # Replace all the template fields
    markdown = markdown_template.replace("{{model_name}}", model_info["model_name"])
    markdown = markdown.replace("{{date_created}}", str(model_info["date_created"])[:10])
    markdown = markdown.replace("{{training_data}}", model_info["training_data"])
    markdown = markdown.replace("{{feature_set}}", model_info["feature_set"])
    markdown = markdown.replace("{{model_type}}", model_type)

    return markdown


def create(model_info: dict) -> dcc.Markdown:
    """Create a Markdown Component for the Model Details"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(id="model_details", children=create_markdown(model_info))
