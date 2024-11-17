"""Model Script Utilities for SageWorks endpoints"""

import os
import logging
from pathlib import Path


# Setup the logger
log = logging.getLogger("sageworks")


def fill_template(template: str, params: dict) -> str:
    """
    Fill in the placeholders in the template with the values provided in params,
    ensuring that the correct Python data types are used.
    Args:
        template (str): The template string with placeholders.
        params (dict): A dictionary with placeholder keys and their corresponding values.
    Returns:
        str: The template with placeholders replaced by their values.
    """
    # Perform the replacements
    for key, value in params.items():
        # For string values wrap them in quotes (except for model_imports and model_class
        if isinstance(value, str) and key not in ["model_imports", "model_class"]:
            value = f'"{value}"'
        # Replace the placeholder in the template
        placeholder = f'"{{{{{key}}}}}"'  # Double curly braces to match the template
        template = template.replace(placeholder, str(value))

    # Sanity check to ensure all placeholders were replaced
    if "{{" in template or "}}" in template:
        msg = "Not all template placeholders were replaced. Please check your params."
        log.critical(msg)
        raise ValueError(msg)
    return template


def generate_model_script(template_params: dict) -> str:
    """
    Fill in the model template with specific parameters.

    Args:
        template_params (dict): Dictionary containing the parameters:
            - model_imports (str): Import string for the model class
            - model_type (ModelType): The enumerated type of model to generate
            - model_class (str): The model class to use (e.g., "RandomForestRegressor")
            - target_column (str): Column name of the target variable
            - feature_list (list[str]): A list of columns for the features
            - model_metrics_s3_path (str): The S3 path to store the model metrics
            - train_all_data (bool): Whether to train on all (100%) of the data

    Returns:
        str: The name of the generated model script
    """
    from sageworks.api import ModelType  # Avoid circular import

    # Output script name
    output_script = "generated_model_script.py"

    # Determine which template to use based on model type
    if template_params.get("model_class"):
        template_name = "scikit_learn.template"
        model_script_dir = "light_scikit_learn"
    elif template_params["model_type"] in [ModelType.REGRESSOR, ModelType.CLASSIFIER]:
        template_name = "xgb_model.template"
        model_script_dir = "light_xgb_model"
    elif template_params["model_type"] == ModelType.QUANTILE_REGRESSOR:
        template_name = "quant_regression.template"
        model_script_dir = "light_quant_regression"
    else:
        log.critical(f"Unknown ModelType: {template_params['model_type']}")
        raise ValueError(f"Unknown ModelType: {template_params['model_type']}")

    # Model Type is an enumerated type, so we need to convert it to a string
    template_params["model_type"] = template_params["model_type"].value

    # Load the template
    dir_path = Path(__file__).parent.absolute()
    model_script_dir = os.path.join(dir_path, model_script_dir)
    template_path = os.path.join(model_script_dir, template_name)
    with open(template_path, "r") as fp:
        template = fp.read()

    # Fill in the template using the utility function
    aws_script = fill_template(template, template_params)

    # Write out the generated model script and return the name
    output_path = os.path.join(model_script_dir, output_script)
    with open(output_path, "w") as fp:
        fp.write(aws_script)

    # Ensure the model script directory only contains the template, model script, and a requirements file
    for file in os.listdir(model_script_dir):
        if file not in [output_script, "requirements.txt"] and not file.endswith(".template"):
            log.warning(f"Unexpected file {file} found in model_script_dir...")

    return output_path


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""
    from sageworks.api import ModelType

    # Define the parameters for the model script (Classifier)
    my_params = {
        "model_type": ModelType.CLASSIFIER,
        "target_column": "wine_class",
        "feature_list": [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315_of_diluted_wines",
            "proline",
        ],
        "model_metrics_s3_path": "s3://sandbox-sageworks-artifacts/models/training/wine-classifier",
        "train_all_data": True,
    }
    my_model_script = generate_model_script(my_params)
    print(my_model_script)

    # Define the parameters for the model script (KMeans Clustering)
    my_params = {
        "model_type": ModelType.CLUSTERER,
        "model_class": "KMeans",
        "model_imports": "from sklearn.cluster import KMeans",
        "target_column": None,
        "feature_list": [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315_of_diluted_wines",
            "proline",
        ],
        "model_metrics_s3_path": "s3://sandbox-sageworks-artifacts/models/training/wine-clusters",
        "train_all_data": True,
    }
    my_model_script = generate_model_script(my_params)
    print(my_model_script)
