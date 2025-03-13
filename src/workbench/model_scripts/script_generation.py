"""Model Script Utilities for Workbench endpoints"""

import os
import shutil
import logging
from pathlib import Path
import importlib.util


# Setup the logger
log = logging.getLogger("workbench")


def copy_imports_to_script_dir(script_path: str, imports: list[str]) -> None:
    """
    Copy specified utility files to the script directory by resolving their locations dynamically.

    Args:
        script_path (str): Full path of the script file (we will copy the imports to the same directory).
        imports (list[str]): A list of imports (e.g., "workbench.utils.chem_utils") to copy.
    """
    # Compute the script directory from the script path
    script_dir = Path(script_path).parent

    for import_path in imports:
        # Try to locate the module's file path
        spec = importlib.util.find_spec(import_path)
        if spec is None or spec.origin is None:
            raise ImportError(f"Cannot find module: {import_path}")

        source_path = Path(spec.origin)  # Get the file path from the module spec

        # Ensure the source file exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found for: {import_path}")

        # Resolve destination path
        destination_path = script_dir / source_path.name

        # Copy the file
        shutil.copy(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")


def fill_template(template_path: str, params: dict, output_script: str) -> str:
    """
    Fill in the placeholders in the template with the values provided in params,
    ensuring that the correct Python data types are used.
    Args:
        template_path (str): The path to the template file.
        params (dict): A dictionary with placeholder keys and their corresponding values.
        output_script (str): The name of the generated model script.
    Returns:
        str: The path to the generated model script.
    """

    # Read the template file
    with open(template_path, "r") as fp:
        template = fp.read()

    # Perform the replacements
    for key, value in params.items():
        # For string values wrap them in quotes (except for model_imports and scikit_model_class)
        if isinstance(value, str) and key not in ["model_imports", "scikit_model_class"]:
            value = f'"{value}"'
        # Replace the placeholder in the template
        placeholder = f'"{{{{{key}}}}}"'  # Double curly braces to match the template
        template = template.replace(placeholder, str(value))

    # Sanity check to ensure all placeholders were replaced
    if "{{" in template or "}}" in template:
        msg = "Not all template placeholders were replaced. Please check your params."
        log.critical(msg)
        raise ValueError(msg)

    # Write out the generated model script and return the name
    output_path = os.path.join(os.path.dirname(template_path), output_script)
    with open(output_path, "w") as fp:
        fp.write(template)
    return output_path


def generate_model_script(template_params: dict) -> str:
    """
    Fill in the model template with specific parameters.

    Args:
        template_params (dict): Dictionary containing the parameters:
            - model_imports (str): Import string for the model class
            - model_type (ModelType): The enumerated type of model to generate
            - scikit_model_class (str): The model class to use (e.g., "RandomForestRegressor")
            - target_column (str): Column name of the target variable
            - feature_list (list[str]): A list of columns for the features
            - model_metrics_s3_path (str): The S3 path to store the model metrics
            - train_all_data (bool): Whether to train on all (100%) of the data

    Returns:
        str: The name of the generated model script
    """
    from workbench.api import ModelType  # Avoid circular import

    # Determine which template to use based on model type
    if template_params.get("scikit_model_class"):
        template_name = "scikit_learn.template"
        model_script_dir = "light_scikit_learn"
    elif template_params["model_type"] in [ModelType.REGRESSOR, ModelType.CLASSIFIER]:
        template_name = "xgb_model.template"
        model_script_dir = "light_xgb_model"
    elif template_params["model_type"] == ModelType.QUANTILE_REGRESSOR:
        template_name = "quant_regression.template"
        model_script_dir = "light_quant_regression"
    else:
        msg = f"ModelType: {template_params['model_type']} needs to set custom_script argument"
        log.critical(msg)
        raise ValueError(msg)

    # Model Type is an enumerated type, so we need to convert it to a string
    template_params["model_type"] = template_params["model_type"].value

    # Load the template
    dir_path = Path(__file__).parent.absolute()
    model_script_dir = os.path.join(dir_path, model_script_dir)
    template_path = os.path.join(model_script_dir, template_name)

    # Fill in the template and write out the generated model script
    output_path = fill_template(template_path, template_params, "generated_model_script.py")

    # Ensure the model script directory only contains the template, model script, and a requirements file
    for file in os.listdir(model_script_dir):
        if file not in ["generated_model_script.py", "requirements.txt"] and not file.endswith(".template"):
            log.warning(f"Unexpected file {file} found in model_script_dir...")

    return output_path


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""
    from workbench.api import ModelType

    copy_imports_to_script_dir("/tmp/model.py", ["workbench.utils.chem_utils"])

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
        "model_metrics_s3_path": "s3://sandbox-workbench-artifacts/models/training/wine-classifier",
        "train_all_data": True,
    }
    my_model_script = generate_model_script(my_params)
    print(my_model_script)

    # Define the parameters for the model script (KMeans Clustering)
    my_params = {
        "model_type": ModelType.CLUSTERER,
        "scikit_model_class": "KMeans",
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
        "model_metrics_s3_path": "s3://sandbox-workbench-artifacts/models/training/wine-clusters",
        "train_all_data": True,
    }
    my_model_script = generate_model_script(my_params)
    print(my_model_script)
