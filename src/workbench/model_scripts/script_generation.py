"""Model Script Utilities for Workbench endpoints"""

import os
import shutil
import logging
import tempfile
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


def fill_template(template_path: str, params: dict, output_script: str, output_dir: str = None) -> str:
    """
    Fill in the placeholders in the template with the values provided in params,
    ensuring that the correct Python data types are used.
    Args:
        template_path (str): The path to the template file.
        params (dict): A dictionary with placeholder keys and their corresponding values.
        output_script (str): The name of the generated model script.
        output_dir (str): The directory to write the output script to (default: same as template).
    Returns:
        str: The path to the generated model script.
    """

    # Read the template file
    with open(template_path, "r") as fp:
        template = fp.read()

    # Perform the replacements
    for key, value in params.items():
        # For string values wrap them in quotes (except for model_imports and model_class)
        if isinstance(value, str) and key not in ["model_imports", "model_class"]:
            value = f'"{value}"'
        # Replace the placeholder in the template
        placeholder = f'"{{{{{key}}}}}"'  # Double curly braces to match the template
        template = template.replace(placeholder, str(value))

    # Sanity check to ensure all placeholders were replaced
    if "{{" in template and "}}" in template:
        msg = "Not all template placeholders were replaced. Please check your params."

        # Show which placeholders are still present
        start = template.index("{{")
        end = template.index("}}", start) + 2
        msg += f" Unreplaced placeholder: {template[start:end]}"
        log.critical(msg)
        raise ValueError(msg)

    # Write out the generated model script to output_dir (or template dir if not specified)
    if output_dir is None:
        output_dir = os.path.dirname(template_path)
    output_path = os.path.join(output_dir, output_script)
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
            - model_framework (str): The enumerated model framework to use
            - model_class (str): The model class to use (e.g., "RandomForestRegressor")
            - target_column (str): Column name of the target variable
            - feature_list (list[str]): A list of columns for the features
            - model_metrics_s3_path (str): The S3 path to store the model metrics
            - train_all_data (bool): Whether to train on all (100%) of the data
            - hyperparameters (dict, optional): Hyperparameters for the model (default: None)
            - child_endpoints (list[str], optional): For META models, list of child endpoint names

    Returns:
        str: The path to the generated model script
    """
    from workbench.api import ModelType, ModelFramework  # Avoid circular import

    # Determine which template to use based on model type
    if template_params.get("model_class"):
        template_name = "scikit_learn.template"
        model_script_dir_name = "scikit_learn"
    elif template_params["model_framework"] == ModelFramework.PYTORCH:
        template_name = "pytorch.template"
        model_script_dir_name = "pytorch_model"
    elif template_params["model_framework"] == ModelFramework.CHEMPROP:
        template_name = "chemprop.template"
        model_script_dir_name = "chemprop"
    elif template_params["model_framework"] == ModelFramework.META:
        template_name = "meta_model.template"
        model_script_dir_name = "meta_model"
    elif template_params["model_type"] in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.CLASSIFIER]:
        template_name = "xgb_model.template"
        model_script_dir_name = "xgb_model"
    elif template_params["model_type"] == ModelType.ENSEMBLE_REGRESSOR:
        template_name = "ensemble_xgb.template"
        model_script_dir_name = "ensemble_xgb"
    else:
        msg = f"ModelType: {template_params['model_type']} needs to set custom_script argument"
        log.critical(msg)
        raise ValueError(msg)

    # Model Type is an enumerated type, so we need to convert it to a string
    template_params["model_type"] = template_params["model_type"].value

    # Load the template from the package directory
    package_dir = Path(__file__).parent.absolute()
    source_script_dir = os.path.join(package_dir, model_script_dir_name)
    template_path = os.path.join(source_script_dir, template_name)

    # Create a temp directory for the generated script and supporting files
    # Note: This directory will be cleaned up by SageMaker after the training job
    output_dir = tempfile.mkdtemp(prefix=f"workbench_{model_script_dir_name}_")
    log.info(f"Generating model script in temp directory: {output_dir}")

    # Copy all supporting files (except templates and generated scripts) to the temp directory
    for file in os.listdir(source_script_dir):
        if file.endswith(".template") or file.startswith("generated_"):
            continue
        source_file = os.path.join(source_script_dir, file)
        if os.path.isfile(source_file):
            shutil.copy(source_file, output_dir)
            log.info(f"Copied supporting file: {file}")

    # Fill in the template and write the generated script to the temp directory
    output_path = fill_template(template_path, template_params, "generated_model_script.py", output_dir)

    return output_path


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""
    from workbench.api import ModelType, ModelFramework

    copy_imports_to_script_dir("/tmp/model.py", ["workbench.utils.chem_utils"])

    # Define the parameters for the model script (Classifier)
    my_params = {
        "model_type": ModelType.CLASSIFIER,
        "model_framework": ModelFramework.XGBOOST,
        "id_column": "id",
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
        "model_metrics_s3_path": "s3://workbench-public-test-bucket/models/training/wine-classifier",
        "train_all_data": True,
        "compressed_features": [],
        "hyperparameters": {},
    }
    my_model_script = generate_model_script(my_params)
    print(f"Generated script: {my_model_script}")

    # Define the parameters for the model script (KMeans Clustering)
    my_params = {
        "model_type": ModelType.CLUSTERER,
        "model_framework": ModelFramework.SKLEARN,
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
        "model_metrics_s3_path": "s3://workbench-public-test-bucket/models/training/wine-clusters",
        "train_all_data": True,
    }
    my_model_script = generate_model_script(my_params)
    print(f"Generated script: {my_model_script}")
