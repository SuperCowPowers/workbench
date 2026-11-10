"""Model Script Utilities for Workbench endpoints.

Assembles a SageMaker training/inference bundle from a per-framework template
and its supporting files. The bundle is *not* self-contained — it relies on
``workbench`` being pip-installed in the SageMaker base image. Templates
import everything they need via ``from workbench.endpoints... import ...`` /
``from workbench.algorithms... import ...``; this module only fills the
template's placeholders and copies any framework-specific supporting files
that live alongside the template.
"""

import json
import os
import shutil
import logging
import tempfile
from pathlib import Path

from workbench.utils.target_transform import check_target_domain

# Setup the logger
log = logging.getLogger("workbench")


def fill_template(template_path: str, params: dict, output_script: str, output_dir: str = None) -> str:
    """
    Fill in the placeholders in the template with the values provided in params,
    ensuring that the correct Python data types are used.

    ``hyperparameters`` crosses as a JSON string that the template ``json.loads``.

    Args:
        template_path (str): The path to the template file.
        params (dict): A dictionary with placeholder keys and their corresponding values.
        output_script (str): The name of the generated model script.
        output_dir (str): The directory to write the output script to (default: same as template).
    Returns:
        str: The path to the generated model script.
    """

    # Read the template file
    with open(template_path, "r", encoding="utf-8") as fp:
        template = fp.read()

    # Perform the replacements
    for key, value in params.items():
        if key == "hyperparameters":
            value = repr(json.dumps(value or {}))
        # For string values wrap them in quotes (except for model_imports and model_class)
        elif isinstance(value, str) and key not in ["model_imports", "model_class"]:
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
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(template)

    return output_path


# Hyperparameters only the templates listed below act on. Every caller of
# generate_model_script reaches it with a built-in template, so a custom script — which owns
# its own hyperparameters — never passes through here.
TARGET_TRANSFORM_KEYS = ("target_transform", "prediction_bounds")
TARGET_TRANSFORM_FRAMEWORKS = ("XGBOOST", "CHEMPROP", "PYTORCH", "TABICL", "ATOMJEPA")


def _check_target_transform_support(template_params: dict, model_type_enum, model_framework_enum) -> None:
    """Reject a target transform the chosen template would silently drop.

    A spec handed to a template that does not invert it would train on the raw target and
    serve predictions that disagree with the request, so creation fails instead. Shared by
    every path that generates a script, AWS and local alike.
    """
    hyperparameters = template_params.get("hyperparameters") or {}

    # The target is transformed once, before HPO runs, so a searched transform would differ
    # between the trials and the winner's saved config that inference reads back
    hpo_block = hyperparameters.get("hpo")
    search_space = hpo_block.get("search_space") if isinstance(hpo_block, dict) else None
    searched = [k for k in TARGET_TRANSFORM_KEYS if isinstance(search_space, dict) and k in search_space]
    if searched:
        raise ValueError(f"{', '.join(searched)} cannot be searched by HPO; set it as a top-level hyperparameter")

    requested = [k for k in TARGET_TRANSFORM_KEYS if hyperparameters.get(k) is not None]
    if not requested:
        return

    supported = {getattr(model_framework_enum, name) for name in TARGET_TRANSFORM_FRAMEWORKS}
    framework = template_params.get("model_framework")
    if framework not in supported:
        raise ValueError(
            f"{getattr(framework, 'value', framework)} does not support {', '.join(requested)}. "
            f"Supported frameworks: {sorted(f.value for f in supported)}"
        )
    if template_params.get("model_type") == model_type_enum.CLASSIFIER:
        raise ValueError(f"{', '.join(requested)} applies to regression targets only")
    if template_params.get("model_class"):
        raise ValueError(f"a scikit-learn model_class does not support {', '.join(requested)}")

    # The target's own range decides every transform's domain, so a spec that cannot work on
    # this data fails here rather than inside a training job
    ranges = template_params.get("target_ranges")
    if ranges:
        targets = template_params.get("target_column")
        targets = [targets] if isinstance(targets, str) else list(targets or [])
        check_target_domain(
            hyperparameters.get("target_transform"), targets, ranges, template_params.get("target_label", "")
        )


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
            - model_metrics_path (str): Destination for model metrics (S3 path or local directory)
            - hyperparameters (dict, optional): Hyperparameters for the model (default: None)
            - endpoints (list[str], optional): For META models, list of endpoint names

    Returns:
        str: The path to the generated model script
    """
    from workbench.core.model_types import ModelType, ModelFramework

    _check_target_transform_support(template_params, ModelType, ModelFramework)

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
    elif template_params["model_framework"] == ModelFramework.TABICL:
        template_name = "tabicl.template"
        model_script_dir_name = "tabicl"
    elif template_params["model_framework"] == ModelFramework.ATOMJEPA:
        template_name = "atomjepa.template"
        model_script_dir_name = "atomjepa"
    elif template_params["model_framework"] == ModelFramework.META:
        template_name = "meta_endpoint.template"
        model_script_dir_name = "meta_endpoint"
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

    # Copy any non-template supporting files that live alongside the template
    # (e.g. requirements.txt). The workbench package itself is pip-installed
    # in the SageMaker container, so we no longer bundle workbench source —
    # templates import via ``from workbench... import ...``.
    for entry in os.listdir(source_script_dir):
        if entry.endswith(".template") or entry.startswith("generated_") or entry == "__pycache__":
            continue
        source_path = os.path.join(source_script_dir, entry)
        if os.path.isfile(source_path):
            shutil.copy(source_path, output_dir)
            log.info(f"Copied supporting file: {entry}")

    # Copy the training_harness.py wrapper (handles post-training inference bundling)
    entrypoint_path = package_dir.parent / "training" / "training_harness.py"
    if entrypoint_path.exists():
        shutil.copy(entrypoint_path, output_dir)
        log.info("Copied training_harness.py")
    else:
        log.warning(f"training_harness.py not found at {entrypoint_path}")

    # Fill in the template and write the generated script to the temp directory
    output_path = fill_template(template_path, template_params, "generated_model_script.py", output_dir)

    return output_path


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""
    from workbench.api import ModelType, ModelFramework

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
        "model_metrics_path": "s3://workbench-public-test-bucket/models/training/wine-classifier",
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
        "model_metrics_path": "s3://workbench-public-test-bucket/models/training/wine-clusters",
    }
    my_model_script = generate_model_script(my_params)
    print(f"Generated script: {my_model_script}")
