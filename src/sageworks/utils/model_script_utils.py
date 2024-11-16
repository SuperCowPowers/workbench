"""Model Script Utilities for SageWorks endpoints"""

import logging

# SageWorks Imports


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


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""
    import os

    # Hard Code for now :)
    home_dir = os.path.expanduser("~")
    sage_src_path = "work/sageworks/src/sageworks/core/transforms/features_to_model"
    template_path = f"{home_dir}/{sage_src_path}/light_scikit_learn/scikit_learn.template"

    # Define the template and params
    my_template = open(template_path).read()
    my_params = {
        "model_imports": "from sklearn.cluster import KMeans",
        "model_type": "clusterer",
        "model_class": "KMeans",
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
    aws_script = fill_template(my_template, my_params)
    print(aws_script)
