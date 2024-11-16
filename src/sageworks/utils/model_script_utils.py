"""Model Script Utilities for SageWorks endpoints"""

import logging
import json

# SageWorks Imports


# Setup the logger
log = logging.getLogger("sageworks")

def fill_template(template: str, params: dict) -> str:
    """
    Fill in the placeholders in the template with the values provided in params.
    Args:
        template (str): The template string with placeholders.
        params (dict): A dictionary with placeholder keys and their corresponding values.
    Returns:
        str: The template with placeholders replaced by their values.
    """
    # Perform the replacements
    for key, value in params.items():
        # Convert value to a string representation suitable for Python code
        if value is None:
            value_str = "None"
        elif isinstance(value, str):
            value_str = f"'{value}'" if key == "target_column" else value
        elif isinstance(value, list):
            value_str = json.dumps(value)
        else:
            value_str = str(value)

        # Replace the placeholder in the template
        placeholder = f"{{{{{key}}}}}"  # Double curly braces to match the template
        template = template.replace(placeholder, value_str)

    # Sanity check to ensure all placeholders were replaced
    if "{{" in template or "}}" in template:
        msg = "Not all template placeholders were replaced. Please check your params."
        log.critical(msg)
        raise ValueError(msg)

    return template


if __name__ == "__main__":
    """Exercise the Model Script Utilities"""

    # Insert test here
    print("my test")
