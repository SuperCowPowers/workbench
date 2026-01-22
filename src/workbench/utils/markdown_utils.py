"""Markdown Utility/helper methods"""

import math

from workbench.utils.symbols import health_icons


def health_tag_markdown(health_tags: list[str]) -> str:
    """Internal method to generate the health tag markdown
    Args:
        health_tags (list[str]): A list of health tags
    Returns:
        str: A markdown string
    """
    # If we have no health tags, then add a bullet for healthy
    markdown = "**Health Checks**\n"  # Header for Health Checks

    # If we have no health tags, then add a bullet for healthy
    if not health_tags:
        markdown += f"* Healthy: {health_icons.get('healthy')}\n\n"
        return markdown

    # Special case for no_activity with no other tags
    if len(health_tags) == 1 and health_tags[0] == "no_activity":
        markdown += f"* Healthy: {health_icons.get('healthy')}\n"
        markdown += f"* No Activity: {health_icons.get('no_activity')}\n\n"
        return markdown

    # If we have health tags, then add a bullet for each tag
    markdown += "\n".join(f"* {tag}: {health_icons.get(tag, '')}" for tag in health_tags)
    markdown += "\n\n"
    return markdown


def tag_styling(tags: list, tag_dict: dict) -> str:
    """Generate a Markdown string with styled spans for tags.

    Args:
        tags (list): List of tags to style.
        tag_dict (dict): Dictionary mapping substrings to CSS classes.

    Returns:
        str: Markdown string with styled spans.
    """
    styled_tags = []
    for tag in tags:
        # Check for a matching class in the dictionary
        class_name = next((class_name for substring, class_name in tag_dict.items() if substring in tag.lower()), None)
        if class_name:
            # Add span with the matched class
            styled_tags.append(f'<span class="{class_name}" style="padding:0" children="{tag}"/>')
        else:
            # Add plain tag if no class matches
            styled_tags.append(tag)
    # Join styled tags with commas
    return ", ".join(styled_tags)


def tags_to_markdown(tags: str) -> str:
    """Convert tags to a Markdown string.

    Args:
        tags (str): Deliminator-separated string of tags.

    Returns:
        str: Markdown string with tags.
    """
    # Split the tags (this needs to be refactored to use a list)
    tag_list = tags.split("::")

    # Separate items with ":" from those without
    with_colon = [item for item in tag_list if ":" in item]
    without_colon = [item for item in tag_list if ":" not in item]
    ordered_tag_list = with_colon
    if without_colon:
        without_colon = ", ".join(without_colon)
        ordered_tag_list += [without_colon]
    tag_markdown = "**Tags:**\n"
    markdown_tag_list = []
    for tag in ordered_tag_list:
        if ":" in tag:
            markdown_tag_list.append(f"*{tag.split(':')[0]}:*{tag.split(':')[1]}")
        else:
            markdown_tag_list.append(tag)
    tag_markdown += ", ".join(markdown_tag_list)
    return tag_markdown


def dict_to_markdown(data: dict, title: str = None) -> str:
    """Convert a dictionary to pretty Markdown format.
    Args:
        data (dict): Dictionary to convert
        title (str, optional): Optional title for the dictionary
    Returns:
        str: Markdown formatted string
    """

    def _convert_dict(data: dict, indent_level: int = 0) -> str:
        markdown = ""
        indent_str = "    " * indent_level  # 4 spaces per level

        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary
                markdown += f"{indent_str}* *{key}:*\n"
                markdown += _convert_dict(value, indent_level + 1)
            elif isinstance(value, list):
                # Check if it's a list of dictionaries
                if value and all(isinstance(item, dict) for item in value):
                    # List of dictionaries
                    markdown += f"{indent_str}* *{key}:*\n"
                    for i, dict_item in enumerate(value):
                        markdown += f"{indent_str}    * Item {i + 1}:\n"
                        markdown += _convert_dict(dict_item, indent_level + 2)
                else:
                    # Regular list values
                    markdown += f"{indent_str}* *{key}:* {', '.join(map(str, value))}\n"
            else:
                # Simple key-value pair
                markdown += f"{indent_str}* *{key}:* {value}\n"

        return markdown

    # Add title if provided (always at root level, no indentation)
    result = ""
    if title:
        result += f"**{title}**\n\n"
    result += _convert_dict(data)
    return result


def dict_to_collapsible_html(data: dict, title: str = None, collapse_all: bool = False) -> str:
    """Convert a dictionary to collapsible HTML format.
    Args:
        data (dict): Dictionary to convert
        title (str, optional): Optional title for the dictionary
        collapse_all (bool): Whether to collapse all sections by default. Defaults to False.
    Returns:
        str: HTML formatted string with collapsible sections
    """

    def _convert_dict_html(data: dict, indent_level: int = 0) -> str:
        html = ""
        indent_style = f'style="margin-left: {indent_level * 20}px;"' if indent_level > 0 else ""
        # Leaf nodes get slightly less indentation to align with content
        leaf_indent_style = (
            f'style="margin-left: {max(0, indent_level - 1) * 10 + 10}px;"'
            if indent_level > 0
            else 'style="margin-left: 10px;"'
        )

        # Determine if details should be open or closed
        open_attr = "" if collapse_all else "open"

        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<details {open_attr} {indent_style}><summary><b>{key}</b></summary>\n"
                html += _convert_dict_html(value, indent_level + 1)
                html += "</details>\n"
            elif isinstance(value, list):
                if value and all(isinstance(item, dict) for item in value):
                    html += f"<details {open_attr} {indent_style}><summary><b>{key}</b></summary>\n"
                    for i, dict_item in enumerate(value):
                        html += f'<details {open_attr} style="margin-left: {(indent_level + 1) * 20}px;">'
                        html += f"<summary>Item {i + 1}</summary>\n"
                        html += _convert_dict_html(dict_item, indent_level + 2)
                        html += "</details>\n"
                    html += "</details>\n"
                else:
                    # Leaf node - use bullet with reduced indentation
                    html += f"<div {leaf_indent_style}>• <em>{key}:</em> {', '.join(map(str, value))}</div>\n"
            else:
                # Leaf node - use bullet with reduced indentation
                html += f"<div {leaf_indent_style}>• <em>{key}:</em> {value}</div>\n"
        return html

    # Add title and content
    result = ""
    open_attr = "" if collapse_all else "open"

    if title:
        result += f"<details {open_attr}><summary><strong>{title}</strong></summary>\n"
        result += _convert_dict_html(data, 1)  # Start with indent level 1 under title
        result += "</details>\n"
    else:
        result += _convert_dict_html(data, 0)  # Start with no indent
    return result


def df_to_html_table(df, round_digits: int = 2, margin_bottom: int = 30) -> str:
    """Convert a DataFrame to a compact styled HTML table (horizontal layout).

    Args:
        df: DataFrame with metrics (can be single or multi-row)
        round_digits: Number of decimal places to round to (default: 2)
        margin_bottom: Bottom margin in pixels (default: 30)

    Returns:
        str: HTML table string
    """
    # Handle index: reset if named (keeps as column), otherwise drop
    if df.index.name:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    # Round numeric columns
    df = df.round(round_digits)

    # Table styles
    container_style = f"display: flex; justify-content: center; margin-top: 10px; margin-bottom: {margin_bottom}px;"
    table_style = "border-collapse: collapse; width: 100%; font-size: 15px;"
    header_style = (
        "background: linear-gradient(to bottom, #4a4a4a 0%, #2d2d2d 100%); "
        "color: white; padding: 4px 8px; text-align: center;"
    )
    cell_style = "padding: 3px 8px; text-align: center; border-bottom: 1px solid #444;"

    # Build the HTML table (wrapped in centered container)
    html = f'<div style="{container_style}"><table style="{table_style}">'

    # Header row
    html += "<tr>"
    for col in df.columns:
        html += f'<th style="{header_style}">{col}</th>'
    html += "</tr>"

    # Data rows
    for _, row in df.iterrows():
        html += "<tr>"
        for val in row:
            # Format value: integers without decimal, floats rounded
            if isinstance(val, float):
                if math.isnan(val):
                    formatted_val = "NaN"
                elif val == int(val):
                    formatted_val = int(val)
                else:
                    formatted_val = round(val, round_digits)
            else:
                formatted_val = val
            html += f'<td style="{cell_style}">{formatted_val}</td>'
        html += "</tr>"

    html += "</table></div>"
    return html


if __name__ == "__main__":
    """Exercise the Markdown Utilities"""
    from workbench.api.model import Model

    # Open a model
    model = Model("abalone-regression")
    health_tags = model.get_health_tags()

    # Print the health tag markdown
    print(health_tag_markdown(health_tags))

    # Print the tag markdown
    tag_str = "tag1::tag2::tag3::key1:value1::key2:value2"
    print(tags_to_markdown(tag_str))

    # Print the tag markdown
    tag_str = "key1:value1::key2:value2::key3:value3"
    print(tags_to_markdown(tag_str))

    # Print the dict to markdown
    sample_dict = {
        "key1": "value1",
        "key2": {"subkey1": "subvalue1", "subkey2": ["item1", "item2"]},
        "key3": ["list_item1", "list_item2"],
        "key4": [{"nested_key": "nested_value"}, {"another_key": "another_value"}],
    }
    print(dict_to_markdown(sample_dict, title="Sample Dictionary"))
    print(dict_to_collapsible_html(sample_dict, title="Sample Dictionary"))
