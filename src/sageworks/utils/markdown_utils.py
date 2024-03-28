"""Markdown Utility/helper methods"""

from sageworks.utils.symbols import health_icons


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
    markdown += "\n\n"  # Add newlines for separation
    return markdown


if __name__ == "__main__":
    """Exercise the Markdown Utilities"""
    from sageworks.api.model import Model

    # Open a model
    model = Model("abalone-regression")
    health_tags = model.get_health_tags()

    # Print the health tag markdown
    print(health_tag_markdown(health_tags))
