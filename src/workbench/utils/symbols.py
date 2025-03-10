"""Unicode Symbols for Workbench"""

# A Dictionary/Map of Health Tags to Symbols
health_icons = {
    "failed": "🔴",
    "5xx_errors": "🔴",
    "no_model": "🔴",
    "5xx_errors_min": "🟠",
    "no_endpoint": "🟡",
    "model_type_unknown": "⚪",
    "model_not_found": "⚫",
    "metrics_needed": "🟣",
    "needs_onboard": "🔵",
    "healthy": "🟢",
    "unknown_error": "⚫",
    "no_activity": "➖",
}


def tag_symbols(tag_list: str) -> str:
    """Return the symbols for the given list of tags"
    Args:
        tag_list (str): A string of tags separated by :
    Returns:
        str: The symbol for the given tag (or "" if no symbol)
    """

    # Split the tag list and return the symbol
    symbol_list = []
    tag_list = tag_list.split(":")

    # Special case for empty tag list
    if len(tag_list) == 0 or tag_list[0] == "":
        return health_icons["healthy"]

    # Special case for no_activity with no other tags
    if len(tag_list) == 1 and tag_list[0] == "no_activity":
        return health_icons["healthy"] + health_icons["no_activity"]

    # Loop through tags, get the symbol, and add it to the list
    for tag in tag_list:
        symbol_list.append(health_icons.get(tag, tag))
    return "".join(symbol_list)
