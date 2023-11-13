"""Unicode Symbols for Sageworks"""

# A Dictionary/Map of Health Tags to Symbols
health_icons = {
    "failed": "🔴",
    "broken": "🔴",
    "no_model": "🟠",
    "no_endpoint": "🟡",
    "orphan": "🟡",
    "mtype_unknown": "🟣",
    "not_ready": "🔵",
    "AOK": "🟢",
    "white": "⚪",
    "black": "⚫",
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
    for tag in tag_list:
        symbol_list.append(health_icons.get(tag, ""))
    return "".join(symbol_list)
