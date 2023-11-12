"""Unicode Symbols for Sageworks"""

# A Dictionary of Unicode Symbols
symbols = {
    "red_circle": "🔴",
    "blue_circle": "🔵",
    "green_circle": "🟢",
    "yellow_circle": "🟡",
    "purple_circle": "🟣",
    "white_circle": "⚪",
    "black_circle": "⚫",
    "orange_circle": "🟠",
}


# Tag Symbols
def tag_symbol(tag_list: str) -> str:
    """Return the symbol for the given tag"
    Args:
        tag_list (str): A string of tags separated by :
    Returns:
        str: The symbol for the given tag (or "" if no symbol)
    """

    # Split the tag list and return the symbol
    tag_list = tag_list.split(":")
    if "broken" in tag_list:
        return symbols["red_circle"]
    elif "orphan" in tag_list:
        return symbols["yellow_circle"]
    else:
        return ""
