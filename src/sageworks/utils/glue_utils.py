"""AWS Glue Job Utilities"""


def glue_args_to_dict(argv: list[str]) -> dict:
    """Take the Glue Jobs argv list of args and organize them into a dictionary

    Args:
        argv (list[str]): The Glue Jobs argv list of args

    Returns:
        dict: The Glue Jobs argv list of args organized into a dictionary
    """
    it = iter(argv[1:])
    return dict(zip(it, it))
