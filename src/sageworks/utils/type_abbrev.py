"""Helper function to convert Athena and Feature Store type names to abbreviations"""
import pandas as pd


def type_abbrev(type_name: str) -> str:
    """Convert a type name to an abbreviation
    Args:
        type_name (str): The type name to convert
    Returns:
        str: The type abbreviation
    """
    type_map = {
        "boolean": "bool",
        "tinyint": "i8",
        "smallint": "i16",
        "integer": "i32",
        "bigint": "i64",
        "float": "f32",
        "double": "f64",
        "decimal": "dec",
        "char": "char",
        "varchar": "var",
        "string": "str",
        "binary": "bin",
        "date": "date",
        "timestamp": "ts",
        "Fractional": "frac",
        "Integral": "int",
        "String": "str",
    }
    return type_map[type_name]


def add_types_to_columns(dataframe: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Add type abbreviations to column names
    Args:
        dataframe (pd.DataFrame): The dataframe to modify
        column_types (dict): A dictionary of column names and types
    Returns:
        pd.DataFrame: The dataframe with type abbreviations added to column names
    """
    # Loop through all the columns and add the type abbreviation
    column_renames = dict()
    for column in dataframe.columns:
        type_name = column_types[column]
        column_renames[column] = f"{column}:{type_abbrev(type_name)}"
    return dataframe.rename(columns=column_renames)
