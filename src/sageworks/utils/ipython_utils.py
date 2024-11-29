"""iPython Utilities"""

from IPython import get_ipython
from IPython.display import display


def is_running_in_ipython() -> bool:
    """Check if the code is running inside a Jupyter notebook or iPython shell"""
    try:
        return get_ipython() is not None
    except NameError:
        return False


def display_error_and_raise(error_message: str) -> None:
    """Display an error message and raise an exception"""
    display(error_message)
    raise Exception(error_message)


if __name__ == "__main__":
    """Exercise the iPython Utilities"""
    if is_running_in_ipython():
        display_error_and_raise("This is an error message!")
    else:
        print("Not running in iPython!")
