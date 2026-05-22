"""iPython Utilities.

These helpers are only useful when IPython is installed (via the
``workbench[repl]`` extra). When IPython is missing, the functions return
sensible defaults so callers don't need to special-case the absence —
:func:`is_running_in_ipython` returns ``False`` and
:func:`display_error_and_raise` falls back to raising without the visual
display.
"""

try:
    from IPython import get_ipython
    from IPython.display import display

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False
    get_ipython = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]


def is_running_in_ipython() -> bool:
    """Check if the code is running inside a Jupyter notebook or iPython shell."""
    if not _IPYTHON_AVAILABLE:
        return False
    try:
        return get_ipython() is not None
    except NameError:
        return False


def display_error_and_raise(error_message: str) -> None:
    """Display an error message and raise an exception."""
    if _IPYTHON_AVAILABLE:
        display(error_message)
    raise Exception(error_message)


if __name__ == "__main__":
    """Exercise the iPython Utilities"""
    if is_running_in_ipython():
        display_error_and_raise("This is an error message!")
    else:
        print("Not running in iPython!")
