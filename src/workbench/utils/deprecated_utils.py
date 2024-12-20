"""Deprecated Utilities for Workbench"""

import traceback
import sys
import logging
from functools import wraps
from typing import Callable, Type, Union

# Initialize the Workbench logger
log = logging.getLogger("workbench")


def deprecated(version: str, stack_trace: bool = False) -> Callable:
    """Decorator to mark classes or functions as deprecated.

    Args:
        version (str): The version in which the class or function is deprecated.
        stack_trace (bool, optional): Include the call stack. Defaults to False.
    """

    def decorator(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
        message = f"{cls_or_func.__name__} is deprecated and will be removed in version {version}."

        if isinstance(cls_or_func, type):
            # Class decorator
            original_init = cls_or_func.__init__

            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                log.warning(message)
                if stack_trace:
                    trimmed_stack = "".join(traceback.format_stack()[:-1])
                    log.warning("Call stack:\n%s", trimmed_stack)
                original_init(self, *args, **kwargs)

            cls_or_func.__init__ = new_init
            return cls_or_func
        else:
            # Function/method decorator
            @wraps(cls_or_func)
            def wrapper(*args, **kwargs):
                log.warning(message)
                if stack_trace:
                    trimmed_stack = "".join(traceback.format_stack()[:-1])
                    log.warning("Call stack:\n%s", trimmed_stack)
                return cls_or_func(*args, **kwargs)

            return wrapper

    return decorator


def hard_deprecated(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
    """Decorator to mark classes or functions as immediately deprecated and cause termination."""

    message = f"{cls_or_func.__name__} is immediately deprecated and has been removed."

    if isinstance(cls_or_func, type):
        # Class decorator
        original_init = cls_or_func.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            log.critical(message)
            trimmed_stack = "".join(traceback.format_stack()[:-1])
            log.critical("Call stack:\n%s", trimmed_stack)
            sys.exit(1)

        cls_or_func.__init__ = new_init
        return cls_or_func
    else:
        # Function/method decorator
        @wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            log.critical(message)
            trimmed_stack = "".join(traceback.format_stack()[:-1])
            log.critical("Call stack:\n%s", trimmed_stack)
            sys.exit(1)

        return wrapper


if __name__ == "__main__":

    @deprecated(version="0.9")
    class OldClass:
        def __init__(self):
            print("OldClass initialized")

    instance = OldClass()

    class MyClass:
        @deprecated(version="0.9")
        def old_method(self):
            print("This is an old method.")

    obj = MyClass()
    obj.old_method()

    @hard_deprecated
    class ImmediateClass:
        def __init__(self):
            print("This should not be seen.")

    # Uncomment to test (this will exit the script)
    # instance = ImmediateClass()

    class AnotherClass:
        @hard_deprecated
        def immediate_method(self):
            print("This should not be seen either.")

    # Uncomment to test (this will exit the script)
    # obj = AnotherClass()
    # obj.immediate_method()
