"""Performance Utilities for Workbench"""

import logging
from functools import wraps
from typing import Callable, Type, Union

# Initialize the Workbench logger
log = logging.getLogger("workbench")


def performance(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
    """Decorator to mark classes or functions as having slow performance."""

    message = f"{cls_or_func.__name__} has slow performance, be careful when/where you use it."

    if isinstance(cls_or_func, type):
        # Class decorator
        original_init = cls_or_func.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            log.warning(message)
            original_init(self, *args, **kwargs)

        cls_or_func.__init__ = new_init
        return cls_or_func
    else:
        # Function/method decorator
        @wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            log.warning(message)
            return cls_or_func(*args, **kwargs)

        return wrapper


if __name__ == "__main__":

    @performance
    class OldClass:
        def __init__(self):
            print("OldClass initialized")

    instance = OldClass()

    class MyClass:
        @performance
        def slow_method(self):
            print("This is a slow method.")

    obj = MyClass()
    obj.slow_method()
